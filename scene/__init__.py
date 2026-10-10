#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import numpy as np
import torch
from tqdm import tqdm

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, load_ply_path=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.source_path = args.source_path

        
        if load_iteration and not load_ply_path:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        # Only the first process writes scene metadata under multi-GPU training.
        is_primary = int(os.environ.get("RANK", "0")) == 0

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter and is_primary:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print(f"cameras_extent {self.cameras_extent}")

        self.multi_view_num = args.multi_view_num
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            
            print("computing nearest_id")
            # Rows of 64 cameras at a time keep the pairwise distances and
            # angles off the GPU and bounded in memory for long captures.
            train_cameras = self.train_cameras[resolution_scale]
            camera_count = len(train_cameras)
            camera_centers = np.empty((camera_count, 3), dtype=np.float32)
            center_rays = np.empty((camera_count, 3), dtype=np.float32)
            for camera_id, cur_cam in enumerate(train_cameras):
                camera_centers[camera_id] = cur_cam.camera_center.detach().cpu().numpy().astype(np.float32)
                ray = np.array([0.0, 0.0, 1.0], dtype=np.float32) @ np.asarray(cur_cam.R, dtype=np.float32).T
                center_rays[camera_id] = ray / float(np.linalg.norm(ray) + 1e-20)

            file = open(os.path.join(self.model_path, "multi_view.json"), 'w') if is_primary else None
            batch_size = 64
            for batch_start in tqdm(range(0, camera_count, batch_size), desc="computing nearest_id", disable=not is_primary):
                batch_end = min(batch_start + batch_size, camera_count)
                diss_batch = np.linalg.norm(camera_centers[None, :, :] - camera_centers[batch_start:batch_end, None, :], axis=-1)
                dot_batch = (center_rays[None, :, :] * center_rays[batch_start:batch_end, None, :]).sum(axis=-1)
                np.clip(dot_batch, -1.0, 1.0, out=dot_batch)
                angles_batch = np.arccos(dot_batch) * (180.0 / np.pi)
                for row_id in range(batch_end - batch_start):
                    cur_cam = train_cameras[batch_start + row_id]
                    diss_row = diss_batch[row_id]
                    angles_row = angles_batch[row_id]
                    sorted_indices = np.lexsort((angles_row, diss_row))
                    mask = (angles_row[sorted_indices] < args.multi_view_max_angle) & \
                        (diss_row[sorted_indices] > args.multi_view_min_dis) & \
                        (diss_row[sorted_indices] < args.multi_view_max_dis)
                    sorted_indices = sorted_indices[mask]
                    multi_view_num = min(self.multi_view_num, len(sorted_indices))
                    json_d = {'ref_name' : cur_cam.image_name, 'nearest_name': []}
                    for index in sorted_indices[:multi_view_num]:
                        index_int = int(index)
                        cur_cam.nearest_id.append(index_int)
                        cur_cam.nearest_names.append(train_cameras[index_int].image_name)
                        json_d["nearest_name"].append(train_cameras[index_int].image_name)
                    if file is not None:
                        file.write(json.dumps(json_d, separators=(',', ':')))
                        file.write('\n')
            if file is not None:
                file.close()
        if load_ply_path:
            self.gaussians.load_ply(load_ply_path)
        else:
            if self.loaded_iter:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "point_cloud.ply"))
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration, mask=None):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"), mask)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]