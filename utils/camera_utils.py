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
from concurrent.futures import ThreadPoolExecutor
from scene.cameras import Camera
from tqdm import tqdm
import numpy as np
from utils.graphics_utils import fov2focal
import sys
from PIL import Image
import cv2

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.width, cam_info.height
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global_down = orig_w / 1600
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    print(f"scale {float(global_down) * float(resolution_scale)}")
                    WARNED = True
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    sys.stdout.write('\r')
    sys.stdout.write("load camera {}".format(id))
    sys.stdout.flush()

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, depth_params=cam_info.depth_params,
                  image_width=resolution[0], image_height=resolution[1],
                  image_path=cam_info.image_path, depth_path=cam_info.depth_path,
                  image_name=cam_info.image_name, uid=cam_info.global_id, 
                  preload_img=args.preload_img and not args.stream_views,
                  ncc_scale=args.ncc_scale,
                  data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    if args.preload_img and not args.stream_views and camera_list:
        # Decoding dominates loading, so views are decoded in parallel threads.
        with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 1)) as executor:
            list(tqdm(executor.map(Camera.load_view, camera_list), total=len(camera_list), desc="Preloading views"))

    return camera_list

class ViewPrefetcher:
    """Decode upcoming training views in background threads.

    Only the scheduled views and the views added by prefetch are decoded or held on
    the host, so memory does not grow with the number of views.
    """

    def __init__(self, workers=8, window=16):
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.window = window
        self.futures = {}

    def schedule(self, cameras):
        """Decode the first window cameras, the next views to be used, and drop all others."""
        cameras = list(cameras)[:self.window]
        keep = {id(camera) for camera in cameras}
        for key in [key for key in self.futures if key not in keep]:
            self.futures.pop(key).cancel()
        for camera in cameras:
            self.prefetch(camera)

    def prefetch(self, camera):
        """Start decoding camera unless it is already pending."""
        if id(camera) not in self.futures:
            self.futures[id(camera)] = self.executor.submit(camera.decode)

    def get_image(self, camera):
        """Return the images of camera on the GPU, as Camera.get_image does."""
        future = self.futures.pop(id(camera), None)
        view = camera.decode() if future is None else future.result()
        return camera.get_image(view)

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
