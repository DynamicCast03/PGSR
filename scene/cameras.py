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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal, getProjectionMatrixCenterShift
import copy
from PIL import Image
from utils.general_utils import PILtoTorch
import os, cv2
import torch.nn.functional as F

def dilate(bin_img, ksize=6):
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out

def erode(bin_img, ksize=12):
    out = 1 - dilate(1 - bin_img, ksize)
    return out

def process_image(image_path, depth_path, resolution, ncc_scale):
    image = Image.open(image_path)
    if depth_path != "":
        try:
            if depth_path.endswith('.npz'):
                invdepthmap = np.load(depth_path)['depth']
            elif depth_path.endswith('.png'):
                depth_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                invdepthmap = depth_mm / 1000.0
            else:
                raise ValueError(f"Unsupported depth format: {depth_path}")
        except FileNotFoundError:
            print(f"Error: The depth file at path '{depth_path}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the image file '{depth_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred when trying to read depth at {depth_path}: {e}")
            raise
    else:
        invdepthmap = None
        
    if len(image.split()) > 3:
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(image.split()[3], resolution)
        gt_image = resized_image_rgb
        if ncc_scale != 1.0:
            ncc_resolution = (int(resolution[0]/ncc_scale), int(resolution[1]/ncc_scale))
            resized_image_rgb = torch.cat([PILtoTorch(im, ncc_resolution) for im in image.split()[:3]], dim=0)
    else:
        resized_image_rgb = PILtoTorch(image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb
        if ncc_scale != 1.0:
            ncc_resolution = (int(resolution[0]/ncc_scale), int(resolution[1]/ncc_scale))
            resized_image_rgb = PILtoTorch(image, ncc_resolution)
    gray_image = (0.299 * resized_image_rgb[0] + 0.587 * resized_image_rgb[1] + 0.114 * resized_image_rgb[2])[None]
    depth_image = cv2.resize(invdepthmap, resolution)
    depth_image = torch.Tensor(depth_image).unsqueeze(0)
    # depth_image = PILtoTorch(invdepthmap, resolution)
    return gt_image, gray_image, loaded_mask, depth_image

def decode_view(image_path, depth_path, resolution, ncc_scale):
    """Decode a view as process_image does, keeping RGB as (3, H, W) uint8.

    Dividing the RGB by 255 gives the gt_image of process_image. The NCC gray
    image is returned only when ncc_scale differs from 1, since it otherwise
    follows from the RGB.
    """
    image = Image.open(image_path)
    bands = image.split()
    if len(bands) > 3:
        rgb = np.stack([np.array(band.resize(resolution)) for band in bands[:3]], axis=0)
        loaded_mask = PILtoTorch(bands[3], resolution)
    else:
        rgb = np.array(image.resize(resolution)).transpose(2, 0, 1)
        loaded_mask = None
    ncc_gray = None
    if ncc_scale != 1.0:
        ncc_resolution = (int(resolution[0]/ncc_scale), int(resolution[1]/ncc_scale))
        if len(bands) > 3:
            ncc_rgb = torch.cat([PILtoTorch(im, ncc_resolution) for im in bands[:3]], dim=0)
        else:
            ncc_rgb = PILtoTorch(image, ncc_resolution)
        ncc_gray = (0.299 * ncc_rgb[0] + 0.587 * ncc_rgb[1] + 0.114 * ncc_rgb[2])[None]
    depth_image = None
    if depth_path != "":
        if depth_path.endswith('.npz'):
            invdepthmap = np.load(depth_path)['depth']
        elif depth_path.endswith('.png'):
            invdepthmap = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        else:
            raise ValueError(f"Unsupported depth format: {depth_path}")
        depth_image = torch.Tensor(cv2.resize(invdepthmap, resolution)).unsqueeze(0)
    return torch.from_numpy(np.ascontiguousarray(rgb)), loaded_mask, ncc_gray, depth_image

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, depth_params,
                 image_width, image_height,
                 image_path, depth_path, image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                 ncc_scale=1.0,
                 preload_img=False, data_device = "cuda"
                 ):
        super(Camera, self).__init__()
        self.uid = uid
        self.nearest_id = []
        self.nearest_names = []
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_path = image_path
        self.image_width = image_width
        self.image_height = image_height
        self.resolution = (image_width, image_height)
        self.Fx = fov2focal(FoVx, self.image_width)
        self.Fy = fov2focal(FoVy, self.image_height)
        self.Cx = 0.5 * self.image_width
        self.Cy = 0.5 * self.image_height
        self.depth_path = depth_path
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.mask = None
        self.preload_img = preload_img
        self.ncc_scale = ncc_scale
        self.depth_params = depth_params
        # Preloaded views, filled by load_view: (3, H, W) uint8 RGB, the NCC
        # gray image when ncc_scale differs from 1, and the (1, H, W) depth.
        self.rgb_uint8, self.ncc_gray, self.invdepthmap = None, None, None
        self.depth_reliable = False

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.plane_mask, self.non_plane_mask = None, None

    def load_view(self):
        """Decode this view once and keep it on data_device for get_image."""
        rgb, mask, ncc_gray, inv_depth_image = decode_view(self.image_path, self.depth_path, self.resolution, self.ncc_scale)
        invdepthmap = None
        depth_reliable = False
        if inv_depth_image is not None:
            invdepthmap = inv_depth_image
            invdepthmap[invdepthmap < 0] = 0
            depth_reliable = True
            if self.depth_params is not None:
                if self.depth_params["scale"] < 0.2 * self.depth_params["med_scale"] or self.depth_params["scale"] > 5 * self.depth_params["med_scale"]:
                    depth_reliable = False
                if self.depth_params["scale"] > 0:
                    invdepthmap = invdepthmap * self.depth_params["scale"] + self.depth_params["offset"]
            invdepthmap = invdepthmap.to(self.data_device)
        self.rgb_uint8 = rgb.to(self.data_device)
        self.ncc_gray = None if ncc_gray is None else ncc_gray.to(self.data_device)
        self.mask = mask
        self.invdepthmap = invdepthmap
        self.depth_reliable = depth_reliable

    def get_image(self):
        if self.preload_img:
            if self.rgb_uint8 is None:
                self.load_view()
            gt_image = self.rgb_uint8.cuda() / 255.0
            if self.ncc_gray is None:
                gray_image = (0.299 * gt_image[0] + 0.587 * gt_image[1] + 0.114 * gt_image[2])[None]
            else:
                gray_image = self.ncc_gray.cuda()
            invdepthmap = None if self.invdepthmap is None else self.invdepthmap.cuda()
            return gt_image, gray_image, invdepthmap
        else:
            gt_image, gray_image, _, inv_depth_image = process_image(self.image_path, self.depth_path, self.resolution, self.ncc_scale)
            if inv_depth_image is not None:
                invdepthmap = inv_depth_image
                # invdepthmap[invdepthmap < 0] = 0
                # depth_reliable = True
                # if self.depth_params is not None:
                #     if self.depth_params["scale"] < 0.2 * self.depth_params["med_scale"] or self.depth_params["scale"] > 5 * self.depth_params["med_scale"]:
                #         depth_reliable = False
                    
                #     if self.depth_params["scale"] > 0:
                #         invdepthmap = invdepthmap * self.depth_params["scale"] + self.depth_params["offset"]
                        
                # if not depth_reliable:
                #     invdepthmap = None
                # else:
                #     invdepthmap = invdepthmap.cuda()
            return gt_image.cuda(), gray_image.cuda(), invdepthmap

    def get_calib_matrix_nerf(self, scale=1.0):
        intrinsic_matrix = torch.tensor([[self.Fx/scale, 0, self.Cx/scale], [0, self.Fy/scale, self.Cy/scale], [0, 0, 1]]).float()
        extrinsic_matrix = self.world_view_transform.transpose(0,1).contiguous() # cam2world
        return intrinsic_matrix, extrinsic_matrix
    
    def get_rays(self, scale=1.0):
        W, H = int(self.image_width/scale), int(self.image_height/scale)
        ix, iy = torch.meshgrid(
            torch.arange(W), torch.arange(H), indexing='xy')
        rays_d = torch.stack(
                    [(ix-self.Cx/scale) / self.Fx * scale,
                    (iy-self.Cy/scale) / self.Fy * scale,
                    torch.ones_like(ix)], -1).float().cuda()
        return rays_d
    
    def get_k(self, scale=1.0):
        K = torch.tensor([[self.Fx / scale, 0, self.Cx / scale],
                        [0, self.Fy / scale, self.Cy / scale],
                        [0, 0, 1]]).cuda()
        return K
    
    def get_inv_k(self, scale=1.0):
        K_T = torch.tensor([[scale/self.Fx, 0, -self.Cx/self.Fx],
                            [0, scale/self.Fy, -self.Cy/self.Fy],
                            [0, 0, 1]]).cuda()
        return K_T

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

def sample_cam(cam_l: Camera, cam_r: Camera):
    cam = copy.copy(cam_l)

    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam_l.R.transpose()
    Rt[:3, 3] = cam_l.T
    Rt[3, 3] = 1.0

    Rt2 = np.zeros((4, 4))
    Rt2[:3, :3] = cam_r.R.transpose()
    Rt2[:3, 3] = cam_r.T
    Rt2[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    C2W2 = np.linalg.inv(Rt2)
    w = np.random.rand()
    pose_c2w_at_unseen =  w * C2W + (1 - w) * C2W2
    Rt = np.linalg.inv(pose_c2w_at_unseen)
    cam.R = Rt[:3, :3]
    cam.T = Rt[:3, 3]

    cam.world_view_transform = torch.tensor(getWorld2View2(cam.R, cam.T, cam.trans, cam.scale)).transpose(0, 1).cuda()
    cam.projection_matrix = getProjectionMatrix(znear=cam.znear, zfar=cam.zfar, fovX=cam.FoVx, fovY=cam.FoVy).transpose(0,1).cuda()
    cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
    cam.camera_center = cam.world_view_transform.inverse()[3, :3]
    return cam
