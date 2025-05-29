import torch
from torch import nn
import numpy as np
import scipy.ndimage
import cv2
from gaussian.utils.graphics_utils import getProjectionMatrix2, getWorld2View2, focal2fov
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

class Camera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        depth,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        device="cuda:0",
        normal=None,
        bg=[0,0,0]
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.device = device

        T = torch.eye(4, device=device)
        self.R = T[:3, :3]
        self.T = T[:3, 3]
        self.R_gt = gt_T[:3, :3]
        self.T_gt = gt_T[:3, 3]

        self.original_image = color.to(self.device)
        if depth is not None:
            self.depth = depth.to(self.device)
        if normal is not None:
            self.normal = normal.to(self.device)
        self.bg = torch.tensor(bg, dtype=torch.float32).to(self.device)
        self.grad_mask = None

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width

        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        
        # for deblur the camera motion blur
        self.weight_this = nn.Parameter(
            torch.tensor([0.5], requires_grad=True, device=device)
        )
        self.weight_blur = nn.Parameter(
            torch.tensor([0.5], requires_grad=True, device=device)
        )
        self.blur_tran_x = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.blur_tran_y = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        self.projection_matrix = projection_matrix.to(device=device)
        
        if normal is None and depth is not None:
            self.normal = self.depth_to_normal()
        
        
    def set_GSRasterization(self, scaling_modifier=1.0):
        # Set up rasterization configuration
        tanfovx = math.tan(self.FoVx * 0.5)
        tanfovy = math.tan(self.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(self.image_height),
            image_width=int(self.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg,
            scale_modifier=scaling_modifier,
            viewmatrix=self.world_view_transform,
            projmatrix=self.full_proj_transform,
            projmatrix_raw=self.projection_matrix,
            sh_degree=0,
            campos=self.camera_center
        )
        
        self.rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            
    def depths_to_points(self, depth=None, world_frame=False):
        W, H = self.image_width, self.image_height
        fx = W / (2 * math.tan(self.FoVx / 2.))
        fy = H / (2 * math.tan(self.FoVy / 2.))
        intrins = torch.tensor([[fx, 0., W/2.], [0., fy, H/2.], [0., 0., 1.0]]).float().cuda()
        grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float() + 0.5, torch.arange(H, device='cuda').float() + 0.5, indexing='xy')
        points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
        if world_frame:
            c2w = (self.world_view_transform.T).inverse()
            rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
            rays_o = c2w[:3,3]
            if depth is not None:
                points = depth.reshape(-1, 1) * rays_d + rays_o
            else:
                points = self.depth.reshape(-1, 1) * rays_d + rays_o
        else:
            rays_d = points @ intrins.inverse().T
            if depth is not None:
                points = depth.reshape(-1, 1) * rays_d
            else:
                points = self.depth.reshape(-1, 1) * rays_d
        return points
    
    
    # link to camera to forbid repeated compulation
    def depth_to_normal(self, world_frame=False):
        # # bilateral smooth 
        # depth = self.depth.cpu().numpy().astype(np.float32)
        # d = 9
        # sigma_color = 555  
        # sigma_space = 555  
        # depth_smoothed = cv2.bilateralFilter(depth, d, sigma_color, sigma_space)
        # depth_smoothed = torch.tensor(depth_smoothed).to(self.depth.device)
        # points = self.depths_to_points(depth=depth_smoothed).reshape(*self.depth.shape, 3)
        points = self.depths_to_points().reshape(*self.depth.shape, 3)
        normal_map = torch.zeros_like(points)
        dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
        dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
        normal_map[1:-1, 1:-1, :] = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
        return normal_map



    @staticmethod
    def init_from_tracking(color, depth, pose, idx, projection_matrix, K, tstamp=None, normal=None, bg=[0,0,0]):
        cam = Camera(
            idx,
            color,
            depth,
            pose,
            projection_matrix,
            K[0],
            K[1],
            K[2],
            K[3],
            focal2fov(K[0], K[-2]),
            focal2fov(K[1], K[-1]),
            K[-1],
            K[-2],
            normal=normal,
            bg=bg)
        cam.R = pose[:3, :3]
        cam.T = pose[:3, 3]
        cam.tstamp = tstamp
        cam.set_GSRasterization()
        return cam
    
    @staticmethod
    def init_from_dataset(dataset, idx, projection_matrix):
        gt_color, gt_depth, gt_pose = dataset[idx]
        return Camera(
            idx,
            gt_color,
            gt_depth,
            None,
            gt_pose,
            projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.fovx,
            dataset.fovy,
            dataset.height,
            dataset.width,
            # device=dataset.device,
        )

    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        ).transpose(0, 1)
        return Camera(
            uid, None, None, None, T, projection_matrix, fx, fy, cx, cy, FoVx, FoVy, H, W
        )

    @property
    def world_view_transform(self):
        return getWorld2View2(self.R, self.T).transpose(0, 1)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    def update_RT(self, R, t):
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)

    def clean(self):
        self.original_image = None
        self.depth = None
        self.grad_mask = None

        self.cam_rot_delta = None
        self.cam_trans_delta = None
