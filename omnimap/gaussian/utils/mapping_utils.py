import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from gaussian.utils.loss_utils import ssim
import cv2
import torch.nn.functional as F

def to_se3_vec(pose_mat):
    quat = R.from_matrix(pose_mat[:3, :3]).as_quat()
    return np.hstack((pose_mat[:3, 3], quat))


def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm


def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )


def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V


def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def smoothness_loss(normal_map):
    normal_map = normal_map.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    grad_x = normal_map[:, :, :, 1:] - normal_map[:, :, :, :-1]  # Horizontals
    grad_y = normal_map[:, :, 1:, :] - normal_map[:, :, :-1, :]  # Vertical
    smooth_loss = grad_x.mean() + grad_y.mean()
    return smooth_loss



def depths_to_points(view, depthmap, world_frame):
    W, H = view.image_width, view.image_height
    # fx = W / (2 * math.tan(view.FoVx / 2.))
    # fy = H / (2 * math.tan(view.FoVy / 2.))
    fx = view.fx
    fy = view.fy
    cx = view.cx
    cy = view.cy
    intrins = torch.tensor([[fx, 0., W/2.], [0., fy, H/2.], [0., 0., 1.0]]).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float() + 0.5, torch.arange(H, device='cuda').float() + 0.5, indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    if world_frame:
        c2w = (view.world_view_transform.T).inverse()
        rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
        rays_o = c2w[:3,3]
        points = depthmap.reshape(-1, 1) * rays_d + rays_o
    else:
        rays_d = points @ intrins.inverse().T
        points = depthmap.reshape(-1, 1) * rays_d
    # debug vis
    # import open3d as o3d
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points.clone().detach().cpu().numpy())
    # o3d.visualization.draw_geometries([pcd])
    return points



def depth_to_normal(view, depth, world_frame=False):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(view, depth, world_frame).reshape(*depth.shape[1:], 3)
    normal_map = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map[1:-1, 1:-1, :] = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    return normal_map, points

def compute_normals(points, block_size=10):
    H, W, _ = points.shape
    points = points.permute(2, 0, 1).unsqueeze(0)
    invalid_mask = (points[:, 2, :, :] == 0)
    kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device='cuda').view(1, 1, 3, 3)
    kernel = kernel.repeat(3, 1, 1, 1)
    grad_x = torch.nn.functional.conv2d(points, kernel, padding=1, groups=3)
    grad_y = torch.nn.functional.conv2d(points, kernel.transpose(2, 3), padding=1, groups=3)
    normals = torch.cross(grad_x, grad_y, dim=1)
    normals = torch.nn.functional.normalize(normals, dim=1)
    normals = torch.where(invalid_mask.expand_as(normals), torch.zeros_like(normals), normals)
    normals = torch.nn.functional.avg_pool2d(normals, kernel_size=block_size, stride=block_size)
    normals = torch.nn.functional.normalize(normals, dim=1)
    return normals.squeeze(0).permute(1, 2, 0)

def depth_to_block_normals(view, depth, world_frame=False, block_size=3):
    points = depths_to_points(view, depth, world_frame).reshape(*depth.shape[1:], 3)
    block_normals = compute_normals(points, block_size)
    return block_normals


def compute_normal_mask(normal_map, threshold=0.2, kernel_size=11):
    normal_map = normal_map.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    # average pooling
    avg_normal = F.avg_pool2d(normal_map, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    # diff with average value
    dot_product = (normal_map * avg_normal).sum(dim=1, keepdim=True)  # (1, 1, H, W)
    angle_diff = torch.acos(torch.clamp(dot_product, -1.0, 1.0))
    mask = (angle_diff > threshold).squeeze(0).squeeze(0)
    return mask

# def gaussian_kernel(size, sigma):
#     """Generate a 2D Gaussian kernel."""
#     # Create 1D Gaussian kernel
#     x = torch.arange(-size//2 + 1, size//2 + 1, dtype=torch.float32)
#     kernel_1d = torch.exp(-x**2 / (2 * sigma**2))  # Gaussian function
#     kernel_1d = kernel_1d / kernel_1d.sum()  # Normalize
#     # Create 2D Gaussian kernel by outer product of the 1D kernel with itself
#     kernel_2d = kernel_1d.view(1, -1) * kernel_1d.view(-1, 1)
#     # Reshape to be 4D for the conv2d operation
#     kernel_2d = kernel_2d.view(1, 1, size, size).to(torch.float32)
#     return kernel_2d

# def depth_to_normal_more(view, depth, world_frame=False):
#     """
#         view: view camera
#         depth: depthmap 
#     """
#     points = depths_to_points(view, depth, world_frame).reshape(*depth.shape[1:], 3)
#     normal_map = torch.zeros_like(points)
#     dx03 = torch.cat([points[8:, 4:-4] - points[:-8, 4:-4]], dim=0)
#     # dx02 = torch.cat([points[3:-1, 2:-2] - points[:-4, 2:-2]], dim=0)
#     # dx13 = torch.cat([points[4:, 2:-2] - points[1:-3, 2:-2]], dim=0)
#     # dx12 = torch.cat([points[3:-1, 2:-2] - points[1:-3, 2:-2]], dim=0)
#     # dx = dx03+dx02+dx13+dx12
#     dx = dx03
#     dy03 = torch.cat([points[4:-4, 8:] - points[4:-4, :-8]], dim=1)
#     # dy02 = torch.cat([points[2:-2, 3:-1] - points[2:-2, :-4]], dim=1)
#     # dy13 = torch.cat([points[2:-2, 4:] - points[2:-2, 1:-3]], dim=1)
#     # dy12 = torch.cat([points[2:-2, 3:-1] - points[2:-2, 1:-3]], dim=1)
#     # dy = dy03+dy02+dy13+dy12
#     dy = dy03
#     normal_map[4:-4, 4:-4, :] = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
#     return normal_map, points


# import open3d as o3d
# def depth_to_normal_pc(view, depth, world_frame=False):
#     """
#         view: view camera
#         depth: depthmap 
#     """
#     points = depths_to_points(view, depth, world_frame).reshape(*depth.shape[1:], 3)
#     normal_map = torch.zeros_like(points)
#     points_np = points.cpu().numpy().reshape(-1, 3) 
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points_np)
#     # pcd = pcd.voxel_down_sample(0.03)
#     pcd.estimate_normals(
#         search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
#     )
#     o3d.visualization.draw_geometries([pcd], point_show_normal=True)
#     normals = np.asarray(pcd.normals)
#     normal_map = torch.tensor(normals).reshape(*depth.shape[1:], 3).float()
#     return normal_map, points

def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
    mask_v, mask_h = image_gradient_mask(depth)
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
    depth_grad_v, depth_grad_h = image_gradient(depth)
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    w_h = torch.exp(-10 * gray_grad_h**2)
    w_v = torch.exp(-10 * gray_grad_v**2)
    err = (w_h * torch.abs(depth_grad_h)).mean() + (
        w_v * torch.abs(depth_grad_v)
    ).mean()
    return err


def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False):
    if config["Training"]["monocular"]:
        return get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    return get_loss_tracking_rgbd(config, image, depth, opacity, viewpoint)


def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
    l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    return l1.mean()


def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()


def get_loss_depth_normal(depth_mean, viewpoint, current_id=None, corr=False):
    
    gt_depth = viewpoint.depth
    mask = gt_depth < 0.01
    
    gt_normal = viewpoint.normal
    
    # mask = gt_normal.norm(dim=-1) < 0.1
    
    pre_normal, pts = depth_to_normal(viewpoint, depth_mean, world_frame=False)
    
    # # good mask
    # mask_no = compute_normal_mask(gt_normal, threshold=0.7, kernel_size=9)
    # mask = mask | mask_no
    
    # pre_normal = depth_to_block_normals(viewpoint, depth_mean, block_size=3)
    # gt_normal = depth_to_block_normals(viewpoint, viewpoint.depth[None].to(torch.float32), block_size=3)

    # pre_normal, _ = depth_to_normal_more(viewpoint, depth_mean, world_frame=False)
    # gt_normal, _ = depth_to_normal_more(viewpoint, viewpoint.depth[None], world_frame=False)

    # 计算法线损失
    normal_error = 1 - (pre_normal * gt_normal).sum(dim=-1)
    normal_error[mask] = 0

    # # debug vis normal
    # if current_id is not None and current_id >5:
    #     gt_normal_copy = gt_normal.clone()
    #     gt_normal_copy[mask] = torch.tensor([0.,0.,0.], dtype=gt_normal.dtype, device=gt_normal.device)
    #     import matplotlib.pyplot as plt
    #     fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    #     prior_normal_show = (((pre_normal.clone()+1.0)*0.5).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
    #     gt_normal_show = (((gt_normal_copy+1.0)*0.5).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
    #     gt_normal_show_raw = (((gt_normal+1.0)*0.5).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
    #     axes[0].imshow(prior_normal_show)
    #     axes[1].imshow(gt_normal_show)
    #     axes[2].imshow(gt_normal_show_raw)
    #     plt.show()

    # normal smooth loss
    # smooth_loss = smoothness_loss(pre_normal)*10
    if corr:
        return normal_error.mean(), pts.reshape(-1,3)
    return normal_error.mean()
    # return 0

# depth_mean是渲染深度值，viewpoint里面存的是一些真值信息，其中真值normal用的是OmnidataModel计算出来的，但是特别的慢
def get_loss_normal(depth_mean, viewpoint):
    # OmnidataModel计算出来的法线，是真值法线，多做了一步归一化
    prior_normal = viewpoint.normal
    prior_normal = prior_normal.reshape(3, *depth_mean.shape[-2:]).permute(1,2,0)
    prior_normal_normalized = torch.nn.functional.normalize(prior_normal, dim=-1)

    # 渲染深度图计算出来的法线
    normal_mean, _ = depth_to_normal(viewpoint, depth_mean, world_frame=False)

    # 计算法线损失
    normal_error = 1 - (prior_normal_normalized * normal_mean).sum(dim=-1)
    normal_error[prior_normal.norm(dim=-1) < 0.2] = 0

    # # debug vis normal
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # prior_normal_show = (((prior_normal_normalized+1.0)*0.5).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
    # gt_normal, _ = depth_to_normal(viewpoint, viewpoint.depth[None], world_frame=False)
    # gt_normal_show = (((gt_normal+1.0)*0.5).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
    # prior_normal_show = (((prior_normal_normalized+1.0)*0.5).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
    # axes[0].imshow(prior_normal_show)
    # axes[0].set_title("Prior Normal")
    # axes[1].imshow(gt_normal_show)
    # axes[1].set_title("Depth to Normal")
    # axes[2].imshow(prior_normal_show)
    # axes[2].set_title("OmnidataModel Normal")
    # plt.show()

    return normal_error.mean()



def get_loss_mapping_rgb(config, image, depth, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)

    return l1_rgb.mean()


def get_img_grad_weight(img, beta=2.0):
    _, hd, wd = img.shape 
    bottom_point = img[..., 2:hd,   1:wd-1]
    top_point    = img[..., 0:hd-2, 1:wd-1]
    right_point  = img[..., 1:hd-1, 2:wd]
    left_point   = img[..., 1:hd-1, 0:wd-2]
    grad_img_x = torch.mean(torch.abs(right_point - left_point), 0, keepdim=True)
    grad_img_y = torch.mean(torch.abs(top_point - bottom_point), 0, keepdim=True)
    grad_img = torch.cat((grad_img_x, grad_img_y), dim=0)
    grad_img, _ = torch.max(grad_img, dim=0)
    grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min())
    grad_img = torch.nn.functional.pad(grad_img[None,None], (1,1,1,1), mode='constant', value=1.0).squeeze()
    return grad_img



def get_loss_mapping_rgbd(config, image, depth, viewpoint, deblur=False):
    
    gt_image = viewpoint.original_image
    gt_depth = viewpoint.depth[None]
    depth_mask = (gt_depth > 0.01).view(*depth.shape)
    # gt_depth[gt_depth<0.01] = 0.01
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    
    if deblur:
        image = image.unsqueeze(0)
        blur_image = image * viewpoint.weight_this
        M = torch.zeros((1, 2, 3), device="cuda")
        M[0, 0, 0] = 1
        M[0, 1, 1] = 1
        M[0, 0, 2] = viewpoint.blur_tran_x
        M[0, 1, 2] = viewpoint.blur_tran_y
        grid = torch.nn.functional.affine_grid(M, image.size(), align_corners=False)
        translated_image = torch.nn.functional.grid_sample(image, grid, align_corners=False)
        blur_image += translated_image * viewpoint.weight_blur
        image = blur_image.squeeze(0)
    
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask).mean()
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask).mean()
    
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    all_l1 = alpha * l1_rgb + (1 - alpha) * l1_depth
    if config["Training"]["use_ssim"]:
        # ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        ssim_score = ssim(image, gt_image)
        all_l1 += (1-ssim_score)*config["Training"]["ssim_weight"]*alpha
    return all_l1


def get_loss_rgb_blur(config, image, viewpoint):
    gt_image = viewpoint.original_image
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (image.sum(dim=0) > rgb_boundary_threshold).view(*image.shape[1:])
    image = image.unsqueeze(0)
    blur_image = image * viewpoint.weight_this
    M = torch.zeros((1, 2, 3), device="cuda")
    M[0, 0, 0] = 1
    M[0, 1, 1] = 1
    M[0, 0, 2] = viewpoint.blur_tran_x
    M[0, 1, 2] = viewpoint.blur_tran_y
    grid = torch.nn.functional.affine_grid(M, image.size(), align_corners=False)
    translated_image = torch.nn.functional.grid_sample(image, grid, align_corners=False)
    blur_image += translated_image * viewpoint.weight_blur
    image = blur_image.squeeze(0)
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask).mean()
    return l1_rgb, image


def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()



def project_to_camera(pcd, R1, T1, R2, T2, K, H, W):
    R1_inv = torch.inverse(R1) 
    pcd_world = torch.matmul(pcd - T1, R1_inv.T)
    pcd_cam2 = torch.matmul(pcd_world, R2.T) + T2  # (N, 3)
    # pcd_world = torch.matmul(pcd - T1, R1.T)
    # pcd_cam2 = torch.matmul(pcd_world, R2) + T2  # (N, 3)
    Z = pcd_cam2[:, 2].clamp(min=1e-6) 
    uv_homo = torch.matmul(pcd_cam2, K.T)  # (N, 3)
    u = (uv_homo[:, 0] / Z).round().long()  # x
    v = (uv_homo[:, 1] / Z).round().long()  # y
    depth_map = torch.zeros((H, W), device=pcd.device, dtype=torch.float64)
    valid_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (Z > 0)
    u, v, Z = u[valid_mask], v[valid_mask], Z[valid_mask]
    depth_map.index_put_((v, u), Z, accumulate=False)
    mask = depth_map > 0.01
    return depth_map, mask



def get_points_depth_in_depth_map(fov_camera, depth, points_in_camera_space, scale=1):
    st = max(int(scale/2)-1,0)
    depth_view = depth[None,:,st::scale,st::scale]
    W, H = int(fov_camera.image_width/scale), int(fov_camera.image_height/scale)
    depth_view = depth_view[:H, :W]
    pts_projections = torch.stack([points_in_camera_space[:,0] * fov_camera.fx / points_in_camera_space[:,2] + fov_camera.cx,
                        points_in_camera_space[:,1] * fov_camera.fy / points_in_camera_space[:,2] + fov_camera.cy], -1).float()
    mask = (pts_projections[:, 0] > 0) & (pts_projections[:, 0] < W) &\
            (pts_projections[:, 1] > 0) & (pts_projections[:, 1] < H) & (points_in_camera_space[:,2] > 0.1)
    pts_projections[..., 0] /= ((W - 1) / 2)
    pts_projections[..., 1] /= ((H - 1) / 2)
    pts_projections -= 1
    pts_projections = pts_projections.view(1, -1, 1, 2)
    map_z = torch.nn.functional.grid_sample(input=depth_view,
                                            grid=pts_projections,
                                            mode='bilinear',
                                            padding_mode='border',
                                            align_corners=True
                                            )[0, :, :, 0]
    return map_z, mask
    
