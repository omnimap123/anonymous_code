import json
import os

import cv2
import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from natsort import natsorted
from util.utils import Log
from gaussian.renderer import render, render_instance
from gaussian.utils.loss_utils import ssim, psnr
from gaussian.utils.camera_utils import Camera
from gaussian.utils.mapping_utils import depth_to_normal

def eval_rendering(
    gtimages,
    gtdepths,
    traj,
    gaussians,
    save_dir,
    background,
    projection_matrix,
    K,
    tsdf,
    iteration="final",
    depth_scale=1000.0,
    cam_params=None,
    keyviewpoints = None, 
    keyframe_stamps = None
):
    psnr_array, ssim_array, lpips_array, l1_array = [], [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to("cuda")
    
    cat_save_dir = f'{save_dir}/renders/cat_{iteration}'
    tsdfrgb_save_dir = f'{save_dir}/renders/tsdfrgb_{iteration}'
    tsdfdepth_save_dir = f'{save_dir}/renders/tsdfdepth_{iteration}'
    
    os.makedirs(cat_save_dir, exist_ok=True)
    os.makedirs(tsdfrgb_save_dir, exist_ok=True)
    os.makedirs(tsdfdepth_save_dir, exist_ok=True)

    for i, (idx, image) in enumerate(gtimages.items()):
        
        if (idx % 5 != 0) and (((keyframe_stamps is not None) and (i not in keyframe_stamps)) or (keyframe_stamps is None)):
            continue
        
        frame = Camera.init_from_tracking(image.squeeze()/255.0, None, traj[idx], idx, projection_matrix, K)
        gtimage = frame.original_image.cuda()

        rendering = render(frame, gaussians, background)

        if cam_params is not None:
            image = rendering["render"].unsqueeze(0) 
            blur_image = image * cam_params[idx, 0]
            M = torch.zeros((1, 2, 3), device="cuda")
            M[0, 0, 0] = 1
            M[0, 1, 1] = 1
            M[0, 0, 2] = cam_params[idx, 2]
            M[0, 1, 2] = cam_params[idx, 3]
            grid = torch.nn.functional.affine_grid(M, image.size(), align_corners=False)
            translated_image = torch.nn.functional.grid_sample(image, grid, align_corners=False)
            blur_image += translated_image * cam_params[idx, 1]
            image = blur_image
        
        image = torch.clamp(image.squeeze(0), 0.0, 1.0)
        
        depth = rendering["depth"].detach().squeeze().cpu().numpy()
        # normal
        normal, _ = depth_to_normal(frame, rendering["depth"].detach(), world_frame=True)
        normal_show = (((normal+1.0)*0.5).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
        rendering = render_instance(frame, gaussians)
        instance = torch.clamp(rendering["render"], 0.0, 1.0)

        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'{tsdfrgb_save_dir}/{idx:06d}.jpg', pred)
        gt =  (gtimage.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        rgb_cat = np.concatenate([gt, pred], axis=0)
        cv2.imwrite(f'{tsdfdepth_save_dir}/{idx:06d}.png', np.clip(depth*6553.5, 0, 65535).astype(np.uint16))
    
        
        gtdepth = gtdepths[i][0]
        if (keyframe_stamps is not None) and  (i in keyframe_stamps):
            closest_index, closest_key_id = min(enumerate(keyframe_stamps), key=lambda x: abs(x[1] - i))
            gtnormal = keyviewpoints[closest_index].normal
        else:
            gtnormal, _ = depth_to_normal(frame, gtdepth[None].cuda(), world_frame=True)
        gtnormal_show = (((gtnormal+1.0)*0.5).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
        normal_cat = np.concatenate([gtnormal_show, normal_show], axis=0)
        
        invalid = gtdepth <= 0.01
        depth_clone = depth.copy()
        depth[invalid] = 0
        l1_array.append(np.abs(gtdepth[depth > 0] - depth[depth > 0]).mean().item()) 
        
        
        min_depth, max_depth = 0.1, 5.0
        depth = np.clip(depth_clone, min_depth, max_depth)
        depth_norm = ((depth - min_depth) / (max_depth - min_depth)) * 255
        depth_norm = depth_norm.astype(np.uint8)
        depth_norm = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        gtdepth = gtdepth.detach().cpu().numpy()
        gtdepth = np.clip(gtdepth,min_depth, max_depth)
        gtdepth_norm = ((gtdepth - min_depth) / (max_depth - min_depth)) * 255
        gtdepth_norm = gtdepth_norm.astype(np.uint8)
        gtdepth_norm = cv2.applyColorMap(gtdepth_norm, cv2.COLORMAP_JET)
        depth_cat = np.concatenate([gtdepth_norm, depth_norm], axis=0)
        # depth diff
        min_depth, max_depth = 0.0, 0.08
        depth_diff = np.abs(gtdepth-depth)
        depth_diff = np.clip(depth_diff, min_depth, max_depth)
        depth_diff[invalid] = 0
        depth_diff_norm = ((depth_diff - min_depth) / (max_depth - min_depth)) * 255
        depth_diff_norm = depth_diff_norm.astype(np.uint8)
        depth_diff_show = cv2.applyColorMap(depth_diff_norm, cv2.COLORMAP_JET) # 这里绘制的depth_color是归一化之后的甚至
        
        ins_gs = (instance.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        ins_gs = cv2.cvtColor(ins_gs, cv2.COLOR_BGR2RGB)
        
        gtdepth[gtdepth<0.01] = depth[gtdepth<0.01]
        cam_intr = torch.tensor([frame.fx, frame.fy, frame.cx, frame.cy]).cuda()
        cam_pose = torch.eye(4, device="cuda")
        cam_pose[:3, :3] = frame.R
        cam_pose[:3, 3] = frame.T
        cam_pose = torch.linalg.inv(cam_pose)
        ins_cat = np.concatenate([depth_diff_show, ins_gs], axis=0)
        all_cat = np.concatenate([rgb_cat, depth_cat, normal_cat, ins_cat], axis=1)
        cv2.imwrite(f'{cat_save_dir}/{idx:06d}.jpg', all_cat)


        mask = gtimage > 0
        psnr_score = psnr((image[mask]).unsqueeze(0), (gtimage[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gtimage).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gtimage).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))
    output["mean_l1"] = float(np.mean(l1_array)) if l1_array else 0

    Log(f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}, depth l1: {output["mean_l1"]}', tag="Eval")

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    os.makedirs(psnr_save_dir, exist_ok=True)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    return output



def eval_rendering_blur(
    allviewpoints, 
    gaussians, 
    save_dir, 
    background,
    iteration="final",
    depth_scale=1000.0
):
    psnr_array, ssim_array, lpips_array, l1_array = [], [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to("cuda")
    
    cat_save_dir = f'{save_dir}/renders/cat_{iteration}'
    tsdfrgb_save_dir = f'{save_dir}/renders/tsdfrgb_{iteration}'
    tsdfdepth_save_dir = f'{save_dir}/renders/tsdfdepth_{iteration}'
    
    os.makedirs(cat_save_dir, exist_ok=True)
    os.makedirs(tsdfrgb_save_dir, exist_ok=True)
    os.makedirs(tsdfdepth_save_dir, exist_ok=True)

    for i,  frame in enumerate(allviewpoints):
        if i % 5 != 0:
            continue
        gtimage = frame.original_image.cuda()
        rendering = render(frame, gaussians, background)
        image = rendering["render"].unsqueeze(0) 
        blur_image = image * frame.weight_this
        M = torch.zeros((1, 2, 3), device="cuda")
        M[0, 0, 0] = 1
        M[0, 1, 1] = 1
        M[0, 0, 2] = frame.blur_tran_x
        M[0, 1, 2] = frame.blur_tran_y
        grid = torch.nn.functional.affine_grid(M, image.size(), align_corners=False)
        translated_image = torch.nn.functional.grid_sample(image, grid, align_corners=False)
        blur_image += translated_image * frame.weight_blur
        image = blur_image
        image = torch.clamp(image.squeeze(0), 0.0, 1.0)
        depth = rendering["depth"].detach().squeeze().cpu().numpy()
        # normal
        normal, _ = depth_to_normal(frame, rendering["depth"].detach(), world_frame=True)
        normal_show = (((normal+1.0)*0.5).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
        rendering = render_instance(frame, gaussians)
        instance = torch.clamp(rendering["render"], 0.0, 1.0)
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'{tsdfrgb_save_dir}/{i:06d}.jpg', pred)
        gt =  (gtimage.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        rgb_cat = np.concatenate([gt, pred], axis=0)
        cv2.imwrite(f'{tsdfdepth_save_dir}/{i:06d}.png', np.clip(depth*6553.5, 0, 65535).astype(np.uint16))
    
        gtdepth = frame.depth
        gtnormal, _ = depth_to_normal(frame, gtdepth[None].cuda(), world_frame=True)
        gtnormal_show = (((gtnormal+1.0)*0.5).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
        normal_cat = np.concatenate([gtnormal_show, normal_show], axis=0)
        gtdepth = gtdepth.detach().cpu().numpy()
        invalid = gtdepth <= 0.1
        depth_clone = depth.copy()
        depth[invalid] = 0
        l1_array.append(np.abs(gtdepth[depth > 0] - depth[depth > 0]).mean().item()) 
        
        
        min_depth, max_depth = 0.1, 5.0
        depth = np.clip(depth_clone, min_depth, max_depth)
        depth_norm = ((depth - min_depth) / (max_depth - min_depth)) * 255
        depth_norm = depth_norm.astype(np.uint8)
        depth_norm = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        gtdepth = np.clip(gtdepth,min_depth, max_depth)
        gtdepth_norm = ((gtdepth - min_depth) / (max_depth - min_depth)) * 255
        gtdepth_norm = gtdepth_norm.astype(np.uint8)
        gtdepth_norm = cv2.applyColorMap(gtdepth_norm, cv2.COLORMAP_JET)
        depth_cat = np.concatenate([gtdepth_norm, depth_norm], axis=0)
        ins_gs = (instance.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        ins_gs = cv2.cvtColor(ins_gs, cv2.COLOR_BGR2RGB)
        
        gtdepth[gtdepth<0.01] = depth[gtdepth<0.01]
        cam_intr = torch.tensor([frame.fx, frame.fy, frame.cx, frame.cy]).cuda()
        cam_pose = torch.eye(4, device="cuda")
        cam_pose[:3, :3] = frame.R
        cam_pose[:3, 3] = frame.T
        cam_pose = torch.linalg.inv(cam_pose)
        # ins_tsdf = tsdf.get_instance_color(torch.from_numpy(gtdepth).cuda().float(), cam_intr.float(), cam_pose.float(), min_count=1)
        ins_tsdf = ins_gs
        ins_cat = np.concatenate([ins_tsdf, ins_gs], axis=0)
        all_cat = np.concatenate([rgb_cat, depth_cat, normal_cat, ins_cat], axis=1)
        cv2.imwrite(f'{cat_save_dir}/{i:06d}.jpg', all_cat)


        mask = gtimage > 0
        psnr_score = psnr((image[mask]).unsqueeze(0), (gtimage[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gtimage).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gtimage).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))
    output["mean_l1"] = float(np.mean(l1_array)) if l1_array else 0

    Log(f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}, depth l1: {output["mean_l1"]}', tag="Eval")

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    os.makedirs(psnr_save_dir, exist_ok=True)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    return output



def set_all_camera_deblur(
    gtimages,
    keyids,
    keyviewpoints,
    save_dir
):
    all_keyid = np.array(keyids)
    blur_tran_xs = []
    blur_tran_ys = []
    weight_thiss = []
    weight_blurs = []
    with open(f"{save_dir}/camera_deblur.txt", 'w') as outfile:
        for id in range(len(gtimages)):
            # closest_key_id = min(all_keyid, key=lambda x: abs(x - id))
            closest_index, closest_key_id = min(enumerate(all_keyid), key=lambda x: abs(x[1] - id))
            blur_tran_xs.append(keyviewpoints[closest_index].blur_tran_x.item())
            blur_tran_ys.append(keyviewpoints[closest_index].blur_tran_y.item())
            weight_thiss.append(keyviewpoints[closest_index].weight_this.item())
            weight_blurs.append(keyviewpoints[closest_index].weight_blur.item())
            if id in keyids:
                line = f"Camera id: {id:03d},      weight_this: {weight_thiss[-1]:.3f},      weight_blur: {weight_blurs[-1]:.3f},      blur_tran_x: {blur_tran_xs[-1]:.4f},      blur_tran_y: {blur_tran_ys[-1]:.4f}"
                outfile.write(line + '\n')
    return torch.tensor([weight_thiss, weight_blurs, blur_tran_xs, blur_tran_ys]).T.cuda()
        
    


def eval_rendering_all(
    viewpoints,
    gaussians,
    background,
    cam_params=None
):
    psnr_array, ssim_array, lpips_array = [], [], []
    for frame in viewpoints:
        gtimage = frame.original_image.cuda()
        rendering = render(frame, gaussians, background)
        # image = (torch.exp(frame.exposure_a)) * rendering["render"] + frame.exposure_b
        image = rendering["render"].unsqueeze(0) 
        blur_image = image * frame.weight_this
        M = torch.zeros((1, 2, 3), device="cuda")
        M[0, 0, 0] = 1
        M[0, 1, 1] = 1
        M[0, 0, 2] = frame.blur_tran_x
        M[0, 1, 2] = frame.blur_tran_y
        grid = torch.nn.functional.affine_grid(M, image.size(), align_corners=False)
        translated_image = torch.nn.functional.grid_sample(image, grid, align_corners=False)
        blur_image += translated_image * frame.weight_blur
        image = blur_image
        
        image = torch.clamp(image.squeeze(0), 0.0, 1.0)
        mask = gtimage > 0
        psnr_score = psnr((image[mask]).unsqueeze(0), (gtimage[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gtimage).unsqueeze(0))
        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    Log(f'all mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}', tag="Eval")
    return output


def eval_rendering_kf(
    viewpoints,
    gaussians,
    background
):
    psnr_array, ssim_array, lpips_array = [], [], []
    for frame in viewpoints:
        gtimage = frame.original_image.cuda()
        rendering = render(frame, gaussians, background)
        # image = (torch.exp(frame.exposure_a)) * rendering["render"] + frame.exposure_b
        image = rendering["render"].unsqueeze(0) 
        
        blur_image = image * frame.weight_this
        M = torch.zeros((1, 2, 3), device="cuda")
        M[0, 0, 0] = 1
        M[0, 1, 1] = 1
        M[0, 0, 2] = frame.blur_tran_x
        M[0, 1, 2] = frame.blur_tran_y
        grid = torch.nn.functional.affine_grid(M, image.size(), align_corners=False)
        translated_image = torch.nn.functional.grid_sample(image, grid, align_corners=False)
        blur_image += translated_image * frame.weight_blur
        image = blur_image
        
        image = torch.clamp(image.squeeze(0), 0.0, 1.0)
        mask = gtimage > 0
        psnr_score = psnr((image[mask]).unsqueeze(0), (gtimage[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gtimage).unsqueeze(0))
        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    Log(f'kf mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}', tag="Eval")
    return output

def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    print('saved to ', point_cloud_path)
    
    
    
    
def eval_fast(
    gtimages,
    traj,
    gaussians,
    background,
    projection_matrix,
    K,
    cam_params=None
):
    # gtdepths = natsorted(os.listdir(gtdepthdir)) if gtdepthdir is not None else None
    psnr_array, ssim_array = [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to("cuda")
    
    for i, (idx, image) in enumerate(gtimages.items()):
        frame = Camera.init_from_tracking(image.squeeze()/255.0, None, traj[idx], idx, projection_matrix, K)
        gtimage = frame.original_image.cuda()

        rendering = render(frame, gaussians, background)
        image = rendering["render"].unsqueeze(0) 

        if cam_params is not None:
            blur_image = image * cam_params[idx, 0]
            M = torch.zeros((1, 2, 3), device="cuda")
            M[0, 0, 0] = 1
            M[0, 1, 1] = 1
            M[0, 0, 2] = cam_params[idx, 2]
            M[0, 1, 2] = cam_params[idx, 3]
            grid = torch.nn.functional.affine_grid(M, image.size(), align_corners=False)
            translated_image = torch.nn.functional.grid_sample(image, grid, align_corners=False)
            blur_image += translated_image * cam_params[idx, 1]
            image = blur_image
        
        image = torch.clamp(image.squeeze(0), 0.0, 1.0)
            

        mask = gtimage > 0
        psnr_score = psnr((image[mask]).unsqueeze(0), (gtimage[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gtimage).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))

    Log(f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}', tag="Eval")
    return output
