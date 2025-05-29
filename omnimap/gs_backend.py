import random
import time
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import trange
from munch import munchify
from lietorch import SE3, SO3
import open3d as o3d
import cv2
from collections import defaultdict
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from util.utils import Log, clone_obj
from util.vis_utils import draw_camera, create_camera_trajectory_line, update_camera_trajectory
from gaussian.renderer import render
from gaussian.utils.loss_utils import l1_loss, ssim
from gaussian.scene.gaussian_model import GaussianModel
from gaussian.utils.graphics_utils import getProjectionMatrix2
from gaussian.utils.sh_utils import SH2RGB
from gaussian.utils.mapping_utils import to_se3_vec, get_loss_normal, get_loss_mapping_rgbd, get_loss_depth_normal
from gaussian.utils.camera_utils import Camera
from gaussian.utils.eval_utils import eval_rendering, eval_rendering_kf, eval_fast, eval_rendering_all, set_all_camera_deblur, eval_rendering_blur
# from gaussian.gui import gui_utils, slam_gui
import warnings
warnings.filterwarnings('ignore')

class GSBackEnd(mp.Process):
    def __init__(self, config, tsdfs, save_dir, vis_gui):
        super().__init__()
        self.config = config
        
        self.iteration_count = 0
        # only need save keyframe viewpoint
        self.allviewpoints = []
        self.keyviewpoints = []
        self.keyframe_stamps = []
        self.initialized = False
        self.save_dir = save_dir
        self.tsdfs = tsdfs

        self.opt_params = munchify(config["opt_params"])

        self.gaussians = GaussianModel(sh_degree=0, config=self.config)
        self.gaussians.init_lr(6.0)
        self.gaussians.training_setup(self.opt_params)
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        self.no_key_count = 0.0
        self.cameras_extent = 6.0
        self.set_hyperparams()
        self.vis_gui = vis_gui
        
        # for multi-view
        self.key_camera_centers = [] 
        self.key_center_rays = []
        self.key_graph = {}

    def set_gui(self):
        # OpenCV window name
        self.window_name = "omnimap - Visualization"
        self.images = [np.zeros((self.vis_h, self.vis_w, 3), dtype=np.uint8)] * 5  # Placeholder for 4 images
        # self.texts = ["", "", "", ""]  # Placeholder for 4 text lines
        # Initialize OpenCV window (First time)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # Display 4 images and 4 text lines in the OpenCV window
        self.edge = 50
        self.img_display = np.zeros(((self.edge*4+self.vis_h*2), (self.edge*4+self.vis_w*3), 3), dtype=np.uint8)  # Create a blank canvas
        if self.vis_w >= 600:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf", 50)
        else:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf", 42)
        
        self.o3d_window = o3d.visualization.VisualizerWithKeyCallback()
        self.o3d_window.create_window(window_name="3DGS Point Viewer", width=860, height=540)
        self.gs_pc_geometries, self.cam_lines, self.cam_plan = None, None, None
        self.cam_traj = create_camera_trajectory_line()
        
        
    def add_camera(self, pose, size=0.1):
        if self.cam_lines is not None:
            self.o3d_window.remove_geometry(self.cam_lines)
            self.o3d_window.remove_geometry(self.cam_plane)
        self.cam_lines, self.cam_plane = draw_camera(pose)
        self.o3d_window.add_geometry(self.cam_lines)
        self.o3d_window.add_geometry(self.cam_plane)
        
    def update_gs_pc(self):
        if self.cam_traj is not None:
            self.o3d_window.remove_geometry(self.cam_traj)
        self.o3d_window.add_geometry(self.cam_traj)
        if self.gs_pc_geometries is not None:
            self.o3d_window.remove_geometry(self.gs_pc_geometries, reset_bounding_box=False)
            
        opacity = self.gaussians.get_opacity.detach().squeeze().cpu().numpy()
        mask = opacity>0.3
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(self.gaussians.get_xyz.detach().cpu().numpy()[mask])
        rgbs = SH2RGB(self.gaussians.get_features.detach()).squeeze().cpu().numpy()
        pc.colors = o3d.utility.Vector3dVector(rgbs[mask])
        if self.config['scene'] == 'room_0':
            bbox = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=(-np.inf, -np.inf, -np.inf),
                max_bound=(np.inf, np.inf, 0.7)
            )
            pc = pc.crop(bbox)
        # debug
        # o3d.visualization.draw_geometries([pc])
        self.gs_pc_geometries = pc
        self.o3d_window.add_geometry(self.gs_pc_geometries)
        self.o3d_window.poll_events()
        self.o3d_window.update_renderer()
        
    
    def update_images(self, tstamp, hz):
        # reset the text area
        self.img_display = np.zeros(((self.edge*4+self.vis_h*2), (self.edge*4+self.vis_w*3), 3), dtype=np.uint8)
        mask_img, ins_img, ins_num = self.tsdfs.get_vis_imgs(self.images[4])
        # Arrange 4 images in a 2x2 grid
        self.img_display[self.edge*2:(self.edge*2+self.vis_h), self.edge:(self.edge+self.vis_w)] = self.images[0]  # Top-left
        self.img_display[self.edge*2:(self.edge*2+self.vis_h), (2*self.edge+self.vis_w):(2*self.edge+self.vis_w*2)] = self.images[2]  # Top-middle
        self.img_display[self.edge*2:(self.edge*2+self.vis_h), (3*self.edge+self.vis_w*2):(3*self.edge+self.vis_w*3)] = mask_img  # Top-right
        self.img_display[(self.edge*3+self.vis_h):(self.edge*3+self.vis_h*2), self.edge:(self.edge+self.vis_w)] = self.images[1]  # Bottom-left
        self.img_display[(self.edge*3+self.vis_h):(self.edge*3+self.vis_h*2), (2*self.edge+self.vis_w):(2*self.edge+self.vis_w*2)] = self.images[3]  # Bottom-middle
        self.img_display[(self.edge*3+self.vis_h):(self.edge*3+self.vis_h*2), (3*self.edge+self.vis_w*2):(3*self.edge+self.vis_w*3)] = ins_img  # Bottom-right

        gaussian_count = len(self.gaussians.get_xyz)
        kf_len = len(self.keyframe_stamps)
        formatted_text = f" Frame: {tstamp:4d}    Gaussians: {gaussian_count:6d}    KFs: {kf_len:3d}"
        
        
        img_pil = Image.fromarray(self.img_display)
        draw = ImageDraw.Draw(img_pil)
        draw.text((400, 20), formatted_text, font=self.font, fill=(255, 255, 255))
        self.img_display = np.array(img_pil)
        img_display = cv2.resize(self.img_display, (int(self.img_display.shape[1] * 0.5), int(self.img_display.shape[0] * 0.5)))
        image_save_dir = f'{self.save_dir}/online_vis/'
        os.makedirs(image_save_dir, exist_ok=True)
        if tstamp % self.config["instance"]["instance_skip"] == 0:
            cv2.imwrite(f'{image_save_dir}/{tstamp}.jpg', img_display)
        cv2.imshow(self.window_name, img_display)
        cv2.resizeWindow(self.window_name, img_display.shape[1], img_display.shape[0])
        cv2.moveWindow(self.window_name, 875, 50)
        cv2.waitKey(1)  # Refresh window
        
        
        
    def set_hyperparams(self):
        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.max_keyframe_skip = self.config["Training"]["max_keyframe_skip"]
        self.window_size = self.config["Training"]["window_size"]
        self.frame_itr = self.config["Training"]["frame_itr"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.gaussian_extent = self.cameras_extent * self.config["Training"]["gaussian_extent"]
        self.use_omni_normal = self.config["Training"]["use_omni_normal"]
        self.normal_weight = self.config["Training"]["normal_weight"]
        self.use_post_refine = self.config["opt_params"]["post_refine"]
        if self.use_post_refine:
            self.post_itr = self.config["opt_params"]["post_itr"]
        self.wait_latest_keyframe = False
        self.deblur =  self.config["Training"]["deblur"]
        self.camera_optimizer = None
            
            
    
    def post_refine(self):
        Log("Starting post refinement", tag="GaussianSplatting")
        if self.config["Training"]["compensate_exposure"]:
            opt_params = []
            for view in self.keyviewpoints:
            # for view in self.allviewpoints:
                # print(view.uid)
                opt_params.append({
                        "params": [view.exposure_a],
                        "lr": self.config["opt_params"]["exposure_lr"],
                        "name": "exposure_a_{}".format(view.uid)})
                opt_params.append({
                        "params": [view.exposure_b],
                        "lr": self.config["opt_params"]["exposure_lr"],
                        "name": "exposure_b_{}".format(view.uid)})
            self.exposure_optimizers = torch.optim.Adam(opt_params)
        
        
        for iteration in (pbar := trange(1, self.post_itr + 1)):
            
            loss = 0.0
            use_indices = torch.randperm(len(self.keyframe_stamps))[:self.window_size]
            viewpoints = [self.keyviewpoints[random_id] for random_id in use_indices] 
            # use_indices = torch.randperm(len(self.allviewpoints))[:self.window_size]
            # viewpoints = [self.allviewpoints[random_id] for random_id in use_indices] 
            for viewpoint in viewpoints:
                render_pkg = render(viewpoint, self.gaussians, self.background)
                image, depth = render_pkg["render"], render_pkg["depth"]
                image = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
                loss += get_loss_mapping_rgbd(self.config, image, depth, viewpoint, self.deblur)
                if self.use_omni_normal:
                    loss += self.normal_weight * get_loss_normal(depth, viewpoint)
                else:
                    loss += self.normal_weight * get_loss_depth_normal(depth, viewpoint)
                
            loss.backward()
            
            with torch.no_grad():
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                lr = self.gaussians.update_learning_rate(iteration)
                if self.deblur:
                    self.camera_optimizer.step()
                    self.camera_optimizer.zero_grad(set_to_none=True)
                if self.config["Training"]["compensate_exposure"]:
                    self.exposure_optimizers.step()
                    self.exposure_optimizers.zero_grad(set_to_none=True)
            pbar.set_description(f"Global GS Refinement lr {lr:.3E} loss {loss.item():.3f}")
                
        
    def finalize(self):
        if self.use_post_refine:
            self.post_refine()
        self.gaussians.save_ply(f'{self.save_dir}/3dgs_final.ply')
        return

    @torch.no_grad()
    def eval_fast(self, gtimages, traj, depth_scale=1000.0):
        self.cam_params = set_all_camera_deblur(gtimages, self.keyframe_stamps,  self.keyviewpoints, self.save_dir)
        eval_fast(gtimages, traj, self.gaussians, self.background,
            self.projection_matrix, self.K, self.cam_params)
        eval_rendering_kf(self.keyviewpoints, self.gaussians, self.background)
        
    @torch.no_grad()
    def eval_rendering(self, gtimages, gtdepths, traj, depth_scale=1000.0):
        eval_rendering(gtimages, gtdepths, traj, self.gaussians,self.save_dir, self.background,
            self.projection_matrix, self.K, self.tsdfs, iteration="after_opt", depth_scale=depth_scale, cam_params = self.cam_params)
        eval_rendering_kf(self.keyviewpoints, self.gaussians, self.background)


    def reset(self):
        self.iteration_count = 0
        self.current_window = []
        self.initialized = False

    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            render_pkg = render(viewpoint, self.gaussians, self.background)
            (image, viewspace_point_tensor, visibility_filter, radii, depth, n_touched) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["n_touched"]
            )
            loss_init = get_loss_mapping_rgbd(self.config, image, depth, viewpoint)
            if self.use_omni_normal:
                loss_init += self.normal_weight * get_loss_normal(depth, viewpoint)
            else:
                loss_init += self.normal_weight * get_loss_depth_normal(depth, viewpoint)
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None
                    )

                if self.iteration_count == self.init_gaussian_reset:
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        Log("Initialized map")
        return render_pkg


    def map(self, viewpoints, iters, current_id, is_keyframe, corr_index=None):
        '''
        3dgs training
        '''
        
                
        for iter in range(iters):
            self.iteration_count += 1
            loss_for_gs = 0
            if corr_index is not None:
                # all_view_rgb = []
                all_view_depth = []
                all_view_points = []
            for view_id, viewpoint in enumerate(viewpoints):
                render_pkg = render(viewpoint, self.gaussians, self.background)
                image, viewspace_point_tensor, visibility_filter, radii, depth, n_touched = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["n_touched"])
                loss_this = get_loss_mapping_rgbd(self.config, image, depth, viewpoint, self.deblur)
                if self.use_omni_normal:
                    loss_this += self.normal_weight * get_loss_normal(depth, viewpoint)
                else:
                    if corr_index is not None:
                        normal_loss, points = get_loss_depth_normal(depth, viewpoint, current_id=current_id, corr=(corr_index is not None))
                    else:
                        normal_loss = get_loss_depth_normal(depth, viewpoint, current_id=current_id, corr=(corr_index is not None))
                    loss_this += self.normal_weight * normal_loss
                if corr_index is not None:
                    # all_view_rgb.append(image)
                    all_view_depth.append(depth)
                    all_view_points.append(points)
                loss_for_gs +=  loss_this
                if self.vis_gui:
                    if iter == iters - 1 and view_id == 0:
                        pre_image = image.permute(1,2,0).clone().detach()
                        self.images[1] = torch.clamp(pre_image, 0, 1).cpu().numpy()[:, :, ::-1] *255
                        depth_vis = depth[0].clone().detach().cpu().numpy()
                        min_depth, max_depth = 0.1, 5.0
                        depth_vis = np.clip(depth_vis, 0.1, 5.0)
                        depth_norm = ((depth_vis - min_depth) / (max_depth - min_depth)) * 255
                        depth_norm = depth_norm.astype(np.uint8)
                        self.images[3] = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                        self.images[4] = depth[0].clone().detach()
            loss_for_gs.backward()

            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
            # loss_for_blur.backward()
            # with torch.no_grad():
                if self.deblur:
                    self.camera_optimizer.step()
                    self.camera_optimizer.zero_grad(set_to_none=True)
                
                
    def process_track_data(self, packet, hz):
        if not hasattr(self, "projection_matrix"):
            H, W = packet["images"].shape[-2:]
            self.K = K = list(packet["intrinsics"]) + [W, H]
            self.projection_matrix = getProjectionMatrix2(znear=0.01, zfar=100.0, fx=K[0], fy=K[1], cx=K[2], cy=K[3], W=W, H=H).transpose(0, 1).cuda()
        w2c = SE3(packet["poses"]).matrix().cuda()
        tstamp = packet['tstamp']
        idx = int(tstamp)
        viewpoint = Camera.init_from_tracking(packet["images"]/255.0, packet["depths"], w2c, idx, self.projection_matrix, self.K, tstamp, \
                       normal=packet["normals"], bg=self.background)

        # add viewpoint to the dict
        if not self.initialized:
            self.reset()
            new_points, new_coplors, is_keyframe = self.tsdfs.initializing_check()
            self.gaussians.extend_from_tsdfs(new_points, new_coplors, self.tsdfs.voxel_size/5)
            # initialize map for a large amount of iterations
            self.initialize_map(0, viewpoint)
            self.initialized = True
            if self.vis_gui:
                self.vis_h, self.vis_w = packet["images"].shape[1], packet["images"].shape[2]
                self.set_gui()
        # new image needs initialize for viewpoint and gs
        else:
            new_points, new_coplors, is_keyframe = self.tsdfs.initializing_check()
            self.gaussians.extend_from_tsdfs(new_points, new_coplors, self.tsdfs.voxel_size/2)
        if self.wait_latest_keyframe:
            is_keyframe = True
            self.wait_latest_keyframe = False
        if not is_keyframe:
            self.no_key_count += 1
            # at least a keyframe per 1 frames
        if self.no_key_count >= self.max_keyframe_skip:
                is_keyframe = True
        if len(self.keyframe_stamps)>0 and (idx-self.keyframe_stamps[-1])<3:
            is_keyframe = False
        # self.allviewpoints.append(viewpoint)
        if is_keyframe:
            self.keyframe_stamps.append(idx)
            self.keyviewpoints.append(viewpoint)
            self.no_key_count = 0
            # tell the tsdf to reset
            self.tsdfs.reset_unregistered()
        if self.deblur:
            if self.camera_optimizer is None:
                opt_params = []
                opt_params.append({"params": [viewpoint.weight_this], "lr": self.config["opt_params"]["deblur_weight"], 
                                                "name": "weight_this_{}".format(viewpoint.uid)})
                opt_params.append({"params": [viewpoint.weight_blur], "lr": self.config["opt_params"]["deblur_weight"], 
                                                "name": "weight_blur_{}".format(viewpoint.uid)})
                opt_params.append({"params": [viewpoint.blur_tran_x], "lr": self.config["opt_params"]["deblur_trans"], 
                                                "name": "blur_tran_x_{}".format(viewpoint.uid)})
                opt_params.append({"params": [viewpoint.blur_tran_y], "lr": self.config["opt_params"]["deblur_trans"], 
                                                "name": "blur_tran_y_{}".format(viewpoint.uid)})
                self.camera_optimizer = torch.optim.Adam(opt_params)
            else:
                new_params = []
                new_params.append({"params": [viewpoint.weight_this], "lr": self.config["opt_params"]["deblur_weight"], 
                                                "name": "weight_this_{}".format(viewpoint.uid)})
                new_params.append({"params": [viewpoint.weight_blur], "lr": self.config["opt_params"]["deblur_weight"], 
                                                "name": "weight_blur_{}".format(viewpoint.uid)})
                new_params.append({"params": [viewpoint.blur_tran_x], "lr": self.config["opt_params"]["deblur_trans"], 
                                                "name": "blur_tran_x_{}".format(viewpoint.uid)})
                new_params.append({"params": [viewpoint.blur_tran_y], "lr": self.config["opt_params"]["deblur_trans"], 
                                                "name": "blur_tran_y_{}".format(viewpoint.uid)})
                for param_group in new_params:
                    self.camera_optimizer.add_param_group(param_group)
        
        
        use_indices = torch.randperm(len(self.keyframe_stamps))[:self.window_size]
        viewpoints = [viewpoint] + [self.keyviewpoints[random_id] for random_id in use_indices] 
        self.map(viewpoints, self.frame_itr, idx, is_keyframe)
        
        if self.vis_gui:
            gt_image = packet["images"].permute(1,2,0).clone()
            self.images[0] = torch.clamp(gt_image, 0, 255).cpu().numpy()[:, :, ::-1] 
            depth = packet["depths"].clone().cpu().numpy()
            min_depth, max_depth = 0.1, 5.0
            depth = np.clip(depth,min_depth, max_depth)
            depth_norm = ((depth - min_depth) / (max_depth - min_depth)) * 255
            depth_norm = depth_norm.astype(np.uint8)
            self.images[2] = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            self.update_images(tstamp, hz)
            pose = np.linalg.inv(w2c.cpu().numpy())
            self.cam_traj = update_camera_trajectory(self.cam_traj, [pose[:3, 3]])
            if idx%10 == 0:
                self.add_camera(pose)
                self.update_gs_pc()
            
            
    def gs_instance(self, vis=False):
        '''assiocate the instance id to gs'''
        gs_xyz = self.gaussians.get_xyz.clone().detach()
        ids, colors = self.tsdfs.get_instance_ids(gs_xyz)
        # save gaussian id
        torch.save(ids, f"{self.save_dir}/gs_id.pt")
        bgr_reversed = torch.flip(colors, dims=[1])
        self.gaussians.set_instance_coloor(bgr_reversed)
        # debug gaussians vis
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(gs_xyz.cpu().numpy())
        pc.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
        if vis:
            o3d.visualization.draw_geometries([pc])
        o3d.io.write_point_cloud(f"{self.save_dir}/instance_gs.ply", pc)
        