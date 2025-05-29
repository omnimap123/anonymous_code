import os    # nopep8
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys   # nopep8
sys.path.append(os.path.join(os.path.dirname(__file__), 'omnimap'))   # nopep8
import time
import torch
import cv2
import re
import os
import argparse
import numpy as np
import lietorch
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (100000, rlimit[1]))

from omnimap.util.utils import load_config
from tqdm import tqdm, trange
from torch.multiprocessing import Process, Queue
from omni import OMNI
from natsort import natsorted
from scipy.spatial.transform import Rotation as R


def save_trajectory(omni, all_inputs, output):
    np.save("{}/intrinsics.npy".format(output), omni.intrinsics.cpu().numpy())
    ttraj_full = [all_inputs[i]['pose'].cpu().numpy()[0] for i in range(len(all_inputs))]
    ttraj_full = np.stack(ttraj_full)
    np.savetxt(f"{output}/traj_full.txt", ttraj_full)
        
        
def to_se3_matrix(pvec):
    pose = np.eye(4)
    pose[:3, :3] = R.from_quat(pvec[4:]).as_matrix()
    pose[:3, 3] = pvec[1:4]
    return pose


def rgbd_stream(rgbddir, depthdir, posefile, calib, undistort=False, cropborder=False, start=0, length=100000, max_depth=12.0, dataset="replica"):
    """ image generator """
    
    all_inputs = []

    calib = np.loadtxt(calib, delimiter=" ")
    if calib.ndim == 2:
        K = calib[:3,:3]
        calib = np.array([calib[0,0], calib[1,1], calib[0,2], calib[1,2]])
        depth_scale = 1000.0
    else:
        K = np.array([[calib[0], 0, calib[2]],[0, calib[1], calib[3]],[0,0,1]])
        depth_scale = calib[4]

    rgb_image_list = natsorted(os.listdir(rgbddir))[start:start+length]
    depth_image_list = natsorted(os.listdir(depthdir))[start:start+length]
    
    poses = []
    poses_4x4 = []
    
    with open(posefile, "r") as f:
        lines = f.readlines()
    for i in range(start, min(len(lines), start+length)):
        line = np.array(list(map(float, lines[i].split())))
        # for N,16
        c2w = line.reshape(4, 4)
        poses_4x4.append(c2w)
        w2c = np.linalg.inv(c2w)
        quat = R.from_matrix(w2c[:3, :3]).as_quat()
        pose = np.hstack((w2c[:3, 3], quat))
        poses.append(pose)
    poses_4x4 = torch.as_tensor(np.array(poses_4x4))
    poses = torch.as_tensor(np.array(poses))
    print("loading data ......")
    for t, (rgbfile, depthfile) in zip(trange(len(rgb_image_list)), zip(rgb_image_list, depth_image_list)):
        image = cv2.imread(os.path.join(rgbddir, rgbfile))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(os.path.join(depthdir, depthfile), cv2.IMREAD_UNCHANGED)
        intrinsics = torch.tensor(calib[:4])
        if len(calib) > 4 and undistort:
            image = cv2.undistort(image, K, calib[4:])
        h0, w0, _ = image.shape
        if h0%10 != 0:
            w1, h1 = 640, 480
        elif h0 == 680:
            w1, h1 = 600, 340
        elif h0 == 720:
            w1, h1 = 480, 270
        image = cv2.resize(image, (w1, h1))
        intrinsics[[0,2]] *= (w1 / w0)
        intrinsics[[1,3]] *= (h1 / h0)
        h0, w0 = depth.shape
        depth = cv2.resize(depth, (w1, h1), interpolation=cv2.INTER_NEAREST)
        depth = depth / depth_scale
        pose = poses[t]
        is_last = (t == len(rgb_image_list) - 1)
        
        if cropborder > 0:
            image = image[cropborder:-cropborder, cropborder:-cropborder]
            depth = depth[cropborder:-cropborder, cropborder:-cropborder]
            intrinsics[2:] -= cropborder
            
        image = torch.as_tensor(image).permute(2, 0, 1)
        depth = torch.as_tensor(depth)
        depth[depth>max_depth]=0.0
        
        # Append a dictionary of data for the current frame
        frame_data = {
            'index': t,
            'image': image[None],
            'depth': depth[None],
            'pose': pose[None],
            'intrinsics': intrinsics[None],
            'pose_44': poses_4x4[t][None],
            'is_last': is_last,
            'depth_scale': depth_scale
        }
        # yield frame_data
        all_inputs.append(frame_data)
    return all_inputs



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="replica", help="dataset name")
    parser.add_argument("--scene", type=str, default="room_0", help="scene name")
    parser.add_argument("--max_depth", type=float, default=7.0, help="the max depth used for depth image")
    parser.add_argument("--cropborder", type=int, default=0, help="crop images to remove black border")
    parser.add_argument("--start", type=int, default=0, help="start frame")
    parser.add_argument("--length", type=int, default=100000, help="number of frames to process")
    parser.add_argument("--vis_gui", action="store_true", help="use opencv to visuliazation the whole process")
    parser.add_argument("--undistort", action="store_true", help="undistort images if calib file contains distortion parameters")
    parser.add_argument("--output", default='None', help="path to save output")
    args = parser.parse_args()
    
    args.config = f"config/{args.dataset}_config.yaml"
    config = load_config(args.config)
    config['scene']=args.scene
    dataset_dir = config['path']['data_path']
    
    if args.dataset == "replica":
        rgbdir = f"{dataset_dir}/{args.scene}/imap/00/rgb"
        depthdir = f"{dataset_dir}/{args.scene}/imap/00/depth"
        pose = f"{dataset_dir}/{args.scene}/imap/00/traj_w_c.txt"
        args.calib = f"calib/{args.dataset}.txt"
    elif args.dataset == "scannet":
        rgbdir = f"{dataset_dir}/{args.scene}/color"
        depthdir = f"{dataset_dir}/{args.scene}/depth"
        pose = f"{dataset_dir}/{args.scene}/traj_w_c.txt" # Converted from pose/
        args.calib = f"{dataset_dir}/{args.scene}/intrinsic/intrinsic_color.txt"
        args.cropborder = 10
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}.")
        
    
    if args.output == "None":
        args.output = f"outputs/{args.scene}"
    os.makedirs(args.output, exist_ok=True)
    torch.multiprocessing.set_start_method('spawn')

    omni = None
    all_inputs = rgbd_stream(rgbdir, depthdir, pose, args.calib, args.undistort, args.cropborder, args.start, args.length, args.max_depth, args.dataset)
    
    progress_bar = tqdm(range(0, len(all_inputs)), desc="Training")
    # progress_bar = tqdm(desc="Training")
    
    for frame_data in all_inputs:
        t, image, depth, pose, intrinsics, pose_44, is_last = frame_data['index'],frame_data['image'],\
            frame_data['depth'],frame_data['pose'],frame_data['intrinsics'],frame_data['pose_44'],frame_data['is_last']
        if omni is None:
            args.image_size = [image.shape[2], image.shape[3]]
            args.depth_scale = frame_data['depth_scale']
            omni = OMNI(args, config)
        omni.track(t, image, depth, pose, progress_bar, intrinsics=intrinsics, is_last=is_last, pose_44=pose_44)
    progress_bar.close()


    omni.terminate()
    save_trajectory(omni, all_inputs, args.output)
    print("Done")
