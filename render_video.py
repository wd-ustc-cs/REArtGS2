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
import cv2
import copy
import torch
from scene import Scene, DeformModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, planar_render
import torchvision
from utils.general_utils import safe_state, vis_depth
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel
from utils.metrics import seed_everything
import numpy as np
from moviepy.editor import VideoFileClip
import json
from utils.image_utils import remove_isolated_noise


def image_process(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    h, w, _ = img.shape

    GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图片灰度化处理

    ret, binary = cv2.threshold(GrayImage, 127, 255, cv2.THRESH_BINARY)  # 图片二值化,灰度值大于40赋值255，反之0

    threshold = h / 30 * w / 30  # 设定阈值

    # cv2.fingContours寻找图片轮廓信息

    contours, hierarch = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])  # 计算轮廓所占面积
        if area < threshold:  # 将area小于阈值区域填充背景色，由于OpenCV读出的是BGR值
            cv2.drawContours(img, [contours[i]], -1, (255, 255, 255), thickness=-1)
            continue
    cv2.imshow('Output', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(img_path, img)

def generate_camera_poses(args, N=30):
    """
    Generate camera poses around the scene center on a circle.

    Parameters:
    - r: Radius of the circle.
    - theta: Elevation angle in degrees.
    - num_samples: Number of samples (camera positions) to generate.

    Returns:
    - poses: A list of camera poses (4x4 transformation matrices).
    """
    poses = []
    file = json.load(open(f'./arguments/cam_traj.json'))
    traj_info = file[args.dataset][args.subset][args.scene_name]
    radius, r_theta, r_phi = traj_info['radius'], traj_info['theta'], traj_info['phi']
    d_theta, d_phi = traj_info['d_theta'], traj_info['d_phi']

    thetas = np.linspace(r_theta[0] * np.pi, r_theta[1] * np.pi, N)
    thetas = np.concatenate([np.zeros(N//2), thetas]) + d_theta * np.pi
    azimuths = np.linspace(r_phi[0] * np.pi, r_phi[1] * np.pi, N)
    azimuths = np.concatenate([np.zeros(N//2), azimuths]) + d_phi * np.pi
    roty180 = np.array([[-1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]]) if traj_info['roty180'] else np.eye(4)
    rotx90 = np.array([[1, 0, 0, 0],
                       [0, 0, -1, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1]]) if traj_info['rotx90'] else np.eye(4)
    
    for theta, azimuth in zip(thetas, azimuths):
        # Convert spherical coordinates to Cartesian coordinates

        x = radius * np.cos(azimuth) * np.cos(theta)
        y = radius * np.sin(azimuth) * np.cos(theta)
        z = radius * np.sin(theta)

        # Camera position
        position = np.array([x, y, z])

        # Compute the forward direction (pointing towards the origin)
        forward = position / np.linalg.norm(position)

        # Compute the right and up vectors for the camera coordinate system
        up = np.array([0, 0, 1])
        if np.allclose(forward, up) or np.allclose(forward, -up):
            up = np.array([0, 1, 0])
        right = np.cross(up, forward)
        up = np.cross(forward, right)

        # Normalize the vectors
        right /= np.linalg.norm(right)
        up /= np.linalg.norm(up)

        # Construct the rotation matrix
        rotation_matrix = np.vstack([right, up, forward]).T

        # Construct the transformation matrix (4x4)
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = position

        transformation_matrix = roty180 @ rotx90.T @ transformation_matrix

        poses.append(transformation_matrix)
    return poses


def generate(views, args, N=30):
    new_views = []
    poses = generate_camera_poses(args, N)
    for i, pose in enumerate(poses):
        view = copy.deepcopy(views[0])
        view.fid = i / (len(poses) - 1)
        view.gt_alpha_mask = None
        matrix = np.linalg.inv(pose)
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]
        view.reset_extrinsic(R, T)
        new_views.append(view)
    return new_views


def generate_video(imgs, video_name, fps=15, brighten=False):
    # imgs: list of img tensors [3, H, W]
    height, width = imgs[0].shape[1], imgs[0].shape[2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for img in imgs:
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if brighten:
            img[:, :width//2] = cv2.convertScaleAbs(img[:, :width//2], alpha=0.8, beta=1)
        video.write(img)
    video.release()

def generate_video_ffmpeg(img_path, video_name, fps=15):
    os.system(f'rm {video_name}')
    os.system(f'ffmpeg -framerate {fps} -vsync 0 -i {img_path}' + '/%05d.png -c:v libx264 -crf 0 ' + video_name)


def video2gif(video_file):
    gif_file = video_file.replace('.mp4', '.gif').replace('video', 'gif')
    clip = VideoFileClip(video_file)
    clip.write_gif(gif_file)
    clip.close()


def render_set(args, name, iteration, views, gaussians, pipeline, background, deform, N_frames=60, inverse=False, recenter=False, save_video = False):
    model_path = args.model_path
    timels = np.concatenate([np.linspace(0, 1, N_frames//4), np.linspace(1, 0, N_frames//4)])
    timels = np.concatenate([timels, timels, timels])
    # timels = np.linspace(0, 1, N_frames)
    dx_list, dr_list, mask = deform.deform.mask_interpolate(gaussians, timels)
    
    save_dir = os.path.join(model_path, name, "ours_{}".format(iteration))
    if os.path.exists(save_dir):
        os.system(f'rm -r {save_dir}')
    makedirs(save_dir, exist_ok=True)

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "rgb")
    seg_path = os.path.join(model_path, name, "ours_{}".format(iteration), "seg")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normal")
    rgbd_path = os.path.join(model_path, name, "ours_{}".format(iteration), "rgbd")
    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(normal_path, exist_ok=True)
    makedirs(seg_path, exist_ok=True)
    makedirs(rgbd_path, exist_ok=True)

    rgbs, depths = [], []
    normals = []
    segs = []
    seg_mask = mask
    if recenter:
        gaussians._xyz -= gaussians._xyz.mean(0, keepdim=True)
    for idx, view in enumerate(tqdm(views)):
        #view.image_width = 1024
        #view.image_height = 768

        d_xyz, d_rotation = dx_list[idx], dr_list[idx]
        results = planar_render(view, gaussians, pipeline, background, d_xyz, d_rotation, mask=seg_mask)
        #rgbs.append(torch.clamp(results["render"], 0.0, 1.0))
        segs.append(torch.clamp(results["render"], 0.0, 1.0))
        # normal = results['rendered_normal']
        # normal = ((normal + 1) * 0.5).clip(0,1).detach().cpu()
        # normals.append(normal)
        results1 = render(view, gaussians, pipeline, background, d_xyz, d_rotation)
        depths.append(results1['depth'])
        rgbs.append(torch.clamp(results1["render"], 0.0, 1.0))
    if inverse:
        rgbs = rgbs[:N_frames//2] + rgbs[:N_frames//2][::-1] + rgbs[N_frames//2:] + rgbs[N_frames//2:][::-1]
        depths = depths[:N_frames//2] + depths[:N_frames//2][::-1] + depths[N_frames//2:] + depths[N_frames//2:][::-1]
    rgbs = torch.stack(rgbs, 0)
    depths = torch.stack(depths, 0)

    rgbds = []
    for i in range(len(rgbs)):
        rgb = rgbs[i].cpu() # [3, H, W]
        torchvision.utils.save_image(rgb, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
        seg = segs[i].cpu()
        torchvision.utils.save_image(seg, os.path.join(seg_path, '{0:05d}'.format(i) + ".png"))
        # remove_isolated_noise(
        #     os.path.join(render_path, '{0:05d}'.format(i) + ".png"),
        #     os.path.join(render_path, 'refine_{0:05d}'.format(i) + ".png"),
        #     method='custom_filter',
        #     min_area=15  # 面积小于15像素的区域视为噪声
        # )
        #image_process(os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
        depth = vis_depth(depths[i], os.path.join(depth_path, '{0:05d}'.format(i) + ".png")) # [3, H, W]
        #normal = normals[i].cpu()
        #torchvision.utils.save_image(normal, os.path.join(normal_path, '{0:05d}'.format(i) + ".png"))
        rgbd = torchvision.utils.make_grid([rgb, depth, seg], nrow=3, padding=0)
        torchvision.utils.save_image(rgbd, os.path.join(rgbd_path, '{0:05d}'.format(i) + ".png"))
        rgbds.append(rgbd)
    # save video
    scene_name = args.scene_name

    if save_video:
        # generate_video_ffmpeg(rgbd_path, os.path.join('data/demo/video', f"{scene_name}.mp4"), fps=10)
        generate_video(rgbds, os.path.join(model_path, name, "ours_{}".format(iteration), f"{scene_name}.mp4"), fps=15)
        video2gif(os.path.join(model_path, name, "ours_{}".format(iteration), f"{scene_name}.mp4"))
        print(f"Saved video to {os.path.join(model_path, name, 'ours_{}'.format(iteration), f'{scene_name}.mp4')}")
    

def render_sets(args, dataset: ModelParams, iteration, pipeline: PipelineParams, N_frames=30):
    with torch.no_grad():
        deform = DeformModel(dataset)
        loaded = deform.load_weights(dataset.model_path, iteration=iteration)
        if not loaded:
            raise ValueError(f"Failed to load weights from {dataset.model_path}")
        deform.update(30000)

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        cam_traj = scene.getTrainCameras()
        # cam_traj = scene.getTestCameras()
        # id = 0
        # cam_traj = cam_traj[id:id+1] * N_frames
        if args.inverse:
            N_frames = N_frames // 2
        print(N_frames)

        cam_traj = generate(cam_traj, args, N_frames)
        #cam_traj = scene.getTrainCameras()[0:1] * N_frames
        render_set(args, "render", scene.loaded_iter, cam_traj, gaussians, pipeline, background, deform, N_frames, args.inverse, args.recenter, args.save_video)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--N_frames", default=60, type=int) # 140 ours
    parser.add_argument("--inverse", action="store_true")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--recenter", action="store_true", help="Recenter the gs for real world demo")

    args = get_combined_args(parser)
    args.source_path = f'/media/wd/work/ArtGS_data/{args.dataset}/{args.subset}/{args.scene_name}'

    print("Rendering " + args.source_path + ' with '+ args.model_path)
    safe_state(args.quiet)
    seed_everything(args.seed)
    render_sets(args, model.extract(args), args.iteration, pipeline.extract(args), args.N_frames)
