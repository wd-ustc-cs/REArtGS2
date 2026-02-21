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
import numpy as np
from utils.mesh_utils import GaussianExtractor
from utils.metrics import *
from utils.geo_utils import find_biggest_cluster
import open3d as o3d
import pandas as pd
import cv2
import seaborn as sns


def create_small_direction_arrow(radius, arrow_size, arc_angle):
    """
       为部分圆环创建表示旋转方向的小箭头
       """
    # 创建圆锥体作为箭头
    cone = o3d.geometry.TriangleMesh.create_cone(radius=arrow_size / 2, height=arrow_size)

    # 计算箭头位置（在圆环的末端）
    end_angle = np.deg2rad(arc_angle)
    x = radius * np.cos(end_angle)
    y = radius * np.sin(end_angle)

    # 将箭头放置在圆环的末端
    cone.translate([x, y, 0])

    # 计算箭头方向（圆环末端的切线方向）
    tangent_x = -np.sin(end_angle)
    tangent_y = np.cos(end_angle)
    tangent_direction = np.array([tangent_x, tangent_y, 0])

    # 计算旋转矩阵，使箭头指向切线方向
    z_axis = np.array([0, 0, 1])  # 默认箭头方向
    rotation_matrix = rotation_matrix_from_vectors(z_axis, tangent_direction)

    # 旋转箭头使其指向切线方向
    cone.rotate(rotation_matrix, center=[x, y, 0])

    return cone



def rotation_matrix_from_vectors(vec1, vec2):
    """计算两个向量之间的旋转矩阵"""
    a, b = vec1 / np.linalg.norm(vec1), vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)

    if np.allclose(c, -1.0):
        return -np.eye(3)

    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def create_torus(radius, tube_radius, arc_angle=240, radial_resolution=30, tubular_resolution=30):
    """
    创建部分圆环mesh

    参数:
    - radius: 圆环半径
    - tube_radius: 圆环管半径
    - arc_angle: 圆环角度范围（度）
    - radial_resolution: 径向分辨率
    - tubular_resolution: 管状分辨率
    """
    mesh = o3d.geometry.TriangleMesh()

    # 将角度转换为弧度
    arc_radians = np.deg2rad(arc_angle)

    # 生成顶点
    vertices = []
    for i in range(radial_resolution + 1):  # +1 确保覆盖整个角度范围
        u = i / radial_resolution * arc_radians  # u在0到arc_radians之间

        for j in range(tubular_resolution):
            v = j / tubular_resolution * 2 * np.pi

            x = (radius + tube_radius * np.cos(v)) * np.cos(u)
            y = (radius + tube_radius * np.cos(v)) * np.sin(u)
            z = tube_radius * np.sin(v)

            vertices.append([x, y, z])

    # 生成三角形面
    triangles = []
    for i in range(radial_resolution):
        for j in range(tubular_resolution):
            # 当前网格的四个顶点
            i_next = i + 1
            j_next = (j + 1) % tubular_resolution

            # 四个顶点的索引
            v00 = i * tubular_resolution + j
            v01 = i * tubular_resolution + j_next
            v10 = i_next * tubular_resolution + j
            v11 = i_next * tubular_resolution + j_next

            # 添加两个三角形
            triangles.append([v00, v10, v01])
            triangles.append([v01, v10, v11])

    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    return mesh


def create_rotation_torus_arrow(origin, axis, radius=0.1, tube_radius=0.01, arrow_scale=3, arc_angle = 240):
    """
    创建表示旋转的环形箭头（圆环+小箭头表示方向）
    """
    # 归一化轴向量
    axis = axis / np.linalg.norm(axis)

    # 创建圆环（torus）
    torus = create_torus(radius, tube_radius, arc_angle)
    torus.paint_uniform_color([1, 0, 0])  # 红色表示旋转

    # 创建表示旋转方向的小箭头
    arrow = create_small_direction_arrow(radius, tube_radius * arrow_scale, arc_angle)
    arrow.paint_uniform_color([1, 0, 0])  # 红色

    # 合并圆环和箭头
    rotation_arrow = torus + arrow

    # 计算旋转矩阵，使圆环平面垂直于轴方向
    # 默认圆环在XY平面，法向量为Z轴
    z_axis = np.array([0, 0, 1])

    if np.abs(np.dot(axis, -z_axis))!=1:
        rotation_matrix = rotation_matrix_from_vectors(z_axis, axis)

        # 应用旋转
        rotation_arrow.rotate(rotation_matrix, center=[0, 0, 0])

    # 应用平移
    rotation_arrow.translate(origin)

    return rotation_arrow


def generate_camera_poses(r=3, N=30):
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
    for i, theta in enumerate(range(-85, 85, 10)):
        theta_rad = np.deg2rad(theta)

        # Generate azimuth angles evenly spaced around the circle
        num_samples = 25 - abs(theta) // 4
        azimuths = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)

        for azimuth in azimuths:
            # Convert spherical coordinates to Cartesian coordinates
            x = r * np.cos(azimuth) * np.cos(theta_rad)
            y = r * np.sin(azimuth) * np.cos(theta_rad)
            z = r * np.sin(theta_rad)

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

            poses.append(transformation_matrix)

    return poses


def generate(views):
    new_views = []
    poses = generate_camera_poses(2)
    for i, pose in enumerate(poses):
        view = copy.deepcopy(views[0])
        view.fid = np.random.randint(2, size=1).item()
        view.gt_alpha_mask = None
        matrix = np.linalg.inv(pose)
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]
        view.reset_extrinsic(R, T)
        new_views.append(view)
    return new_views


def get_rotation_axis_angle(k, theta):
    '''
    Rodrigues' rotation formula
    args:
    * k: direction unit vector of the axis to rotate about
    * theta: the (radian) angle to rotate with
    return:
    * 3x3 rotation matrix
    '''
    if np.linalg.norm(k) == 0.:
        return np.eye(3)
    k = k / np.linalg.norm(k)
    kx, ky, kz = k[0], k[1], k[2]
    cos, sin = np.cos(theta), np.sin(theta)
    R = np.zeros((3, 3))
    R[0, 0] = cos + (kx ** 2) * (1 - cos)
    R[0, 1] = kx * ky * (1 - cos) - kz * sin
    R[0, 2] = kx * kz * (1 - cos) + ky * sin
    R[1, 0] = kx * ky * (1 - cos) + kz * sin
    R[1, 1] = cos + (ky ** 2) * (1 - cos)
    R[1, 2] = ky * kz * (1 - cos) - kx * sin
    R[2, 0] = kx * kz * (1 - cos) - ky * sin
    R[2, 1] = ky * kz * (1 - cos) + kx * sin
    R[2, 2] = cos + (kz ** 2) * (1 - cos)
    return R


def save_axis_mesh(k, center, filepath):
    '''support rotate only for now'''
    axis = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.02, cone_radius=0.04, cylinder_height=0.7, # cylinder_radius=0.01, cylinder_height=0.7, cone_radius=0.02
                                                  cone_height=0.04)
    arrow = np.array([0., 0., 1.], dtype=np.float32)
    n = np.cross(arrow, k)
    rad = np.arccos(np.dot(arrow, k))
    R_arrow = get_rotation_axis_angle(n, rad)
    axis.rotate(R_arrow, center=(0, 0, 0))
    axis.translate(center[:3])
    o3d.io.write_triangle_mesh(filepath, axis)


joint_type_dict = {
    'r': 'hinge',
    'p': 'slider',
}


def export_joint_info_json(pred_joint_list, mesh_files, exp_dir):
    meta_info = []
    for i, joint_info in enumerate(pred_joint_list):
        if i == 0:
            entry = {
                "id": i,
                "parent": -1,
                "name": "root",
                "joint": 'heavy',
                "jointData": {},
                "visuals": [
                    mesh_files[i]
                ]
            }
        else:
            entry = {
                "id": i,
                "parent": 0,
                "name": f"joint_{i}",
                "joint": joint_type_dict[joint_info['type']],
                "jointData": {
                    "axis": {
                        "origin": joint_info['axis_position'].tolist(),
                        "direction": joint_info['axis_direction'].tolist()
                    },
                    "limit": {
                    }
                },
                "visuals": [
                    mesh_files[i]
                ]
            }
        meta_info.append(entry)
    with open(os.path.join(exp_dir, 'joint_info.json'), 'w') as f:
        json.dump(meta_info, f, indent=4)


def render_set(args, load2gpt_on_the_fly, name, iteration, views, gaussians, pipeline, background, deform,
               eval_app=False, visualize=False):
    model_path = args.model_path
    d_values_list = deform.step(gaussians, is_training=False)
    pred_joint_types = deform.deform.joint_types[1:]
    num_d_joints = len(pred_joint_types)
    pred_joint_list = deform.deform.get_joint_param(pred_joint_types)

    save_dir = os.path.join(model_path, name, "ours_{}".format(iteration))
    makedirs(save_dir, exist_ok=True)
    mesh_path = os.path.join(model_path, name, "ours_{}".format(iteration), "meshes")
    makedirs(mesh_path, exist_ok=True)
    mesh_starts = o3d.geometry.TriangleMesh()
    mesh_ends = o3d.geometry.TriangleMesh()
    # visualize axis
    centers = deform.deform.seg_model.center[1:].cpu().numpy()
    for i, joint_info in enumerate(pred_joint_list):
        pos = joint_info['axis_position']
        if pred_joint_types[i] == 'p':
            pos = centers[i]
        else:
            pos += joint_info['axis_direction'] * np.dot(joint_info['axis_direction'], centers[i] - pos)
        save_axis_mesh(joint_info['axis_direction'], pos,
                       f'{mesh_path}/axis_{i}_{pred_joint_types[i]}.ply')

    PSNR, SSIM, LPIPS = -1, -1, -1
    for mask_id in range(-1, num_d_joints + 1):
        torch.cuda.empty_cache()
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders", "{}".format(mask_id))
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt", "{}".format(mask_id))
        depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth", "{}".format(mask_id))

        makedirs(render_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)
        makedirs(depth_path, exist_ok=True)

        rgbs_start, depths_start = [], []
        rgbs_end, depths_end = [], []
        alphas_start, alphas_end = [], []
        gt_rgbs_start, gt_rgbs_end = [], []
        views_start, views_end = [], []

        if mask_id > 0 and 'real_' in args.source_path:  # filter noise gaussians for real-wolrd objects
            x = gaussians.get_xyz
            mask_part = d_values_list[0]['mask'] == mask_id
            _, mask_cluster = find_biggest_cluster(x[mask_part].cpu().numpy(), eps=0.05, min_samples=2)
            keep_mask = torch.ones(len(x), dtype=torch.bool).cuda()
            keep_mask[mask_part] = torch.tensor(mask_cluster, dtype=torch.bool).cuda()
        else:
            keep_mask = None
        for idx, view in enumerate(tqdm(views)):
            if load2gpt_on_the_fly:
                view.load2device()
            gt_image = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
            if view.fid < 0.5:
                gt_rgbs_start.append(gt_image)
                d_values = d_values_list[0]
                mask = d_values['mask']
                vis_mask = mask == mask_id if mask_id != -1 else None
                if keep_mask != None and vis_mask != None:
                    vis_mask = vis_mask & keep_mask
                d_xyz, d_rotation = d_values['d_xyz'], d_values['d_rotation']
                #results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, vis_mask=vis_mask)
                results = planar_render(view, gaussians, pipeline, background, d_xyz, d_rotation, vis_mask=vis_mask)
                views_start.append(view)
                rgbs_start.append(torch.clamp(results["render"], 0.0, 1.0))
                #alphas_start.append(torch.clamp(results["alpha"], 0.0, 1.0))

                if mask_id not in [-1,0]:
                    depth = render(view, gaussians, pipeline, background, d_xyz, d_rotation, vis_mask=vis_mask)['depth']
                    depths_start.append(depth)
                else:
                    depths_start.append(results['plane_depth'])
                    #depths_start.append(view.depth.cuda()[None])


            else:
                gt_rgbs_end.append(gt_image)
                d_values = d_values_list[1]
                mask = d_values['mask']
                vis_mask = mask == mask_id if mask_id != -1 else None
                if keep_mask != None and vis_mask != None:
                    vis_mask = vis_mask & keep_mask
                d_xyz, d_rotation = d_values['d_xyz'], d_values['d_rotation']
                results = planar_render(view, gaussians, pipeline, background, d_xyz, d_rotation, vis_mask=vis_mask)
                views_end.append(view)
                rgbs_end.append(torch.clamp(results["render"], 0.0, 1.0))
                if mask_id not in [-1, 0]:
                    depth = render(view, gaussians, pipeline, background, d_xyz, d_rotation, vis_mask=vis_mask)['depth']
                    depths_end.append(depth)
                else:
                    depths_end.append(results['plane_depth'])
                    # depths_start.append(view.depth.cuda()[None])
            # rendering_np = (
            #             results["render"].permute(1, 2, 0)[:, :, [2, 1, 0]] * 255).detach().cpu().numpy().astype(
            #     np.uint8)
            # cv2.imwrite(os.path.join(render_path, view.image_name + ".jpg"), rendering_np)

        if name == 'train':
            gsExtractor_start = GaussianExtractor(views_start, rgbs_start, depths_start, depth_trunc=5)
            mesh_start = gsExtractor_start.extract_mesh()
            gsExtractor_end = GaussianExtractor(views_end, rgbs_end, depths_end, depth_trunc=5)
            mesh_end = gsExtractor_end.extract_mesh()
            if mask_id != -1:
                mesh_starts  = mesh_starts + mesh_start
                mesh_ends  = mesh_ends + mesh_end

                save_path_start = os.path.join(mesh_path, f'start_{mask_id}.ply')
                save_path_end = os.path.join(mesh_path, f'end_{mask_id}.ply')

                o3d.io.write_triangle_mesh(save_path_start, mesh_start, write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
                o3d.io.write_triangle_mesh(save_path_end, mesh_end,  write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)

        rgbs = torch.stack(rgbs_start + rgbs_end, 0)
        depths = torch.stack(depths_start + depths_end, 0)
        gt_rgbs = torch.stack(gt_rgbs_start + gt_rgbs_end, 0)

        if mask_id == -1 and eval_app:
            PSNR, SSIM, LPIPS = 0, 0, 0
            for rgb, gt_rgb in zip(rgbs, gt_rgbs):
                PSNR += psnr(rgb[None], gt_rgb[None])
                SSIM += ssim_func(rgb[None], gt_rgb[None])
                LPIPS += lpips(rgb[None], gt_rgb[None])
            n = len(rgbs)
            PSNR, SSIM, LPIPS = PSNR / n, SSIM / n, LPIPS / n
            PSNR, SSIM, LPIPS = PSNR.item(), SSIM.item(), LPIPS.item()

        if visualize:
            for i in range(len(rgbs)):
                rgb = rgbs[i]
                gt = gt_rgbs[i]
                vis_depth(depths[i], os.path.join(depth_path, '{0:05d}'.format(i) + ".png"))
                torchvision.utils.save_image(rgb, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
                torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(i) + ".png"))

    w_save_path_start = os.path.join(mesh_path, f'start_-1.ply')
    w_save_path_end = os.path.join(mesh_path, f'end_-1.ply')

    o3d.io.write_triangle_mesh(w_save_path_start, mesh_starts, write_triangle_uvs=True, write_vertex_colors=True,
                               write_vertex_normals=True)
    o3d.io.write_triangle_mesh(w_save_path_end, mesh_ends, write_triangle_uvs=True, write_vertex_colors=True,
                               write_vertex_normals=True)

    gt_path = args.source_path + '/gt'

    # export joint info
    pred_joint_list = [{}] + pred_joint_list
    mesh_files = [f'meshes/start_{i}.ply' for i in range(len(pred_joint_list))]
    export_joint_info_json(pred_joint_list, mesh_files, save_dir)


def render_sets(args, dataset: ModelParams, iteration, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                mode: str, load2device_on_the_fly=False, render_mesh_seg= False):
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
        if mode == 'render':
            cam_traj = generate(cam_traj)

        if not skip_train:
            render_set(args, load2device_on_the_fly, "train", scene.loaded_iter, cam_traj, gaussians, pipeline,
                       background, deform, eval_app=args.eval_app, visualize=args.visualize)

        if not skip_test:
            render_set(args, load2device_on_the_fly, "test", scene.loaded_iter, scene.getTestCameras(), gaussians,
                       pipeline, background, deform, eval_app=args.eval_app, visualize=args.visualize)

        if render_mesh_seg:
            centers = deform.deform.seg_model.center[1:].cpu().numpy()
            pred_joint_types = deform.deform.joint_types[1:]
            pred_joint_list = deform.deform.get_joint_param(pred_joint_types)
            gt_info_list = read_gt(os.path.expanduser(f'{args.source_path}/gt/trans.json'))
            mesh_path = os.path.join(args.model_path, "train", "ours_{}".format(iteration), "meshes")
            for i, joint_info in enumerate(pred_joint_list):
                pos = joint_info['axis_position']
                if pred_joint_types[i].strip() == 'p':
                    pos = centers[i]
                else:
                    pos += joint_info['axis_direction'] * np.dot(joint_info['axis_direction'], centers[i] - pos)
                save_axis_mesh(joint_info['axis_direction'], pos,
                               f'{mesh_path}/axis_{i}_{pred_joint_types[i]}.ply')
                save_axis_mesh(gt_info_list[i]['axis_direction'], pos,
                               f'{mesh_path}/gt_axis_{i}_{pred_joint_types[i]}.ply')
            mask = deform.step(gaussians, is_training=False)[0]['mask']
            pallete = np.array(sns.color_palette("hls", mask.max() + 1)) 
            pallete[0] = np.array([0.737, 0.706, 0.663])
            pallete_gt = pallete

            num_d_joints = len(pred_joint_types)
            meshes = []
            for mask_id in range(num_d_joints + 1):
                color = pallete[mask_id][None, ...]

                mesh_path = os.path.join(args.model_path, "train", "ours_{}".format(iteration), "meshes")
                mesh_id_path = os.path.join(mesh_path, f'start_{mask_id}.ply')
                mesh = o3d.io.read_triangle_mesh(mesh_id_path)
                mesh.compute_vertex_normals()
                mesh.vertex_colors = o3d.utility.Vector3dVector(color.repeat(len(mesh.vertices), 0))
                meshes.append(mesh)
                try:
                    axis_path = os.path.join(mesh_path, f'axis_{mask_id}_{pred_joint_types[mask_id]}.ply')
                    axis = o3d.io.read_triangle_mesh(axis_path)
                    axis.paint_uniform_color([1, 0, 0])
                    meshes.append(axis)
                except:
                    pass
            o3d.visualization.draw_geometries(meshes)

            meshes = []

            for mask_id in range(num_d_joints + 1):
                color = pallete[mask_id][None, ...]
                mesh_path = os.path.join(args.model_path, "train", "ours_{}".format(iteration), "meshes")
                mesh_id_path = os.path.join(mesh_path, f'end_{mask_id}.ply')
                mesh = o3d.io.read_triangle_mesh(mesh_id_path)
                mesh.compute_vertex_normals()
                mesh.vertex_colors = o3d.utility.Vector3dVector(color.repeat(len(mesh.vertices), 0))
                meshes.append(mesh)
                try:
                    axis_path = os.path.join(mesh_path, f'axis_{mask_id}_{pred_joint_types[mask_id]}.ply')
                    axis = o3d.io.read_triangle_mesh(axis_path)
                    axis.paint_uniform_color([1,0,0])
                    meshes.append(axis)
                except:
                    pass
            o3d.visualization.draw_geometries(meshes)








if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_mesh_seg", action="store_true")
    parser.add_argument("--visualize", action="store_true", help="Visualize the rendered images")
    parser.add_argument("--eval_app", action="store_true",
                        help="Evaluate the rendered images with PSNR, SSIM, and LPIPS")
    parser.add_argument("--mode", default='eval', choices=['render', 'eval'])

    args = get_combined_args(parser)
    args.source_path = f'/media/wd/work/ArtGS_data/{args.dataset}/{args.subset}/{args.scene_name}'

    print("Rendering " + args.source_path + ' with ' + args.model_path)
    safe_state(args.quiet)
    seed_everything(args.seed)
    render_sets(args, model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,
                args.mode, load2device_on_the_fly=args.load2gpu_on_the_fly, render_mesh_seg = args.render_mesh_seg)
