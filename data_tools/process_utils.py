import os
import cv2
import math
import json
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from plyfile import PlyData, PlyElement
from utils.geo_utils import compute_pcd
from sklearn.cluster import SpectralClustering
from utils.other_utils import match_pcd
from pytorch3d.loss import chamfer_distance


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def generate_pcd(scene, K, poses_start, poses_end, num_slots, reprocess=False, cluster=False, visualize=False):
    path_start = f'{scene}/point_cloud_start.ply'
    scene_name = os.path.basename(scene)
    if not os.path.exists(path_start) or reprocess:
        rgbs = [np.array(Image.open(f'{scene}/start/train/rgba/{str(i).zfill(4)}.png').convert('RGBA')) for i in range(len(poses_start))]
        masks = [rgb[..., -1] for rgb in rgbs]
        colors = [rgb[..., :-1] for rgb in rgbs]
        depths = [np.array(cv2.imread(f'{scene}/start/train/depth/{str(i).zfill(4)}.png', -1)) / 1e3 for i in range(len(poses_start))]
        if depths[0].shape != rgbs[0].shape:
            depths = [cv2.resize(d, (rgbs[0].shape[1], rgbs[0].shape[0]), interpolation=cv2.INTER_NEAREST) for d in depths]
        glcam_in_worlds = [np.array(pose) for pose in poses_start]
        xyz_start, color_start = compute_pcd(glcam_in_worlds, K, colors, depths, masks, cluster=cluster)
        storePly(path_start, xyz_start, color_start * 255)
        print('Processing start point clouds for', scene_name, 'with', xyz_start.shape[0], 'points')
    
    path_end = f'{scene}/point_cloud_end.ply'
    if not os.path.exists(path_end) or reprocess:
        rgbs = [np.array(Image.open(f'{scene}/end/train/rgba/{str(i).zfill(4)}.png').convert('RGBA')) for i in range(len(poses_end))]
        masks = [rgb[..., -1] for rgb in rgbs]
        colors = [rgb[..., :-1] for rgb in rgbs]
        depths = [np.array(cv2.imread(f'{scene}/end/train/depth/{str(i).zfill(4)}.png', -1)) / 1e3 for i in range(len(poses_end))]
        if depths[0].shape != rgbs[0].shape:
            depths = [cv2.resize(d, (rgbs[0].shape[1], rgbs[0].shape[0]), interpolation=cv2.INTER_NEAREST) for d in depths]
        glcam_in_worlds = [np.array(pose) for pose in poses_end]
        xyz_end, color_end = compute_pcd(glcam_in_worlds, K, colors, depths, masks, cluster=cluster)
        storePly(path_end, xyz_end, color_end * 255)
        print('Processing end point clouds for', scene_name, 'with', xyz_end.shape[0], 'points')

    path_cano = f'{scene}/point_cloud_cano.ply'
    if not os.path.exists(path_cano) or reprocess:
        print('Processing canonical point clouds for', scene_name)
        pcd_start = o3d.io.read_point_cloud(path_start)
        pcd_end = o3d.io.read_point_cloud(path_end)
        xyzs = [np.asarray(pcd.points, np.float32) for pcd in [pcd_start, pcd_end]]
        colors = [np.asarray(pcd.colors, np.float32) for pcd in [pcd_start, pcd_end]]
        
        pc0, pc1 = torch.tensor(xyzs[0])[None].cuda(), torch.tensor(xyzs[1])[None].cuda()
        idx = match_pcd(pc0, pc1) # idx: [idx_start, idx_end]
        cd, _ = chamfer_distance(pc0, pc1, batch_reduction=None, point_reduction=None) # cd: [cd_start2end, cd_end2start]
    
        larger_motion_state = 0 if cd[0].mean().item() > cd[1].mean().item() else 1
        print("Larger motion state: ", larger_motion_state)

        threshould = [0.02 * cd[0].max().item(), 0.02 * cd[1].max().item()]
        mask_static = [(cd[i].squeeze() < threshould[i]).cpu().numpy() for i in range(2)]
        mask_dynamic = [(cd[i].squeeze() >= threshould[i]).cpu().numpy() for i in range(2)]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyzs[0])
        pcd.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [1, 0, 0]])[mask_dynamic[0].astype(np.int32)])
        o3d.visualization.draw_geometries([pcd])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyzs[1])
        pcd.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [1, 0, 0]])[mask_dynamic[1].astype(np.int32)])
        o3d.visualization.draw_geometries([pcd])
        s0 = larger_motion_state
        xyz_static = xyzs[s0][mask_static[s0]]
        xyz_dynamic = xyzs[s0][mask_dynamic[s0]]
        xyz = np.concatenate([xyz_static, (xyzs[0][idx[0]] + xyzs[1][idx[1]]) * 0.5])
        print(f"Number of points: start--{xyzs[0].shape}, end--{xyzs[1].shape}, cano--{xyz.shape}")
        color = np.concatenate([colors[s0][mask_static[s0]], (colors[0][idx[0]] + colors[1][idx[1]]) * 0.5])
        storePly(path_cano, xyz, color * 255)
    
        path_center = f'{scene}/center_info.npy'
        if not os.path.exists(path_center) or reprocess:
            if num_slots > 2:
                cluster = SpectralClustering(n_clusters=num_slots - 1, assign_labels='discretize', random_state=0, gamma=10)
                labels = cluster.fit_predict(xyz_dynamic)
                center_dynamic = np.array([xyz_dynamic[labels == i].mean(0) for i in range(num_slots - 1)])
                labels = np.concatenate([np.zeros(xyz_static.shape[0]), labels + 1])
                center = np.concatenate([xyz_static.mean(0, keepdims=True), center_dynamic])
            else:
                labels = np.concatenate([np.zeros(xyz_static.shape[0]), np.ones(xyz_dynamic.shape[0])])
                center = np.concatenate([xyz_static.mean(0, keepdims=True), xyz_dynamic.mean(0, keepdims=True)])
            x = np.concatenate([xyz_static, xyz_dynamic])
            labels = np.asarray(labels, np.int32)
            dist = (x - center[labels]) # [N, 3]
            mask = np.zeros([dist.shape[0], num_slots])
            mask[np.arange(dist.shape[0]), labels] = 1
            dist_max = (np.linalg.norm(dist, axis=-1)[:, None] * mask).max(0)[:, None] / 2 # [K, 1]
            center_info = np.concatenate([center, dist_max], -1)
            np.save(path_center, center_info)

    if visualize:
        import seaborn as sns
        pallete = np.array(sns.color_palette("hls", num_slots))
        center = np.load(f'{scene}/center_info.npy')[:, :3]
        xyz_center = (center[None] + np.random.randn(1000, 1, 3) * 0.02).reshape(-1, 3)
        color_center = pallete[None].repeat(1000, 0).reshape(-1, 3)

        pcd_start = o3d.io.read_point_cloud(path_start)
        pcd_end = o3d.io.read_point_cloud(path_end)
        pcd_cano = o3d.io.read_point_cloud(path_cano)
        o3d.visualization.draw_geometries([pcd_cano])
        xyzs = [np.asarray(pcd.points, np.float32) for pcd in [pcd_start, pcd_end]] + [xyz_center]
        colors = [np.asarray(pcd.colors, np.float32) for pcd in [pcd_start, pcd_end]] + [color_center]
        
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.concatenate(xyzs))
        point_cloud.colors = o3d.utility.Vector3dVector(np.concatenate(colors))
        o3d.visualization.draw_geometries([point_cloud])


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def saveTransformFilesOneScene(start_poses, end_poses, split, fov_x, fov_y, scene_path):
    transforms = {
            "camera_angle_x": fov_x,
            "camera_angle_y": fov_y,
            "frames": [],
        }

    with open(f'{scene_path}/transforms_{split}.json', 'w') as f:
        for i, pose in enumerate(start_poses):
            info = {
                "file_path": f'start/{split}/rgba/{str(i).zfill(4)}.png',
                "time": 0.,
                "transform_matrix": pose,
            }
            transforms["frames"].append(info)
        for i, pose in enumerate(end_poses):
            info = {
                "file_path": f'end/{split}/rgba/{str(i).zfill(4)}.png',
                "time": 1.,
                "transform_matrix": pose,
            }
            transforms["frames"].append(info)
        json.dump(transforms, f, indent=4)


def saveTransformFilesOneScene2s(start_poses, end_poses, split, fov_x, fov_y, scene_path):
    transforms = {
            "camera_angle_x": fov_x,
            "camera_angle_y": fov_y,
            "frames": [],
        }
    with open(f'{scene_path}/transforms_{split}_start.json', 'w') as f:
        for i, pose in enumerate(start_poses):
            info = {
                "file_path": f'start/{split}/rgba/{str(i).zfill(4)}.png',
                "time": 0.,
                "transform_matrix": pose,
            }
            transforms["frames"].append(info)
        json.dump(transforms, f, indent=4)
    
    transforms = {
            "camera_angle_x": fov_x,
            "camera_angle_y": fov_y,
            "frames": [],
        }
    with open(f'{scene_path}/transforms_{split}_end.json', 'w') as f:
        for i, pose in enumerate(end_poses):
            info = {
                "file_path": f'end/{split}/rgba/{str(i).zfill(4)}.png',
                "time": 1.,
                "transform_matrix": pose,
            }
            transforms["frames"].append(info)
        json.dump(transforms, f, indent=4)