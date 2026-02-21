# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# copy from https://github.com/NVlabs/DigitalTwinArt/tree/master, modified by Yu Liu

import open3d as o3d
import numpy as np
import os
import joblib
from sklearn.cluster import DBSCAN

os.environ['PYOPENGL_PLATFORM'] = 'egl'


glcam_in_cvcam = np.array([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]])


def toOpen3dCloud(points, colors=None, normals=None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        if colors.max() > 1:
            colors = colors / 255.0
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    return cloud


def depth2xyzmap(depth, K):
    invalid_mask = depth < 0.1
    H, W = depth.shape[:2]
    vs, us = np.meshgrid(np.arange(0, H), np.arange(0, W), sparse=False, indexing='ij')
    vs = vs.reshape(-1)
    us = us.reshape(-1)
    zs = depth.reshape(-1)
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)  # (N,3)
    xyz_map = pts.reshape(H, W, 3).astype(np.float32)
    xyz_map[invalid_mask] = 0
    return xyz_map.astype(np.float32)


def find_biggest_cluster(pts, eps=0.005, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    dbscan.fit(pts)
    ids, cnts = np.unique(dbscan.labels_, return_counts=True)
    best_id = ids[cnts.argsort()[-1]]
    keep_mask = dbscan.labels_ == best_id
    pts_cluster = pts[keep_mask]
    return pts_cluster, keep_mask


def compute_translation_scales(pts, max_dim=2, cluster=True, eps=0.005, min_samples=5):
    if cluster:
        pts, keep_mask = find_biggest_cluster(pts, eps, min_samples)
    else:
        keep_mask = np.ones((len(pts)), dtype=bool)
    max_xyz = pts.max(axis=0)
    min_xyz = pts.min(axis=0)
    center = (max_xyz + min_xyz) / 2
    sc_factor = max_dim / (max_xyz - min_xyz).max()  # Normalize to [-1,1]
    sc_factor *= 0.9
    translation_cvcam = -center
    return translation_cvcam, sc_factor, keep_mask


def compute_pcd_worker(K, glcam_in_world, rgb, depth, mask):
    xyz_map = depth2xyzmap(depth, K)
    valid = depth >= 0.1
    valid = valid & (mask > 0)
    pts = xyz_map[valid].reshape(-1, 3)
    if len(pts) == 0:
        return None
    colors = rgb[valid].reshape(-1, 3)

    pcd = toOpen3dCloud(pts, colors)

    pcd = pcd.voxel_down_sample(0.01)
    new_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    cam_in_world = glcam_in_world @ glcam_in_cvcam
    new_pcd.transform(cam_in_world)

    return np.asarray(new_pcd.points).copy(), np.asarray(new_pcd.colors).copy()


def compute_pcd(glcam_in_worlds, K, rgbs=None, depths=None,
                         masks=None, cluster=False, eps=0.075, min_samples=5):

    args = []
    for i in range(len(rgbs)):
        args.append((K, glcam_in_worlds[i], rgbs[i], depths[i], masks[i]))

    ret = joblib.Parallel(n_jobs=6, prefer="threads")(joblib.delayed(compute_pcd_worker)(*arg) for arg in args)

    pcd_all = None
    for r in ret:
        if r is None or len(r[0]) == 0:
            continue
        if pcd_all is None:
            pcd_all = toOpen3dCloud(r[0], r[1])
        else:
            pcd_all += toOpen3dCloud(r[0], r[1])

    pcd = pcd_all.voxel_down_sample(eps / 5)
    pts = np.asarray(pcd.points).copy()
    _, _, keep_mask = compute_translation_scales(pts, cluster=cluster, eps=eps,
                                                                            min_samples=min_samples)
    return pts[keep_mask], np.asarray(pcd.colors)[keep_mask]




