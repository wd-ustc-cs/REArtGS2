import os
from pathlib import Path
import pycolmap
import numpy as np
from pycolmap import Rigid3d

def center_colmap_reconstruction(colmap_path, output_path):
    """
    直接处理COLMAP重建结果
    """
    # 读取重建结果
    reconstruction = pycolmap.Reconstruction(colmap_path)

    # 计算场景中心（使用所有3D点）
    all_points = []
    for point3D_id, point3D in reconstruction.points3D.items():
        all_points.append(point3D.xyz)

    if not all_points:
        # 如果没有3D点，使用相机位置
        all_points = []
        for image_id, image in reconstruction.images.items():
            camera_center = image.projection_center()
            all_points.append(camera_center)

    scene_center = np.mean(all_points, axis=0)

    # 平移所有3D点
    for point3D_id, point3D in reconstruction.points3D.items():
        point3D.xyz -= scene_center

    # 更新所有相机位姿
    for image_id, image in reconstruction.images.items():
        # 新的平移向量：t' = t - R * center
        # 注意：这里我们直接更新变换矩阵中的平移部分
        R = image.cam_from_world.rotation.matrix()
        t = image.cam_from_world.translation
        new_t = t - R @ scene_center

        # 更新图像的变换矩阵
        # 可能需要重新创建 cam_from_world 变换

        image.cam_from_world = Rigid3d(
            rotation=image.cam_from_world.rotation,
            translation=new_t
        )

    # 保存中心化后的重建结果
    reconstruction.write(output_path)

    return scene_center


def center_scene_using_transform(colmap_path, output_path):
    reconstruction = pycolmap.Reconstruction(colmap_path)

    # 计算场景中心
    points = np.array([p.xyz for p in reconstruction.points3D.values()])
    #scene_center = np.mean(points, axis=0)
    #scene_center = np.array([-0.13, 2.05, 1.72]) # toy
    #scene_center = np.array([-0.24, 1.26, 3.10]) # handwasher
    scene_center = np.array([0.78, 1.88, 1.62]) # drawer
    # 构建一个4x4的变换矩阵，将场景中心平移到原点
    # 这是一个简单的平移变换，没有旋转和缩放
    R =  pycolmap.Rotation3d(np.eye(3))
    t = -scene_center
    scale = 1.0
    transform_matrix =pycolmap.Sim3d(scale=scale, rotation=R, translation=t)

    # 应用变换
    reconstruction.transform(transform_matrix)

    reconstruction.write(output_path)
colmap_path = "/media/wd/work/ArtGS_data/reartgs/real_world/drawer/all/sparse/0"
output_path = "/media/wd/work/ArtGS_data/reartgs/real_world/drawer/all/sparse/0_norm"
os.makedirs(output_path, exist_ok= True)
center_scene_using_transform(colmap_path, output_path)