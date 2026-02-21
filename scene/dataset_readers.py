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

import os
from PIL import Image
from typing import NamedTuple, Optional
from utils.graphics_utils import getWorld2View2, fov2focal, focal2fov
import numpy as np
import json
import cv2 as cv
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fid: float
    depth: Optional[np.array] = None
    mono_depth: Optional[np.array] = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    train_cameras_2s: list
    test_cameras_2s: list


def getNerfppNorm(cam_info, apply=False):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal
    cam_centers = []
    if apply:
        c2ws = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        if apply:
            c2ws.append(C2W)
        cam_centers.append(C2W[:3, 3:4])
    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal
    translate = -center
    if apply:
        c2ws = np.stack(c2ws, axis=0)
        c2ws[:, :3, -1] += translate
        c2ws[:, :3, -1] /= radius
        w2cs = np.linalg.inv(c2ws)
        for i in range(len(cam_info)):
            cam = cam_info[i]
            cam_info[i] = cam._replace(R=w2cs[i, :3, :3].T, T=w2cs[i, :3, 3])
        apply_translate = translate
        apply_radius = radius
        translate = 0
        radius = 1.
        return {"translate": translate, "radius": radius, "apply_translate": apply_translate, "apply_radius": apply_radius}
    else:
        return {"translate": translate, "radius": radius}
    

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'],
                        vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


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


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", no_bg=False, load_depth=True, load_mono_depth=True):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        fovy = contents["camera_angle_y"]

        frames = contents["frames"]
        # frames = sorted(frames, key=lambda x: int(os.path.basename(x['file_path']).split('.')[0].split('_')[-1]))
        frames = sorted(frames, key=lambda x: x['file_path'])
        for idx, frame in enumerate(frames):
            cam_name = frame["file_path"]
            if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.join(path, cam_name))), 'rgba')):
                cam_name = os.path.join(os.path.dirname(os.path.dirname(os.path.join(path, cam_name))), 'rgba', os.path.basename(cam_name)).replace('.jpg', '.png')
            if cam_name.endswith('jpg') or cam_name.endswith('png'):
                cam_name = os.path.join(path, cam_name)
            else:
                cam_name = os.path.join(path, cam_name + extension)
            frame_time = frame['time']

            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            try:
                im_data = np.array(image.convert("RGBA"))
            except:
                print(f'{image_path} is damaged')
                continue

            bg = np.array(
                [1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            mask = norm_data[..., 3:4]

            arr = norm_data[:, :, :3] 
            if no_bg:
                norm_data[:, :, :3] = norm_data[:, :, 3:4] * norm_data[:, :, :3] + bg * (1 - norm_data[:, :, 3:4])
            
            arr = np.concatenate([arr, mask], axis=-1)

            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGBA" if arr.shape[-1] == 4 else "RGB")

            FovY = fovy
            FovX = fovx

            idx = str(int(image_name)).zfill(3)
            depth_path = image_path.replace('rgba', 'depth')
            if load_depth and os.path.exists(depth_path):
                depth = cv.imread(depth_path, -1) / 1e3
                h, w = depth.shape
                if depth.size == mask.size:
                    depth[mask[..., 0] < 0.5] = 0
                else:
                    depth[cv.resize(mask[..., 0], [w, h], interpolation=cv.INTER_NEAREST) < 0.5] = 0
                depth[depth < 0.1] = 0
            else:
                depth = None

            mono_depth_path = image_path.replace('rgba', 'mono_depth')
            if load_mono_depth and os.path.exists(mono_depth_path):
                mono_depth = cv.imread(mono_depth_path, cv.IMREAD_GRAYSCALE) / 255
                h, w = mono_depth.shape
                if mono_depth.size == mask.size:
                    mono_depth[mask[..., 0] < 0.5] = 0
                else:
                    mono_depth[cv.resize(mask[..., 0], [w, h], interpolation=cv.INTER_NEAREST) < 0.5] = 0
            else:
                mono_depth = None

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth, mono_depth=mono_depth,
                                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], fid=frame_time))

    return cam_infos


def readInfo_2states(path, white_background, eval, extension=".png", no_bg=True):
    print("Reading Training Transforms")
    train_cam_infos = []
    test_cam_infos = []
    for state in ['start', 'end']:
        train_infos = readCamerasFromTransforms(
            path, f"transforms_train_{state}.json", white_background, extension, no_bg=no_bg)
        try:
            test_infos = readCamerasFromTransforms(
                path, f"transforms_test_{state}.json", white_background, extension, no_bg=no_bg)
        except:
            test_infos = []
        if not eval:
            train_infos.extend(test_infos)
        train_cam_infos.append(train_infos)
        test_cam_infos.append(test_infos)
        print(f"Read train_{state} transforms with {len(train_infos)} cameras")
        print(f"Read test_{state} transforms with {len(test_infos)} cameras")

    nerf_normalization = getNerfppNorm(train_cam_infos[0] + train_cam_infos[1])

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos[0] + train_cam_infos[1],
                           test_cameras=test_cam_infos[0] + test_cam_infos[1],
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           train_cameras_2s=train_cam_infos,
                           test_cameras_2s=test_cam_infos)
    return scene_info

def readDeformableCamerasFromPartNet(path, transformsfile, white_background, state, extension=".png"):
    cam_infos = []
    #states = ["start", "end"]
    #for state in states:
    state_path = os.path.join(path, state)
    mode = transformsfile.split(".")[0].split("_")[-1] + "/"

    with open(os.path.join(state_path, transformsfile)) as json_file:
        frames = json.load(json_file)
        frame_time = 0. if state == "start" else 1.
        #fovx = contents["camera_angle_x"]
        K = np.array(frames["K"]).astype(np.float32)
        fx, fy = K[0][0], K[1][1]
        sample_cam_name = os.path.join(state_path, mode + list(frames.keys())[1] + extension)
        sample_image_path = os.path.join(state_path, sample_cam_name)
        sample_image = Image.open(sample_image_path)
        img_width = sample_image.width
        img_height = sample_image.height
        #K_ = np.array(cam_dict['K']).astype(np.float32)
        fovx = 2 * np.arctan(0.5 * img_width / fx)
        #_sample = np.array(sample_image)
        fovy = focal2fov(fov2focal(fovx, sample_image.size[0]), sample_image.size[1])
        #_fovy = focal2fov(fov2focal(fovx, img_height), img_width)

        # focal = K_[0][0] * img_scale
        # K = torch.tensor(K_)
        # K[0][0], K[1][1] = focal, focal
        # K[0][2], K[1][2] = self.w / 2, self.h / 2
        frames.pop("K")
        #frames = contents["frames"]
        #for idx, frame in enumerate(frames):
        idx = 0
        for image_name, transform_matrix in frames.items():
            #cam_name = os.path.join(path, mode+  frame["file_path"] + extension)
            cam_name = os.path.join(state_path, mode + image_name + extension)
            # NeRF 'transform_matrix' is a camera-to-world transform
            #c2w = np.array(frame["transform_matrix"])
            c2w = np.array(transform_matrix)
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(state_path, cam_name)
            #image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0

            mask = norm_data[..., 3:4]

            arr = norm_data[:, :, :3]

            norm_data[:, :, :3] = norm_data[:, :, 3:4] * norm_data[:, :, :3] + bg * (1 - norm_data[:, :, 3:4])

            arr = np.concatenate([arr, mask], axis=-1)

            #arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            #image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGBA" if arr.shape[-1] == 4 else "RGB")
            #depth = None
            mono_depth = None

            depth_path = image_path.replace('mode', 'depth')
            if os.path.exists(depth_path):
                depth = cv.imread(depth_path, 0) / 1e3
                h, w = depth.shape
                if depth.size == mask.size:
                    depth[mask[..., 0] < 0.5] = 0
                else:
                    depth[cv.resize(mask[..., 0], [w, h], interpolation=cv.INTER_NEAREST) < 0.5] = 0
                depth[depth < 0.1] = 0
            else:
                depth = None

            # cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=fovy, FovX=fovx, image=image,
            #                             image_path=image_path, image_name=image_name, width=image.size[0],
            #                             height=image.size[1], fx =fx, fy=fy))
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=fovy, FovX=fovx, image=image, depth=depth, mono_depth=mono_depth,
                                        image_path=image_path, image_name=image_name, width=image.size[0],
                                        height=image.size[1],  fid=frame_time))

    return cam_infos


def readREArtGS(path, white_background, eval, extension=".png", sparse_view_num = None):
    train_cam_infos = []
    test_cam_infos = []

    for state in ['start', 'end']:
        print("Reading Training Transforms")
        train_infos = readDeformableCamerasFromPartNet(path, "camera_train.json", white_background, state, extension)
        print("Reading Test Transforms")
        test_infos = readDeformableCamerasFromPartNet(path, "camera_test.json", white_background, state, extension)
        if not eval:
            train_infos.extend(test_infos)
            test_cam_infos = []

        if sparse_view_num is not None:  # means sparse setting
            #eval = False
            assert os.path.exists(
                os.path.join(path, state, f"sparse_{str(sparse_view_num)}.txt")), "sparse_id.txt not found!"
            ids = np.loadtxt(os.path.join(path, state, f"sparse_{str(sparse_view_num)}.txt"), dtype=np.int32)
            # ids_test = np.loadtxt(osp.join(path, f"sparse_test.txt"), dtype=np.int32)
            # test_cam_infos = [train_cam_infos[i] for i in ids_test]
            train_infos = [train_infos[i] for i in ids]
            print("Sparse view, only {} images are used for training, others are used for eval.".format(len(ids)))

        train_cam_infos.append(train_infos)
        test_cam_infos.append(test_infos)
    nerf_normalization = getNerfppNorm(train_cam_infos[0] + train_cam_infos[1])




    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000

        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    # scene_info = SceneInfo(point_cloud=pcd,
    #                        train_cameras=train_cam_infos,
    #                        test_cameras=test_cam_infos,
    #                        nerf_normalization=nerf_normalization,
    #                        ply_path=ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos[0] + train_cam_infos[1],
                           test_cameras=test_cam_infos[0] + test_cam_infos[1],
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           train_cameras_2s=train_cam_infos,
                           test_cameras_2s=test_cam_infos)
    return scene_info


# def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, state, white_background = True):
#     import sys
#     cam_infos = []
#     for idx, key in enumerate(cam_extrinsics):
#         sys.stdout.write('\r')
#         # the exact output you're looking for:
#         sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
#         sys.stdout.flush()
#
#         extr = cam_extrinsics[key]
#         intr = cam_intrinsics[extr.camera_id]
#         height = intr.height
#         width = intr.width
#
#         uid = intr.id
#         R = np.transpose(qvec2rotmat(extr.qvec))
#         T = np.array(extr.tvec)
#
#         if intr.model == "SIMPLE_PINHOLE":
#             focal_length_x = intr.params[0]
#             FovY = focal2fov(focal_length_x, height)
#             FovX = focal2fov(focal_length_x, width)
#         elif intr.model == "PINHOLE":
#             focal_length_x = intr.params[0]
#             focal_length_y = intr.params[1]
#             FovY = focal2fov(focal_length_y, height)
#             FovX = focal2fov(focal_length_x, width)
#         else:
#             assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
#
#         image_path = os.path.join(images_folder, os.path.basename(extr.name))
#         image_name = os.path.basename(image_path).split(".")[0]
#         mask_folder = images_folder.replace("images", "masks")
#         mask_path = os.path.join(mask_folder, os.path.basename(extr.name).replace("jpg", "png") )
#         if state == 'start':
#             frame_time = 0.0
#         else:
#             frame_time = 1.0
#
#         if not os.path.exists(image_path) or "sky_mask" in image_path:
#             print("skip =====", image_path)
#             continue
#
#         image = Image.open(image_path)
#         bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
#         im_data = np.array(image.convert("RGBA"))
#         norm_data = im_data / 255.0
#         mask = norm_data[..., 3:4]
#
#         arr = norm_data[:, :, :3]
#
#         norm_data[:, :, :3] = norm_data[:, :, 3:4] * norm_data[:, :, :3] + bg * (1 - norm_data[:, :, 3:4])
#
#         arr = np.concatenate([arr, mask], axis=-1)
#
#         image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGBA" if arr.shape[-1] == 4 else "RGB")
#         cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
#                             image_path=image_path, image_name=image_name, width=width,
#                             height=height, fid=frame_time)
#         cam_infos.append(cam_info)
#     sys.stdout.write('\n')
#     return cam_infos


# def readColmapSceneInfo(path, white_background, eval, extension=".png"):
#     llffhold = 8
#     train_cam_infos = []
#     test_cam_infos = []
#
#     for state in ['start', 'end']:
#         state_path = state + '_col'
#         try:
#             cameras_extrinsic_file = os.path.join(path, state_path, "sparse/0", "images.bin")
#             cameras_intrinsic_file = os.path.join(path, state_path, "sparse/0", "cameras.bin")
#             cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
#             cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
#         except:
#             cameras_extrinsic_file = os.path.join(path, state_path, "sparse/0", "images.txt")
#             cameras_intrinsic_file = os.path.join(path, state_path, "sparse/0", "cameras.txt")
#             cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
#             cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
#
#         reading_dir = "images"
#         cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, state=state, images_folder=os.path.join(path, state_path, reading_dir), white_background=white_background)
#         cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
#
#         # if eval:
#         #     train_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
#         #     test_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
#         # else:
#         train_infos = cam_infos
#         test_infos = []
#         train_cam_infos.append(train_infos)
#         test_cam_infos.append(test_infos)
#     nerf_normalization = getNerfppNorm(train_cam_infos[0] + train_cam_infos[1])
#
#     # ply_path = os.path.join(path, state_path, "sparse/0/points3D.ply")
#     # bin_path = os.path.join(path, state_path, "sparse/0/points3D.bin")
#     # txt_path = os.path.join(path, state_path, "sparse/0/points3D.txt")
#     # if not os.path.exists(ply_path):
#     #     print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
#     #     try:
#     #         xyz, rgb, _ = read_points3D_binary(bin_path)
#     #     except:
#     #         xyz, rgb, _ = read_points3D_text(txt_path)
#     #     storePly(ply_path, xyz, rgb)
#     # try:
#     #     pcd = fetchPly(ply_path)
#     # except:
#     #     pcd = None
#
#     ply_path = os.path.join(path, "points3d.ply")
#     if not os.path.exists(ply_path):
#         # Since this data set has no colmap data, we start with random points
#         num_pts = 100_000
#
#         print(f"Generating random point cloud ({num_pts})...")
#
#         # We create random points inside the bounds of the synthetic Blender scenes
#         xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
#         shs = np.random.random((num_pts, 3)) / 255.0
#         pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
#
#         storePly(ply_path, xyz, SH2RGB(shs) * 255)
#     try:
#         pcd = fetchPly(ply_path)
#     except:
#         pcd = None
#
#
#     scene_info = SceneInfo(point_cloud=pcd,
#                            train_cameras=train_cam_infos[0] + train_cam_infos[1],
#                            test_cameras=test_cam_infos[0] + test_cam_infos[1],
#                            nerf_normalization=nerf_normalization,
#                            ply_path=ply_path,
#                            train_cameras_2s=train_cam_infos,
#                            test_cameras_2s=test_cam_infos)
#     return scene_info
def readColmapCameras(cam_extrinsics, cam_intrinsics, start_num, images_folder, white_background = True):
    import sys
    start_cam_infos = []
    end_cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        mask_folder = images_folder.replace("images", "masks")
        #mask_path = os.path.join(mask_folder, os.path.basename(extr.name).replace("jpg", "png") )
        if int(image_name) <= start_num:
            frame_time = 0.0
        else:
            frame_time = 1.0

        if not os.path.exists(image_path) or "sky_mask" in image_path:
            print("skip =====", image_path)
            continue

        image = Image.open(image_path)
        bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
        im_data = np.array(image.convert("RGBA"))
        norm_data = im_data / 255.0
        mask = norm_data[..., 3:4]

        arr = norm_data[:, :, :3]

        norm_data[:, :, :3] = norm_data[:, :, 3:4] * norm_data[:, :, :3] + bg * (1 - norm_data[:, :, 3:4])

        arr = np.concatenate([arr, mask], axis=-1)

        image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGBA" if arr.shape[-1] == 4 else "RGB")
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=width,
                            height=height, fid=frame_time)
        if frame_time == 0.0:
            start_cam_infos.append(cam_info)
        else:
            end_cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return start_cam_infos, end_cam_infos


def readColmapSceneInfo(path, white_background, start_num):

    train_cam_infos = []
    test_cam_infos = []


    try:
        cameras_extrinsic_file = os.path.join(path, "all", "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "all", "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "all", "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "all", "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images"
    start_cam_infos_unsorted, end_cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, start_num=start_num, images_folder=os.path.join(path, "all", reading_dir), white_background=white_background)
    start_cam_infos = sorted(start_cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    end_cam_infos = sorted(end_cam_infos_unsorted.copy(), key=lambda x: x.image_name)
    # if eval:
    #     train_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
    #     test_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    # else:

    test_infos = []
    train_cam_infos.append(start_cam_infos)
    train_cam_infos.append(end_cam_infos)
    test_cam_infos.append(test_infos)
    test_cam_infos.append(test_infos)
    nerf_normalization = getNerfppNorm(train_cam_infos[0] + train_cam_infos[1])

    ply_path = os.path.join(path, "all", "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "all", "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "all", "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    # ply_path = os.path.join(path, "points3d.ply")
    # if not os.path.exists(ply_path):
    #     # Since this data set has no colmap data, we start with random points
    #     num_pts = 100_000
    #
    #     print(f"Generating random point cloud ({num_pts})...")
    #
    #     # We create random points inside the bounds of the synthetic Blender scenes
    #     xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
    #     shs = np.random.random((num_pts, 3)) / 255.0
    #     pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    #
    #     storePly(ply_path, xyz, SH2RGB(shs) * 255)
    # try:
    #     pcd = fetchPly(ply_path)
    # except:
    #     pcd = None


    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos[0] + train_cam_infos[1],
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           train_cameras_2s=train_cam_infos,
                           test_cameras_2s=test_cam_infos)
    return scene_info


sceneLoadTypeCallbacks = {
    "Blender": readInfo_2states,
    "REArtGS": readREArtGS,
    "colmap": readColmapSceneInfo
}
