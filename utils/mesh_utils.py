#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#
# copy from https://github.com/hbb1/2d-gaussian-splatting/blob/main/utils/mesh_utils.py
# and modified by Yu Liu

import torch
import numpy as np
from tqdm import tqdm
import open3d as o3d
import os

def post_process_mesh(mesh, cluster_to_keep=1):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    # print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    # n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    # n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    n_cluster = cluster_n_triangles.max() * 0.1

    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    # print("num vertices raw {}".format(len(mesh.vertices)))
    # print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        W = viewpoint_cam.image_width
        H = viewpoint_cam.image_height
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, 0, 1]]).float().cuda().T
        intrins =  (viewpoint_cam.projection_matrix @ ndc2pix)[:3,:3].T
        intrinsic=o3d.camera.PinholeCameraIntrinsic(
            width=viewpoint_cam.image_width,
            height=viewpoint_cam.image_height,
            cx = intrins[0,2].item(),
            cy = intrins[1,2].item(), 
            fx = intrins[0,0].item(), 
            fy = intrins[1,1].item()
        )

        extrinsic=np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj


def focus_point_fn(poses: np.ndarray) -> np.ndarray:
  """Calculate nearest point to all focal axes in poses."""
  directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
  m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
  mt_m = np.transpose(m, [0, 2, 1]) @ m
  focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
  return focus_pt


def pad_poses(p: np.ndarray) -> np.ndarray:
  """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
  bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
  return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p: np.ndarray) -> np.ndarray:
  """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
  return p[..., :3, :4]


def transform_poses_pca(poses: np.ndarray):
  """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
  """
  t = poses[:, :3, 3]
  t_mean = t.mean(axis=0)
  t = t - t_mean

  eigval, eigvec = np.linalg.eig(t.T @ t)
  # Sort eigenvectors in order of largest to smallest eigenvalue.
  inds = np.argsort(eigval)[::-1]
  eigvec = eigvec[:, inds]
  rot = eigvec.T
  if np.linalg.det(rot) < 0:
    rot = np.diag(np.array([1, 1, -1])) @ rot

  transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
  poses_recentered = unpad_poses(transform @ pad_poses(poses))
  transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

  # Flip coordinate system if z component of y-axis is negative
  if poses_recentered.mean(axis=0)[2, 1] < 0:
    poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
    transform = np.diag(np.array([1, -1, -1, 1])) @ transform

  return poses_recentered, transform

from utils.geo_utils import compute_pcd
from plyfile import PlyData, PlyElement


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


class GaussianExtractor(object):
    def __init__(self, viewpoint_stack, rgbmaps, depthmaps, depth_trunc=5.0, voxel_size=0.004, source_path = None): #voxel_size=0.004
        self.viewpoint_stack = viewpoint_stack
        self.rgbmaps = torch.stack(rgbmaps, dim=0)
        self.depthmaps = torch.stack(depthmaps, dim=0)
        self.path = source_path
        # self.estimate_bounding_sphere()
        # self.depth_trunc = (self.radius * 2.0)
        self.depth_trunc = depth_trunc
        self.voxel_size = voxel_size
        #self.sdf_trunc = 6 * voxel_size
        self.sdf_trunc = 5 * voxel_size
        print('depth_trunc: ', self.depth_trunc)
        print('voxel_size: ', self.voxel_size)

    def extract_mesh(self, planar_depth = False, mask_backgrond= True):
        mesh = self.extract_mesh_bounded(voxel_size=self.voxel_size, sdf_trunc=self.sdf_trunc, depth_trunc=self.depth_trunc, mask_backgrond=mask_backgrond)
        return post_process_mesh(mesh)
        #return mesh

    def extract_planar_mesh(self,  render_path, mask_backgrond= True):
        mesh = self.extract_planar_mesh_bounded(render_path, voxel_size=self.voxel_size, sdf_trunc=self.sdf_trunc, depth_trunc=self.depth_trunc, mask_backgrond=mask_backgrond)
        return post_process_mesh(mesh)
        #return mesh

    def estimate_bounding_sphere(self):
        """
        Estimate the bounding sphere given camera pose
        """
        torch.cuda.empty_cache()
        c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in self.viewpoint_stack])
        poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
        center = (focus_point_fn(poses))
        self.radius = np.linalg.norm(c2ws[:,:3,3] - center, axis=-1).min()
        self.center = torch.from_numpy(center).float().cuda()
        # print(f"The estimated bounding radius is {self.radius:.2f}")
        # print(f"Use at least {2.0 * self.radius:.2f} for depth_trunc")

    @torch.no_grad()
    def extract_mesh_bounded(self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, mask_backgrond=True, planar_depth = False):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.
        
        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        for i, cam_o3d in tqdm(enumerate(to_cam_open3d(self.viewpoint_stack)), desc="TSDF integration progress"):
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i]
            
            #if we have mask provided, use it
            if mask_backgrond and (self.viewpoint_stack[i].gt_alpha_mask is not None):
                depth[(self.viewpoint_stack[i].gt_alpha_mask < 0.5)] = 0
            if planar_depth:
                depth = depth.squeeze().detach().cpu().numpy()
                depth = o3d.geometry.Image((depth * 1000).astype(np.uint16))
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(np.asarray(rgb.permute(1, 2, 0).cpu().numpy() * 255, order="C", dtype=np.uint8)),
                    o3d.geometry.Image(depth),
                    depth_trunc=depth_trunc, convert_rgb_to_intensity=False,
                    depth_scale=1000.0
                )
            else:
                # make open3d rgbd
                depth = np.asarray(depth.permute(1,2,0).cpu().numpy())
                depth = o3d.geometry.Image((depth * 1000).astype(np.uint16))
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(np.asarray(rgb.permute(1,2,0).cpu().numpy() * 255, order="C", dtype=np.uint8)),
                    #o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy())),
                    depth,
                    depth_trunc = depth_trunc, convert_rgb_to_intensity=False,
                    depth_scale = 1000.0
                )

            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

        mesh = volume.extract_triangle_mesh()
        return mesh


    @torch.no_grad()
    def extract_planar_mesh_bounded(self, render_path, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, mask_backgrond=True):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.

        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=4.0*voxel_size,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        ref_depth = self.depthmaps[0]
        H, W = ref_depth.squeeze().shape
        for i, view in tqdm(enumerate(self.viewpoint_stack), desc="TSDF integration progress"):
            #rgb = self.rgbmaps[i]
            rgb = o3d.io.read_image(os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
            depth = self.depthmaps[i]
            #rgb = o3d.io.read_image(os.path.join(render_path, view.image_name + ".jpg"))
            # if we have mask provided, use it
            # if mask_backgrond and (self.viewpoint_stack[i].gt_alpha_mask is not None):
            #     depth[(self.viewpoint_stack[i].gt_alpha_mask < 0.5)] = 0
            pose = np.identity(4)
            pose[:3, :3] = view.R.transpose(-1, -2)
            pose[:3, 3] = view.T
            depth[ref_depth > depth_trunc] = 0
            depth = depth.squeeze().detach().cpu().numpy()
            depth = o3d.geometry.Image((depth * 1000).astype(np.uint16))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                #o3d.geometry.Image(np.asarray(rgb.permute(1, 2, 0).cpu().numpy() * 255, order="C", dtype=np.uint8)),
                rgb,
                o3d.geometry.Image(depth),
                depth_trunc=depth_trunc, convert_rgb_to_intensity=False,
                depth_scale=1000.0
            )
            volume.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(W, H, view.Fx, view.Fy, view.Cx, view.Cy),
                pose)

            #volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

        mesh = volume.extract_triangle_mesh()
        return mesh