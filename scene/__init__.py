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
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos
from utils.other_utils import match_gaussians, cal_cluster_centers


class Scene:
    gaussians: GaussianModel
    def __init__(self, args: ModelParams, gaussians: GaussianModel, gaussians1=None, load_iteration=None, init_with_random_pcd=False, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.args = args

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = [{},{}] # 0: start, 1: end
        self.test_cameras = [{},{}] # 0: start, 1: end
        if os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "start")):
            print("Found camera_train.json file, assuming d-PartNet set!")
            scene_info = sceneLoadTypeCallbacks["REArtGS"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "all")):
            print("Found camera_train.json file, assuming colmap set!")
            scene_info = sceneLoadTypeCallbacks["colmap"](args.source_path, args.white_background, args.start_num)
        else:
            raise ValueError("No scene info file found!")
    
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print("Cameras extent: ", self.cameras_extent)
        print("Loading Cameras")
        for i in range(2):
            for resolution_scale in resolution_scales:
                self.train_cameras[i][resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras_2s[i], resolution_scale, args)
                self.test_cameras[i][resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras_2s[i], resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"),
                                    og_number_points=len(scene_info.point_cloud.points))
        elif init_with_random_pcd:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
        
        if gaussians1 is not None:
            self.gaussians1 = gaussians1
            self.gaussians1.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
    

    def save(self, iteration, is_best=False):
        if is_best:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_best")
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
            with open(os.path.join(point_cloud_path, "iter.txt"), 'w') as f:
                f.write(f"iteration: {iteration}")
        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def save_2gs(self, iteration, num_slots, vis_cano=False, vis_center=False):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud_0.ply"))
        self.gaussians1.save_ply(os.path.join(point_cloud_path, "point_cloud_1.ply"))
        cano_gs = GaussianModel(self.gaussians.max_sh_degree)
        large_motion_state = match_gaussians(os.path.join(point_cloud_path, "point_cloud.ply"), cano_gs, num_slots, vis_cano)
        cal_cluster_centers(os.path.join(point_cloud_path, "point_cloud.ply"), num_slots, vis_center)
        return large_motion_state
    
    def getTrainCameras_start(self, scale=1.0):
        return self.train_cameras[0][scale]
    
    def getTrainCameras_end(self, scale=1.0):
        return self.train_cameras[1][scale]
    
    def getTestCameras_start(self, scale=1.0):
        return self.test_cameras[0][scale]
    
    def getTestCameras_end(self, scale=1.0):
        return self.test_cameras[1][scale]

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[0][scale] + self.train_cameras[1][scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[0][scale] + self.test_cameras[1][scale]
