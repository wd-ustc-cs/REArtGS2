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
import sys

import numpy as np
import tqdm
import torch
import torchvision
from random import randint
from utils.loss_utils import l1_loss, ssim, get_img_grad_weight, lncc
from utils.image_utils import psnr, erode
from utils.vote_utils import BoundaryPointVotingModule, LocalConsistencyVotingLoss
from utils.graphics_utils import patch_offsets, patch_warp, fov2focal
from utils.dual_quaternion import axis_angle_to_quaternion, joint_params_to_dual_quaternion
from gaussian_renderer import render, planar_render
from scene import Scene, GaussianModel, DeformModel
from scene.app_model import AppModel
from utils.camera_utils import gen_virtul_cam
from utils.general_utils import safe_state, get_linear_noise_func, vis_depth
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.log_utils import prepare_output_and_logger, training_report
from pytorch_lightning import seed_everything
from utils.metrics import *
from pytorch3d.loss import chamfer_distance
from utils.depth_loss import DepthLoss
import dearpygui.dearpygui as dpg
from utils.gui_utils import OrbitCamera
from utils.image_utils import depth_tensor_to_img, normal_tensor_to_img
import time
import random
import seaborn as sns


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar, fid):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.fid = fid
        self.Fx = fov2focal(fovx, self.image_width)
        self.Fy = fov2focal(fovy, self.image_height)
        self.Cx = 0.5 * self.image_width
        self.Cy = 0.5 * self.image_height
        w2c = np.linalg.inv(c2w)

        # rectify...
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1

        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda()
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = -torch.tensor(c2w[:3, 3]).cuda()

    def get_calib_matrix_nerf(self, scale=1.0):
        intrinsic_matrix = torch.tensor(
            [[self.Fx / scale, 0, self.Cx / scale], [0, self.Fy / scale, self.Cy / scale], [0, 0, 1]]).float()
        extrinsic_matrix = self.world_view_transform.transpose(0, 1).contiguous()  # cam2world
        return intrinsic_matrix, extrinsic_matrix

class Trainer:
    def __init__(self, args, dataset, opt, pipe, saving_iterations):
        self.dataset = dataset
        self.args = args
        self.opt = opt
        self.pipe = pipe
        self.saving_iterations = saving_iterations
        self.test_iterations  = args.test_iterations
        self.tb_writer = prepare_output_and_logger(args)
        self.report_mask = True
        self.gaussians = GaussianModel(dataset.sh_degree, use_app = args.app_model)
        self.deform = DeformModel(self.dataset)
        print('Init GaussianModel and DeformModel.')
        #self.scene = Scene(dataset, self.gaussians, load_iteration=-1)
        self.scene = Scene(dataset, self.gaussians)
        if args.app_model:
            self.app_model = AppModel().train().cuda()
        else:
            self.app_model = None
        if self.args.canonical_init == 'cgs':
            #p = args.source_path.replace('data/', 'outputs/')
            p = args.source_path.replace('ArtGS_data/', 'REArtGS2/outputs/')
            coarse_name = self.args.coarse_name
            self.xyzs = self.gaussians.load_ply_cano(f'{p}/{coarse_name}/point_cloud/iteration_10000/point_cloud.ply')
            print('Init canonical gaussians from coarse gaussian.')
        elif self.args.canonical_init == 'load':
            p = args.source_path.replace('ArtGS_data/', 'REArtGS2/outputs/')
            self.gaussians.load_ply(f'{p}/art_gs/point_cloud/iteration_30000/point_cloud.ply')

        else:
            print('Init canonical gaussians randomly.')

        self.init_deform()
        self.gaussians.training_setup(opt, planar= True)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.iter_start = torch.cuda.Event(enable_timing=True)
        self.iter_end = torch.cuda.Event(enable_timing=True)
        self.iteration = 1 if self.scene.loaded_iter is None else self.scene.loaded_iter

        self.viewpoint_stacks = [self.scene.getTrainCameras_start(), self.scene.getTrainCameras_end()]

        self.ema_loss_for_log = 0.0
        self.best_iteration = 15000
        self.best_joint_error = 1e10
        self.joint_metrics = []
        self.ema_loss_for_log = 0.0
        self.ema_single_view_for_log = 0.0
        self.ema_multi_view_geo_for_log = 0.0
        self.ema_multi_view_pho_for_log = 0.0
        self.depth_loss_for_log = 0.0


        self.best_psnr = 0.0
        #self.best_iteration = 0
        #self.progress_bar = tqdm.tqdm(range(self.iteration - 1, opt.iterations), desc="Training progress")
        self.progress_bar = tqdm.tqdm(range(self.iteration - 1, opt.iterations), desc="Training progress")
        self.smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)

        self.cd_loss_weight = args.cd_loss_weight
        self.metric_depth_loss_weight = args.metric_depth_loss_weight
        self.mono_depth_loss_weight = args.mono_depth_loss_weight

        self.depth_loss = DepthLoss()
        # vote
        self.voting = BoundaryPointVotingModule(
            neighbor_k=50,  # 20
            different_label_threshold=0.05,  # 0.25
            boundary_neighbor_radius=0.02,
            dbscan_eps=0.15,  # 0.08
            dbscan_min_samples=5,  # 5
            voting_radius=0.05,
            voting_k=16,
            boundary_radius=1
        ).to("cuda")
        self.vote_loss_fn = LocalConsistencyVotingLoss(loss_type='kl').to('cuda')
        self.use_voting = True

        # For UI
        self.visualization_mode = 'RGB'

        self.gui = args.gui  # enable gui
        self.W = args.W
        self.H = args.H
        self.cam = OrbitCamera(args.W, args.H, r=args.radius, fovy=args.fovy)

        self.mode = "render"
        self.seed = "random"
        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.training = False

        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()

    def __del__(self):
        if self.gui:
            dpg.destroy_context()

    def register_dpg(self):
        ### register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window
        # the rendered image, as the primary window
        with dpg.window(
                tag="_primary_window",
                width=self.W,
                height=self.H,
                pos=[0, 0],
                no_move=True,
                no_title_bar=True,
                no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
                label="Control",
                tag="_control_window",
                width=600,
                height=self.H,
                pos=[self.W, 0],
                no_move=True,
                no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_setattr(sender, app_data, user_data):
                setattr(self, user_data, app_data)

            # init stuff
            with dpg.collapsing_header(label="Initialize", default_open=True):

                # seed stuff
                def callback_set_seed(sender, app_data):
                    self.seed = app_data
                    self.seed_everything()

                dpg.add_input_text(
                    label="seed",
                    default_value=self.seed,
                    on_enter=True,
                    callback=callback_set_seed,
                )

                # input stuff
                def callback_select_input(sender, app_data):
                    # only one item
                    for k, v in app_data["selections"].items():
                        dpg.set_value("_log_input", k)
                        self.load_input(v)

                    self.need_update = True

                with dpg.file_dialog(
                        directory_selector=False,
                        show=False,
                        callback=callback_select_input,
                        file_count=1,
                        tag="file_dialog_tag",
                        width=700,
                        height=400,
                ):
                    dpg.add_file_extension("Images{.jpg,.jpeg,.png}")

                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="input",
                        callback=lambda: dpg.show_item("file_dialog_tag"),
                    )
                    dpg.add_text("", tag="_log_input")

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Visualization: ")

                    def callback_vismode(sender, app_data, user_data):
                        self.visualization_mode = user_data
                        if user_data == 'Node':
                            self.node_vis_fea = True if not hasattr(self, 'node_vis_fea') else not self.node_vis_fea
                            print("Visualize node features" if self.node_vis_fea else "Visualize node importance")
                            if self.node_vis_fea or True:
                                from motion import visualize_featuremap
                                if True:  # self.renderer.gaussians.motion_model.soft_edge:
                                    if hasattr(self.renderer.gaussians.motion_model, 'nodes_fea'):
                                        node_rgb = visualize_featuremap(
                                            self.renderer.gaussians.motion_model.nodes_fea.detach().cpu().numpy())
                                        self.node_rgb = torch.from_numpy(node_rgb).cuda()
                                    else:
                                        self.node_rgb = None
                                else:
                                    self.node_rgb = None
                            else:
                                node_imp = self.renderer.gaussians.motion_model.cal_node_importance(
                                    x=self.renderer.gaussians.get_xyz)
                                node_imp = (node_imp - node_imp.min()) / (node_imp.max() - node_imp.min())
                                node_rgb = torch.zeros([node_imp.shape[0], 3], dtype=torch.float32).cuda()
                                node_rgb[..., 0] = node_imp
                                node_rgb[..., -1] = 1 - node_imp
                                self.node_rgb = node_rgb

                    dpg.add_button(
                        label="RGB",
                        tag="_button_vis_rgb",
                        callback=callback_vismode,
                        user_data='RGB',
                    )
                    dpg.bind_item_theme("_button_vis_rgb", theme_button)

                    dpg.add_button(
                        label="UV_COOR",
                        tag="_button_vis_uv",
                        callback=callback_vismode,
                        user_data='UV_COOR',
                    )
                    dpg.bind_item_theme("_button_vis_uv", theme_button)
                    dpg.add_button(
                        label="MotionMask",
                        tag="_button_vis_motion_mask",
                        callback=callback_vismode,
                        user_data='MotionMask',
                    )
                    dpg.bind_item_theme("_button_vis_motion_mask", theme_button)

                    dpg.add_button(
                        label="Node",
                        tag="_button_vis_node",
                        callback=callback_vismode,
                        user_data='Node',
                    )
                    dpg.bind_item_theme("_button_vis_node", theme_button)

                    def callback_use_const_var(sender, app_data):
                        self.use_const_var = not self.use_const_var

                    dpg.add_button(
                        label="Const Var",
                        tag="_button_const_var",
                        callback=callback_use_const_var
                    )
                    dpg.bind_item_theme("_button_const_var", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("Scale Const: ")

                    def callback_vis_scale_const(sender):
                        self.vis_scale_const = 10 ** dpg.get_value(sender)
                        self.need_update = True

                    dpg.add_slider_float(
                        label="Log vis_scale_const (For debugging)",
                        default_value=-3,
                        max_value=-.5,
                        min_value=-5,
                        callback=callback_vis_scale_const,
                    )

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Temporal Speed: ")
                    self.video_speed = 1.

                    def callback_speed_control(sender):
                        self.video_speed = dpg.get_value(sender)
                        self.need_update = True

                    dpg.add_slider_float(
                        label="Play speed",
                        default_value=1.,
                        max_value=2.,
                        min_value=0.0,
                        callback=callback_speed_control,
                    )

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Save: ")

                    def callback_save(sender, app_data, user_data):
                        self.save_model(mode=user_data)

                    dpg.add_button(
                        label="model",
                        tag="_button_save_model",
                        callback=callback_save,
                        user_data='model',
                    )
                    dpg.bind_item_theme("_button_save_model", theme_button)

                    dpg.add_button(
                        label="geo",
                        tag="_button_save_mesh",
                        callback=callback_save,
                        user_data='geo',
                    )
                    dpg.bind_item_theme("_button_save_mesh", theme_button)

                    dpg.add_button(
                        label="geo+tex",
                        tag="_button_save_mesh_with_tex",
                        callback=callback_save,
                        user_data='geo+tex',
                    )
                    dpg.bind_item_theme("_button_save_mesh_with_tex", theme_button)

                    dpg.add_button(
                        label="pcl",
                        tag="_button_save_pcl",
                        callback=callback_save,
                        user_data='pcl',
                    )
                    dpg.bind_item_theme("_button_save_pcl", theme_button)

                    def call_back_save_train(sender, app_data, user_data):
                        self.render_all_train_data()

                    dpg.add_button(
                        label="save_train",
                        tag="_button_save_train",
                        callback=call_back_save_train,
                    )

            # training stuff
            with dpg.collapsing_header(label="Train", default_open=True):
                # lr and train button
                with dpg.group(horizontal=True):
                    dpg.add_text("Train: ")

                    def callback_train(sender, app_data):
                        if self.training:
                            self.training = False
                            dpg.configure_item("_button_train", label="start")
                        else:
                            # self.prepare_train()
                            self.training = True
                            dpg.configure_item("_button_train", label="stop")

                    dpg.add_button(
                        label="start", tag="_button_train", callback=callback_train
                    )
                    dpg.bind_item_theme("_button_train", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_psnr")
                    dpg.add_text("", tag="_log_train_log")

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    ("part", "render", "plane_depth", "bound", "rendered_normal", "depth_normal"),
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoV (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=np.rad2deg(self.cam.fovy),
                    callback=callback_set_fovy,
                )

        ### register camera handler

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

        dpg.create_viewport(
            title="REArtGS",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        ### register a larger font
        # get it from: https://github.com/lxgw/LxgwWenKai/releases/download/v1.300/LXGWWenKai-Regular.ttf
        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        # dpg.show_metrics()

        dpg.show_viewport()

    def render(self):

        assert self.gui
        # for param_group in self.gaussians.optimizer.param_groups:
        #     params = param_group["params"]
        #     for param in params:
        #         param.requires_grad = False

        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:
                self.train_step()
            self.test_step()
            dpg.render_dearpygui_frame()
        print("1")




    @torch.no_grad()
    def init_deform(self):
        if self.args.center_init == 'cgs':
            #p = args.source_path.replace('data/', 'outputs/')
            p = args.source_path.replace('ArtGS_data/', 'REArtGS2/outputs/')
            coarse_name = self.args.coarse_name
            center, scale = self.deform.deform.seg_model.init_from_file(
                f'{p}/{coarse_name}/point_cloud/iteration_10000/center_info.npy')
            print('Init center from coarse gaussian.')
        elif self.args.center_init == 'pcd':
            center, scale = self.deform.deform.seg_model.init_from_file(f'{self.args.source_path}/center_info.npy')
            print('Init center from pcd.')
        else:
            print('Init center randomly.')
        self.deform.load_weights(self.dataset.model_path, iteration=-1)
        self.deform.train_setting(self.opt)
       


    def train(self, iters=5000):
        for i in tqdm.trange(iters):
            self.train_step()

    def train_step(self):
        self.iter_start.record()

        if self.iteration == 15000:
            print("debug")

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.iteration % self.opt.oneupSHdegree_step == 0:
            self.gaussians.oneupSHdegree()

        state = randint(0, 1)
        id = randint(0, len(self.viewpoint_stacks[state]) - 1)
        viewpoint_cam = self.viewpoint_stacks[state][id]

        # Render
        random_bg = (
                                not self.dataset.white_background and self.opt.random_bg_color) and viewpoint_cam.gt_alpha_mask is not None
        bg = self.background if not random_bg else torch.rand_like(self.background).cuda()
        d_values = self.deform.deform.one_transform(self.gaussians, state, is_training=True)
        d_xyz, d_rot, masks = d_values['d_xyz'], d_values['d_rotation'], d_values['mask_p']
        render_pkg = planar_render(viewpoint_cam, self.gaussians, self.pipe, bg, d_xyz =d_xyz, d_rot = d_rot, app_model=self.app_model,
                            return_plane=self.iteration > self.opt.single_view_weight_from_iter,
                            return_depth_normal=self.iteration > self.opt.single_view_weight_from_iter)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
            "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
        gt_image_gray = viewpoint_cam.gt_image_gray.cuda()
        if random_bg:
            gt_image = gt_alpha_mask * gt_image + (1 - gt_alpha_mask) * bg[:, None, None]
        elif self.dataset.white_background and viewpoint_cam.gt_alpha_mask is not None:
            gt_image = gt_alpha_mask * gt_image + (1 - gt_alpha_mask) * self.background[:, None, None]

        normal_loss, geo_loss, ncc_loss = None, None, None
        loss = 0
        # point loss
        # if self.cd_loss_weight > 0 and self.iteration > self.args.cd_min_steps and self.iteration < self.args.cd_max_steps and 'p' not in self.dataset.joint_types:
        #     xt = self.gaussians.get_xyz.detach() + d_xyz
        #     cd, _ = chamfer_distance(xt[None], self.xyzs[state][None], single_directional=True)
        #     cd_loss = self.cd_loss_weight * cd
        #     loss = loss + cd_loss
        #     self.tb_writer.add_scalar('train/cd_loss', cd_loss.item(), self.iteration)

        # depth loss
        depth_loss = torch.tensor([0.])
        # if self.metric_depth_loss_weight > 0 and self.iteration > self.opt.single_view_weight_from_iter:
        #     depth = render_pkg['plane_depth']
        #     gt_depth = viewpoint_cam.depth.cuda()
        #     invalid_mask = (gt_depth < 0.1) & (gt_alpha_mask > 0.5)
        #     valid_mask = ~invalid_mask
        #     n_valid_pixel = valid_mask.sum()
        #     if n_valid_pixel > 100:
        #         depth_loss = (torch.log(1 + torch.abs(depth - gt_depth)) * valid_mask).sum() / n_valid_pixel
        #         loss = loss + depth_loss * self.metric_depth_loss_weight
        #
        mono_depth_loss = torch.tensor([0.])
        # if self.mono_depth_loss_weight > 0:
        #     depth = render_pkg_re['depth']
        #     mono_depth = viewpoint_cam.mono_depth.cuda()
        #     # mono_depth_loss = depth_rank_loss(depth, mono_depth, gt_alpha_mask)
        #     mono_depth_loss = self.depth_loss(depth, mono_depth[None], gt_alpha_mask)
        #     loss = loss + mono_depth_loss * self.mono_depth_loss_weight

        if self.iteration > 3000:
            loss = loss + self.deform.reg_loss

        ssim_loss = (1.0 - ssim(image, gt_image))
        if 'app_image' in render_pkg and ssim_loss < 0.5:
            app_image = render_pkg['app_image']
            Ll1 = l1_loss(app_image, gt_image)
        else:
            Ll1 = l1_loss(image, gt_image)
        image_loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * ssim_loss
        loss += image_loss.clone()

        # scale loss
        if visibility_filter.sum() > 0:
            scale = self.gaussians.get_scaling[visibility_filter]
            sorted_scale, _ = torch.sort(scale, dim=-1)
            min_scale_loss = sorted_scale[..., 0]
            loss += 100.0 * min_scale_loss.mean()

        # voting loss

        if self.iteration in [self.opt.densify_until_iter, self.opt.densify_until_iter + 5000,
                              self.opt.densify_until_iter + 10000]:
            # if self.iteration == 1:
            with torch.no_grad():
                self.boundary_mask, self.boundary_indices, self.voting_mask = self.voting(self.gaussians.get_xyz, masks)

            print(f"\nBoundary Detection Results:")
            print(f"Total points: {self.gaussians.get_xyz.shape}")
            print(f"Boundary points: {len(self.boundary_indices)}")

        if self.iteration > self.opt.densify_until_iter and self.use_voting:
            # 计算损失
            vote_loss = self.vote_loss_fn(masks, self.voting_mask, self.boundary_indices)
            # print(f"Loss: {vote_loss.item():.6f}")
            loss += 1e-5 * vote_loss

        # single-view loss
        if self.iteration > self.opt.single_view_weight_from_iter:
            weight = self.opt.single_view_weight
            normal = render_pkg["rendered_normal"]
            depth_normal = render_pkg["depth_normal"]

            image_weight = (1.0 - get_img_grad_weight(gt_image))
            image_weight = (image_weight).clamp(0, 1).detach() ** 5
            image_weight = erode(image_weight[None, None]).squeeze()

            normal_loss = weight * (image_weight * (((depth_normal - normal)).abs().sum(0))).mean()

            with torch.no_grad():
                c_render_pkg = planar_render(viewpoint_cam, self.gaussians, self.pipe, bg, d_xyz=None, d_rot=None,
                                           app_model=self.app_model)
                canonical_normal = c_render_pkg["rendered_normal"].detach()
                canonical_depth_normal = c_render_pkg["depth_normal"].detach()
            normal_gradient = normal - canonical_normal
            depth_normal_gradient = depth_normal - canonical_depth_normal
            normal_gradient_loss = weight * (
                        image_weight * (((depth_normal_gradient - normal_gradient)).abs().sum(0))).mean()
            loss += normal_gradient_loss
            loss += (normal_loss)

        # multi-view loss
        if self.iteration > self.opt.multi_view_weight_from_iter:
            nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else self.scene.getTrainCameras()[
                random.sample(viewpoint_cam.nearest_id, 1)[0]]
            use_virtul_cam = False
            if self.opt.use_virtul_cam and (np.random.random() < self.opt.virtul_cam_prob or nearest_cam is None):
                nearest_cam = gen_virtul_cam(viewpoint_cam, trans_noise=self.dataset.multi_view_max_dis,
                                             deg_noise=self.dataset.multi_view_max_angle)
                use_virtul_cam = True
            if nearest_cam is not None:
                patch_size = self.opt.multi_view_patch_size
                sample_num = self.opt.multi_view_sample_num
                pixel_noise_th = self.opt.multi_view_pixel_noise_th
                total_patch_size = (patch_size * 2 + 1) ** 2
                ncc_weight = self.opt.multi_view_ncc_weight
                geo_weight = self.opt.multi_view_geo_weight
                ## compute geometry consistency mask and loss
                H, W = render_pkg['plane_depth'].squeeze().shape
                ix, iy = torch.meshgrid(
                    torch.arange(W), torch.arange(H), indexing='xy')
                pixels = torch.stack([ix, iy], dim=-1).float().to(render_pkg['plane_depth'].device)

                nearest_render_pkg = planar_render(nearest_cam, self.gaussians, self.pipe, bg,
                                                 app_model=self.app_model,
                                                 return_plane=True, return_depth_normal=False)

                pts = self.gaussians.get_points_from_depth(viewpoint_cam, render_pkg['plane_depth'])
                pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3, :3] + nearest_cam.world_view_transform[
                                                                                      3, :3]
                map_z, d_mask = self.gaussians.get_points_depth_in_depth_map(nearest_cam,
                                                                             nearest_render_pkg['plane_depth'],
                                                                             pts_in_nearest_cam)

                pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:, 2:3])
                pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[..., None]
                R = torch.tensor(nearest_cam.R).float().cuda()
                T = torch.tensor(nearest_cam.T).float().cuda()
                pts_ = (pts_in_nearest_cam - T) @ R.transpose(-1, -2)
                pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,
                                         :3] + viewpoint_cam.world_view_transform[3, :3]
                pts_projections = torch.stack(
                    [pts_in_view_cam[:, 0] * viewpoint_cam.Fx / pts_in_view_cam[:, 2] + viewpoint_cam.Cx,
                     pts_in_view_cam[:, 1] * viewpoint_cam.Fy / pts_in_view_cam[:, 2] + viewpoint_cam.Cy], -1).float()
                pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
                d_mask = d_mask & (pixel_noise < pixel_noise_th)
                weights = (1.0 / torch.exp(pixel_noise)).detach()
                weights[~d_mask] = 0

                if d_mask.sum() > 0:
                    geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()
                    loss += geo_loss
                    if use_virtul_cam is False:
                        with torch.no_grad():
                            ## sample mask
                            d_mask = d_mask.reshape(-1)
                            valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                            if d_mask.sum() > sample_num:
                                index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace=False)
                                valid_indices = valid_indices[index]

                            weights = weights.reshape(-1)[valid_indices]
                            ## sample ref frame patch
                            pixels = pixels.reshape(-1, 2)[valid_indices]
                            offsets = patch_offsets(patch_size, pixels.device)
                            ori_pixels_patch = pixels.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float()

                            H, W = gt_image_gray.squeeze().shape
                            pixels_patch = ori_pixels_patch.clone()
                            pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                            pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
                            ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2),
                                                         align_corners=True)
                            ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

                            ref_to_neareast_r = nearest_cam.world_view_transform[:3, :3].transpose(-1,
                                                                                                   -2) @ viewpoint_cam.world_view_transform[
                                                                                                         :3, :3]
                            ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,
                                                                     :3] + nearest_cam.world_view_transform[3, :3]

                        ## compute Homography
                        ref_local_n = render_pkg["rendered_normal"].permute(1, 2, 0)
                        ref_local_n = ref_local_n.reshape(-1, 3)[valid_indices]

                        ref_local_d = render_pkg['rendered_distance'].squeeze()
                        # rays_d = viewpoint_cam.get_rays()
                        # rendered_normal2 = rendered_normal.reshape(-1,3)
                        # ref_local_d = render_pkg['plane_depth'].view(-1) * ((rendered_normal2 * rays_d.reshape(-1,3)).sum(-1).abs())
                        # ref_local_d = ref_local_d.reshape(H,W)

                        ref_local_d = ref_local_d.reshape(-1)[valid_indices]
                        H_ref_to_neareast = ref_to_neareast_r[None] - \
                                            torch.matmul(
                                                ref_to_neareast_t[None, :, None].expand(ref_local_d.shape[0], 3, 1),
                                                ref_local_n[:, :, None].expand(ref_local_d.shape[0], 3, 1).permute(0, 2,
                                                                                                                   1)) / \
                                            ref_local_d[..., None, None]
                        H_ref_to_neareast = torch.matmul(
                            nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3),
                            H_ref_to_neareast)
                        H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale)

                        ## compute neareast frame patch
                        grid = patch_warp(H_ref_to_neareast.reshape(-1, 3, 3), ori_pixels_patch)
                        grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
                        grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
                        _, nearest_image_gray = nearest_cam.get_image()
                        sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2),
                                                         align_corners=True)
                        sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)

                        ## compute loss
                        ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
                        mask = ncc_mask.reshape(-1)
                        ncc = ncc.reshape(-1) * weights
                        ncc = ncc[mask].squeeze()

                        if mask.sum() > 0:
                            ncc_loss = ncc_weight * ncc.mean()
                            loss += ncc_loss
                # if iteration % 200 == 0:

        loss.backward()
        self.iter_end.record()

        with torch.no_grad():
            # Progress bar
            self.ema_loss_for_log = 0.4 * image_loss.item() + 0.6 * self.ema_loss_for_log
            self.ema_single_view_for_log = 0.4 * normal_loss.item() if normal_loss is not None else 0.0 + 0.6 * self.ema_single_view_for_log
            self.ema_multi_view_geo_for_log = 0.4 * geo_loss.item() if geo_loss is not None else 0.0 + 0.6 * self.ema_multi_view_geo_for_log
            self.ema_multi_view_pho_for_log = 0.4 * ncc_loss.item() if ncc_loss is not None else 0.0 + 0.6 * self.ema_multi_view_pho_for_log
            self.depth_loss_for_log = 0.4 * depth_loss.item() if depth_loss is not None else 0.0 + self.depth_loss_for_log
            if self.iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{self.ema_loss_for_log:.{5}f}",
                    "Single": f"{self.ema_single_view_for_log:.{5}f}",
                    "Geo": f"{self.ema_multi_view_geo_for_log:.{5}f}",
                    "Pho": f"{self.ema_multi_view_pho_for_log:.{5}f}",
                    "depth": f"{self.depth_loss_for_log:.{5}f}",
                    "Points": f"{len(self.gaussians.get_xyz)}"
                }
                self.progress_bar.set_postfix(loss_dict)
                self.progress_bar.update(10)
            if self.iteration == self.opt.iterations:
                self.progress_bar.close()

            # Log and save
            training_report(self.tb_writer, self.iteration, Ll1, depth_loss, mono_depth_loss, loss,
                            self.iter_start.elapsed_time(self.iter_end), self.scene, self.joint_metrics)
            if self.iteration in self.saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(self.iteration))
                self.scene.save(self.iteration)
                self.deform.save_weights(self.args.model_path, self.iteration)
                if self.args.app_model:
                    self.app_model.save_weights(self.dataset.model_path, self.iteration)

            # Densification
            if self.iteration < self.opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                mask = (render_pkg["out_observe"] > 0) & visibility_filter
                self.gaussians.max_radii2D[mask] = torch.max(self.gaussians.max_radii2D[mask], radii[mask])
                viewspace_point_tensor_abs = render_pkg["viewspace_points_abs"]
                self.gaussians.planar_add_densification_stats(viewspace_point_tensor, viewspace_point_tensor_abs,
                                                            visibility_filter)

                if self.iteration > self.opt.densify_from_iter and self.iteration % self.opt.densification_interval == 0:
                    size_threshold = 20 if self.iteration > self.opt.opacity_reset_interval else None
                    self.gaussians.planar_densify_and_prune(0.0002, #self.opt.densify_grad_threshold,
                                                          0.0008, #self.opt.densify_abs_grad_threshold,
                                                          0.005, #self.opt.opacity_cull_threshold,
                                                          self.scene.cameras_extent,
                                                          size_threshold)

            # multi-view observe trim
            if self.opt.use_multi_view_trim and self.iteration % 1000 == 0 and self.iteration < self.opt.densify_until_iter:
                observe_the = 2
                observe_cnt = torch.zeros_like(self.gaussians.get_opacity)
                for view in self.scene.getTrainCameras():
                    render_pkg_tmp = planar_render(view, self.gaussians, self.pipe, bg, app_model=self.app_model,
                                                 return_plane=False,
                                                 return_depth_normal=False)
                    out_observe = render_pkg_tmp["out_observe"]
                    observe_cnt[out_observe > 0] += 1
                prune_mask = (observe_cnt < observe_the).squeeze()
                if prune_mask.sum() > 0:
                    self.gaussians.planar_prune_points(prune_mask)

            # reset_opacity
            if self.iteration < self.opt.densify_until_iter:
                if self.iteration % self.opt.opacity_reset_interval == 0 or (
                        self.dataset.white_background and self.iteration == self.opt.densify_from_iter):
                    self.gaussians.reset_opacity()



        self.gaussians.optimizer.step()
        self.gaussians.update_learning_rate(self.iteration)
        self.gaussians.optimizer.zero_grad(set_to_none=True)
        if self.args.app_model:
            self.app_model.optimizer.step()
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            self.app_model.optimizer.zero_grad(set_to_none=True)
        self.deform.optimizer.step()
        self.deform.optimizer.zero_grad()
        self.deform.update_learning_rate(self.iteration)

        self.deform.update(max(0, self.iteration))



        if self.gui:
            dpg.set_value(
                "_log_train_psnr",
                "Best PSNR = {} in Iteration {}".format(self.best_psnr, self.best_iteration)
            )
        else:
            print("Best PSNR = {} in Iteration {}".format(self.best_psnr, self.best_iteration))
        self.iteration += 1

        if self.gui:
            dpg.set_value(
                "_log_train_log",
                f"step = {self.iteration: 5d} loss = {loss.item():.4f}",
            )
        torch.cuda.empty_cache()

    @torch.no_grad()
    def test_step(self):

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        if not hasattr(self, 't0'):
            self.t0 = time.time()
            self.fps_of_fid = 10

        cur_cam = MiniCam(
            self.cam.pose,
            self.W,
            self.H,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
            fid=torch.remainder(torch.tensor((time.time() - self.t0) * self.fps_of_fid).float().cuda() / len(
                self.scene.getTrainCameras()), 1.)
        )
        fid = cur_cam.fid

        d_xyz, d_rot = self.deform.deform.interpolate_single_state(self.gaussians, fid)
        out = planar_render(cur_cam, self.gaussians, self.pipe, self.background, d_xyz=d_xyz, d_rot=d_rot,
                            return_plane= False, return_depth_normal= False)

        if self.mode == 'part':
            mask = self.deform.step(self.gaussians, is_training=False)[0]['mask']
            pallete = np.array(sns.color_palette("hls", mask.max() + 1))
            if self.report_mask:
                print(pallete)
                self.report_mask = False
            buffer_image = planar_render(viewpoint_camera=cur_cam, pc=self.gaussians, pipe=self.pipe, bg_color=self.background,
                                    d_xyz=d_xyz, d_rot=d_rot, mask=mask, return_plane= False, return_depth_normal= False)['render']
        elif self.mode == 'plane_depth':
            buffer_image = \
            render(viewpoint_camera=cur_cam, pc=self.gaussians, pipe=self.pipe, bg_color=self.background, d_xyz=d_xyz, d_rot=d_rot)['depth']
            # buffer_image = \
            # planar_render(viewpoint_camera=cur_cam, pc=self.gaussians, pipe=self.pipe, bg_color=self.background,
            #               d_xyz=d_xyz, d_rot=d_rot, return_plane=True, return_depth_normal=False)['plane_depth']
        elif self.mode == 'bound':

            mask = self.boundary_mask * 1
            buffer_image = \
            planar_render(viewpoint_camera=cur_cam, pc=self.gaussians, pipe=self.pipe, bg_color=self.background,
                      d_xyz=d_xyz, d_rot=d_rot, mask=mask, return_plane= False, return_depth_normal= False)['render']
        elif self.mode in ["rendered_normal", "depth_normal"]:
            buffer_image = \
            planar_render(viewpoint_camera=cur_cam, pc=self.gaussians, pipe=self.pipe, bg_color=self.background,
                          d_xyz=d_xyz, d_rot=d_rot)[self.mode]


        else:
            buffer_image = out[self.mode]  # [3, H, W]

        if self.mode in ['plane_depth']:
            buffer_image = buffer_image / (buffer_image.max() + 1e-5)
            buffer_image = buffer_image.repeat(3, 1, 1)
        try:
            buffer_image = torch.nn.functional.interpolate(
                buffer_image.unsqueeze(0),
                size=(self.H, self.W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            self.buffer_image = (
                buffer_image.permute(1, 2, 0)
                .contiguous()
                .clamp(0, 1)
                .contiguous()
                .detach()
                .cpu()
                .numpy()
            )

            self.need_update = True

            ender.record()
            #torch.cuda.synchronize()
            t = starter.elapsed_time(ender)

            if self.gui:
                dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000 / t)} FPS FID: {fid.item()})")
                dpg.set_value(
                    "_texture", self.buffer_image
                )  # buffer must be contiguous, else seg fault!
        except:
            pass

    def visualize(self, image, gt_image, gt_depth, depth):
        torchvision.utils.save_image(image.detach(), "img.png")
        torchvision.utils.save_image(gt_image, "img_gt.png")
        torchvision.utils.save_image(vis_depth(gt_depth), "gt.png")
        torchvision.utils.save_image(vis_depth(depth.detach()), "pred.png")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--gui', action='store_false', help="start a GUI")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--elevation', type=float, default=0, help="default GUI camera elevation")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")

    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--app_model", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    args.source_path = f"{args.source_path}/{args.dataset}/{args.subset}/{args.scene_name}"
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    seed_everything(args.seed)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # args.joint_types = json.load(open(f'./arguments/joint_types_{args.center_init}.json', 'r'))[args.dataset][args.subset][args.scene_name]
    # args.num_slots = len(args.joint_types.split(','))
    args.use_art_type_prior = False
    args.num_slots = json.load(open(f'./arguments/num_slots.json', 'r'))[args.dataset][args.subset][args.scene_name]
    trainer = Trainer(args=args, dataset=lp.extract(args), opt=op.extract(args), pipe=pp.extract(args),
                      saving_iterations=args.save_iterations)
    #trainer.train(args.iterations)

    if args.gui:
        trainer.render()
    else:
        trainer.train(args.iterations)
    print("\nTraining complete.")
