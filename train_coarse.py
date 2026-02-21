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

import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from scene.dataset_readers import fetchPly
from utils.general_utils import safe_state, get_linear_noise_func
import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from pytorch_lightning import seed_everything
from utils.metrics import *
from utils.log_utils import prepare_output_and_logger


class Trainer:
    def __init__(self, args, dataset, opt, pipe):
        self.dataset = dataset
        self.args = args
        self.opt = opt
        self.pipe = pipe

        self.tb_writer = prepare_output_and_logger(dataset)
        self.gaussians = [GaussianModel(dataset.sh_degree, fea_dim=0), 
                          GaussianModel(dataset.sh_degree, fea_dim=0)]

        self.scene = Scene(dataset, self.gaussians[0], self.gaussians[1], 
                           load_iteration=None, init_with_random_pcd=True)
        
        if args.init_from_pcd:
            print('Init Gaussians with pcd from depth.')
            for i, state in enumerate(['start', 'end']):
                self.gaussians[i].create_from_pcd(fetchPly(f'{args.source_path}/point_cloud_{state}.ply'))
        
        for i in [0, 1]:
            self.gaussians[i].training_setup(opt)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.iter_start = torch.cuda.Event(enable_timing=True)
        self.iter_end = torch.cuda.Event(enable_timing=True)
        self.iteration = 1

        self.viewpoint_stacks = [self.scene.getTrainCameras_start(), self.scene.getTrainCameras_end()]
        self.ema_loss_for_log = 0.0
        self.progress_bar = tqdm.tqdm(range(self.iteration-1, opt.iterations), desc="Training progress")
        self.smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)

        self.reg_weight = self.args.opacity_reg_weight

    # no gui mode
    def train(self, iters=5000):
        for i in tqdm.trange(iters):
            self.train_step()
    
    def train_step(self):
        self.iter_start.record()

        for state in (0, 1):
            # Every 1000 its we increase the levels of SH up to a maximum degree
            if self.iteration % self.opt.oneupSHdegree_step == 0:
                self.gaussians[state].oneupSHdegree()
            id = randint(0, len(self.viewpoint_stacks[state]) - 1)
            viewpoint_cam = self.viewpoint_stacks[state][id]
            
            # Render
            random_bg = (not self.dataset.white_background and self.opt.random_bg_color) and viewpoint_cam.gt_alpha_mask is not None
            bg = self.background if not random_bg else torch.rand_like(self.background).cuda()
            d_xyz, d_rot = None, None
            render_pkg_re = render(viewpoint_cam, self.gaussians[state], self.pipe, bg, d_xyz=d_xyz, d_rot=d_rot)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re["viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
            if random_bg:
                gt_image = gt_alpha_mask * gt_image + (1 - gt_alpha_mask) * bg[:, None, None]
            elif self.dataset.white_background and viewpoint_cam.gt_alpha_mask is not None:
                gt_image = gt_alpha_mask * gt_image + (1 - gt_alpha_mask) * self.background[:, None, None]

            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            opacity = self.gaussians[state].get_opacity
            reg_loss = F.binary_cross_entropy(opacity, (opacity.detach() > 0.5) * 1.0)
            loss = loss + reg_loss * self.reg_weight

            loss.backward()

            with torch.no_grad():
                 # Keep track of max radii in image-space for pruning
                if self.gaussians[state].max_radii2D.shape[0] == 0:
                    self.gaussians[state].max_radii2D = torch.zeros_like(radii)
                self.gaussians[state].max_radii2D[visibility_filter] = torch.max(self.gaussians[state].max_radii2D[visibility_filter], radii[visibility_filter])
                # Densification
                if self.iteration < self.opt.densify_until_iter:
                    self.gaussians[state].add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if self.iteration > self.opt.densify_from_iter and self.iteration % self.opt.densification_interval == 0:
                        size_threshold = 20 if self.iteration > self.opt.opacity_reset_interval else None
                        self.gaussians[state].densify_and_prune(self.opt.densify_grad_threshold, 0.005, self.scene.cameras_extent, size_threshold)

                    if self.iteration % self.opt.opacity_reset_interval == 0 or (
                            self.dataset.white_background and self.iteration == self.opt.densify_from_iter):
                        self.gaussians[state].reset_opacity()

                # Optimizer step
                if self.iteration < self.opt.iterations:
                    self.gaussians[state].optimizer.step()
                    self.gaussians[state].update_learning_rate(self.iteration)
                    self.gaussians[state].optimizer.zero_grad(set_to_none=True)
                    
        self.iter_end.record()

        with torch.no_grad():
            # Progress bar
            self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
            if self.iteration % 10 == 0:
                self.progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{7}f}"})
                self.progress_bar.update(10)
            if self.iteration == self.opt.iterations:
                self.progress_bar.close()
                large_motion_state = self.scene.save_2gs(self.iteration, self.args.num_slots, self.args.vis_cano, self.args.vis_center)
                file = json.load(open('arguments/larger_motion_state.json', 'r'))
                file[self.args.dataset][self.args.subset][self.args.scene_name] = large_motion_state
                json.dump(file, open('arguments/larger_motion_state.json', 'w'), indent=4)
        self.iteration += 1


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--vis_cano", action="store_true")
    parser.add_argument("--vis_center", action="store_true")
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args(sys.argv[1:])
    args.source_path = f"{args.source_path}/{args.dataset}/{args.subset}/{args.scene_name}"

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    seed_everything(args.seed)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    args.num_slots = json.load(open('./arguments/num_slots.json', 'r'))[args.dataset][args.subset][args.scene_name]
    trainer = Trainer(args=args, dataset=lp.extract(args), opt=op.extract(args), pipe=pp.extract(args))
    trainer.train(args.iterations)
    print("\nTraining complete.")
