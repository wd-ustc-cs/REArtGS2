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
from scene import Scene
import uuid
from piq import LPIPS
lpips = LPIPS()
from argparse import Namespace

from torch.utils.tensorboard import SummaryWriter


def prepare_output_and_logger(args, use_tensorboard=True):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    if use_tensorboard:
        tb_writer = SummaryWriter(args.model_path)
    else:
        tb_writer = None
    return tb_writer


def training_report(tb_writer, iteration, Ll1, depth_loss, mono_depth_loss, loss, elapsed, scene: Scene, joint_metrics):
    if tb_writer:
        tb_writer.add_scalar('train/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train/depth_loss', depth_loss.item(), iteration)
        tb_writer.add_scalar('train/mono_depth_loss', mono_depth_loss.item(), iteration)
        tb_writer.add_scalar('train/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        total_error = 0
        for i, m in enumerate(joint_metrics):
            tb_writer.add_scalar(f'Angle_{i}', m[0], iteration)
            tb_writer.add_scalar(f'Distance_{i}', m[1], iteration)
            tb_writer.add_scalar(f'ThetaDiff_{i}', m[2], iteration)
            total_error += sum(m)
        tb_writer.add_scalar(f'Total_Error', total_error, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        if iteration % 4000 == 0:
            tb_writer.add_histogram("opacity_hist", scene.gaussians.get_opacity, iteration)