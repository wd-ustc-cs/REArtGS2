
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scene.gaussian_model import GaussianModel
from utils.dual_quaternion import *
from scene.module import gumbel_softmax, ProgressiveBandHashGrid
from utils.pose_utils import rigid_transform

class CenterBasedSeg(nn.Module):
    def __init__(self, num_slots, slot_size, scale_factor=1.0, shift_weight=0.5):
        super().__init__()
        self.num_slots = num_slots

        self.grid = ProgressiveBandHashGrid(3, start_level=6, n_levels=12, start_step=0, update_steps=500)
        dim = num_slots * 4 + self.grid.n_output_dims + 3
        self.mlp = nn.Sequential(
                nn.Linear(dim, slot_size),
                nn.ReLU(),
                nn.Linear(slot_size, num_slots * 2),
            )
        self.center = nn.Parameter(torch.randn(num_slots, 3) * 0.01)
        self.logscale = nn.Parameter(torch.randn(num_slots, 3) * 0.01)
        self.rot = nn.Parameter(torch.Tensor([[1, 0, 0, 0]]).repeat(self.num_slots, 1))

        self.scale_factor = scale_factor
        self.shift_weight = shift_weight
        
    def forward(self, x, tau, is_training=False):
        '''
            x: position of canonical gaussians [N, 3]
        '''
        rel_pos = self.cal_relative_pos(x) # [N, K, 3]
        dist = (rel_pos ** 2).sum(-1) # [N, K]

        x_rel = torch.cat([rel_pos, torch.norm(rel_pos, p=2, dim=-1, keepdim=True)], dim=-1) # [N, K, 4]
        info = torch.cat([x_rel.reshape(x.shape[0], -1), self.grid(x), x], -1)
        delta = self.mlp(info) # [N, K * 2]
        logscale, shift = torch.split(delta, delta.shape[-1] // 2, dim=-1) # [N, K]

        dist = dist * (self.shift_weight * logscale).exp()
        logits = -dist + shift * self.shift_weight

        slots = None
        hard = (tau - 0.1) < 1e-3
        mask, _ = gumbel_softmax(logits, tau=tau / (self.num_slots - 1), hard=hard, dim=1, is_training=is_training)
        return slots, mask

    def init_from_file(self, path):
        center_info = torch.from_numpy(np.load(path)).float().to(self.rot.device) # [K, 4], center and radius


        self.center = nn.Parameter(center_info[:, :3])

        self.logscale = nn.Parameter(torch.log(center_info[:, 3:4].repeat(1, 3)))
        return center_info[:, :3], center_info[:, 3:4]

    def cal_relative_pos(self, x):
        center = self.center[None]
        rot = self.get_rot[None]
        scale = self.get_scale[None] * self.scale_factor
        return quaternion_apply(rot, (x[:, None] - center)) / scale # [N, K, 3]
    
    @property
    def get_scale(self):
        return torch.exp(self.logscale)
    
    @property
    def get_rot(self):
        return F.normalize(self.rot, p=2, dim=-1)


class ArtGS(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.slot_size = args.slot_size
        self.joint_types = args.joint_types.split(',')
        self.use_art_type_prior = args.use_art_type_prior
        if self.use_art_type_prior:
            self.num_slots = len(self.joint_types)
        else:
            self.num_slots = args.num_slots
            self.joint_types = ['s'] + ['r' for _ in range(self.num_slots - 1)]

        joints = torch.zeros(self.num_slots - 1, 7) + torch.randn(self.num_slots - 1, 7) * 1e-5
        joints[:, 0] = 1
        self.joints = nn.Parameter(joints)
        #trans = torch.zeros(self.num_slots, 3)+ torch.randn(self.num_slots, 3) * 1e-5
        #self.trans = nn.Parameter(trans)
        self.register_buffer('Ts', torch.eye(4).float())
        self.register_buffer('qr_s', torch.Tensor([1, 0, 0, 0]))
        self.register_buffer('qd_s', torch.Tensor([0, 0, 0, 0]))
        self.register_buffer('tran_s', torch.Tensor([0, 0, 0]))
        self.register_buffer('o_s', torch.Tensor([0, 0, 0]))
        self.seg_model = CenterBasedSeg(self.num_slots, self.slot_size, scale_factor=args.scale_factor, shift_weight=args.shift_weight)
        self.revolute_constraint = args.revolute_constraint
        self.reg_loss = 0.
        self.tau = 1.0
        self.tau_decay_steps = args.tau_decay_steps
        self.noise = torch.ones(self.num_slots).cuda()
    @torch.no_grad()
    def cal_art_type(self):
        qr, qd = self.get_slot_deform()
        axis_dir, theta = quaternion_to_axis_angle(qr[1:])
        theta = theta.rad2deg()
        self.joint_types = ['s']
        self.joint_types += ['r' if t.item() > 10 else 'p' for t in theta]
        self.use_art_type_prior = True
        print(self.joint_types)
        return ','.join(self.joint_types)
    
    def slotdq_to_gsdq(self, slot_qr, slot_qd, mask):
        # slot_qr: [K, 4], slot_qd: [K, 4], mask: [N, K]
        qr = torch.einsum('nk, kl->nl', mask, slot_qr)   # [N, 4]
        qd = torch.einsum('nk, kl->nl', mask, slot_qd)   # [N, 4]
        return normalize_dualquaternion(qr, qd)
    
    def get_slot_deform(self):
        qrs = []
        qds = []
        for i, joint_type in enumerate(self.joint_types):
            if i == 0:
                assert joint_type == 's'
                qr, qd = self.qr_s, self.qd_s
            else:
                joint = self.joints[i - 1]
                qr = F.normalize(joint[:4], p=2, dim=-1)
                t0 = torch.cat([torch.zeros(1).to(qr.device), joint[4:7]])
                if self.use_art_type_prior:
                    if joint_type == 'p':
                        qr = self.qr_s
                        qd = 0.5 * quaternion_mul(t0, qr)
                    elif joint_type == 'r':
                        if self.revolute_constraint:
                            qd = 0.5 * (quaternion_mul(t0, qr) - quaternion_mul(qr, t0)) # better for multi-part real-world objects, but sensitive to initialization
                        else:
                            #t0 = self.qd_s # t0是关节轴pivot
                            qd = 0.5 * quaternion_mul(t0, qr)
                else:
                    qd = 0.5 * quaternion_mul(t0, qr)
            qrs.append(qr)
            qds.append(qd)
        qrs, qds = torch.stack(qrs), torch.stack(qds)
        return qrs, qds

    def get_slot_rigid_deform(self):
        qrs = []
        trans = []
        axes_o = []
        for i, joint_type in enumerate(self.joint_types):
            if i == 0:
                assert joint_type == 's'
                qr, tran, axis_o = self.qr_s, self.tran_s, self.o_s
            else:
                joint = self.joints[i - 1]
                qr = F.normalize(joint[:4], p=2, dim=-1)
                axis_o = joint[4:7]
                tran = self.trans[i].to(qr.device)
                if self.use_art_type_prior:
                    if joint_type == 'p':
                        qr = self.qr_s
                        axis_o  = self.o_s
                    elif joint_type == 'r':
                        tran = self.tran_s

            qrs.append(qr)
            trans.append(tran)
            axes_o.append(axis_o)
        qrs, trans, axes_o = torch.stack(qrs), torch.stack(trans), torch.stack(axes_o)
        return qrs, trans, axes_o


    def deform_pts(self, xc, mask, slot_qr, slot_qd, state):
        if state < 0.5:
            slot_qr, slot_qd = dual_quaternion_inverse((slot_qr, slot_qd))
        gs_qr, gs_qd = self.slotdq_to_gsdq(slot_qr, slot_qd, mask)
        xt = dual_quaternion_apply((gs_qr, gs_qd), xc)
        return xt, gs_qr
    
    def trainable_parameters(self):
        params = [
            {'params': self.joints, 'name': 'mlp'},

            {'params': list(self.seg_model.parameters()), 'name': 'slot'},
            ]
        return params
    
    def get_mask(self, x, is_training=False):
        tau = self.tau if is_training else 0.1
        slots, mask = self.seg_model(x, tau, is_training)
        self.slots = slots
        return mask
    
    @torch.no_grad()
    def get_joint_param(self, joint_type_list):
        qrs, qds = self.get_slot_deform()
        qrs, qds = qrs[1:], qds[1:]
        joint_info_list = []
        for i, joint_type in enumerate(joint_type_list):
            qr, qd = qrs[i], qds[i]
            qr, t = dual_quaternion_to_quaternion_translation((qr, qd))
            R = quaternion_to_matrix(qr).cpu().numpy()
            t = t.cpu().numpy()
            
            if joint_type == 'r':
                axis_dir, theta = quaternion_to_axis_angle(qr)
                axis_dir, theta = axis_dir.cpu().numpy(), theta.cpu().numpy()
                theta = 2 * theta
                axis_position = np.matmul(np.linalg.inv(np.eye(3) - R), t.reshape(3, 1)).reshape(-1)
                axis_position += axis_dir * np.dot(axis_dir, -axis_position)
                R = R @ R
                t = R @ t + t
                joint_info = {'type': joint_type,
                            'axis_position': axis_position,
                            'axis_direction': axis_dir,
                            'theta': np.rad2deg(theta),
                            'rotation': R, 'translation': t}
            elif joint_type == 'p':
                t = t * 2
                theta = np.linalg.norm(t)
                axis_dir = t / theta
                joint_info = {'type': joint_type,
                            'axis_position': np.zeros(3), 
                            'axis_direction': axis_dir, 
                            'theta': theta,
                            'rotation': R, 'translation': t}
            joint_info_list.append(joint_info)
        return joint_info_list
    
    def one_transform(self, gaussians:GaussianModel, state, is_training):
        xc = gaussians.get_xyz.detach()
        N = xc.shape[0]
        mask = self.get_mask(xc, is_training) # [N, K]
        qr, qd = self.get_slot_deform()
        #qrs, trans, axes_o = self.get_slot_rigid_deform()
        #xt, rot = rigid_transform(xc, mask, qrs, trans, axes_o, state)
        xt, rot = self.deform_pts(xc, mask, qr, qd, state)

        # regularization loss for center
        opacity = gaussians.get_opacity.detach()
        m = mask * opacity
        m = m / (m.sum(0, keepdim=True) + 1e-5)
        c = torch.einsum('nk,nj->kj', m, xc)
        self.reg_loss = F.mse_loss(self.seg_model.center, c) * 0.1

        d_xyz = xt - xc
        d_rotation = rot.detach()

        return {
            'd_xyz': d_xyz,
            'd_rotation': d_rotation,
            'xt': xt,
            'mask': mask.argmax(-1),
            'mask_p': mask
        }

    def interpolate_single_state(self, gaussians: GaussianModel, t):
        xc = gaussians._xyz.detach()
        mask = self.get_mask(xc, False)  # [N, K]
        qr1, qd1 = self.get_slot_deform()
        qr0, qd0 = dual_quaternion_inverse((qr1, qd1))

        slot_qr = (1 - t) * qr0 + t * qr1
        slot_qd = (1 - t) * qd0 + t * qd1
        gs_qr, gs_qd = self.slotdq_to_gsdq(slot_qr, slot_qd, mask)
        xt = dual_quaternion_apply((gs_qr, gs_qd), xc)
        dx = xt - xc
        dr = gs_qr
        return dx, dr

    def mask_interpolate_dual_state(self, gaussians: GaussianModel, time):
        t1 = time
        t2 = 1-time
        xc = gaussians._xyz.detach()
        mask = self.get_mask(xc, False)  # [N, K]
        qr1, qd1 = self.get_slot_deform()
        qr0, qd0 = dual_quaternion_inverse((qr1, qd1))

        dx_list = []
        dr_list = []
        for t in [t1, t2]:
            slot_qr = (1 - t) * qr0 + t * qr1
            slot_qd = (1 - t) * qd0 + t * qd1
            gs_qr, gs_qd = self.slotdq_to_gsdq(slot_qr, slot_qd, mask)
            xt = dual_quaternion_apply((gs_qr, gs_qd), xc)
            dx_list.append(xt - xc)
            dr_list.append(gs_qr)

        return dx_list, dr_list, mask.argmax(-1)


    # def interpolate_single_state(self, gaussians: GaussianModel, t):
    #     xc = gaussians._xyz.detach()
    #     mask = self.get_mask(xc, False)  # [N, K]
    #     qrs, trans, axes_o = self.get_slot_rigid_deform()
    #     xt, dr = rigid_transform(xc, mask, qrs, trans, axes_o, t)
    #
    #     dx = xt - xc
    #     return dx, dr


    def forward(self, gaussians: GaussianModel, is_training=False):
        xc = gaussians._xyz.detach()
        N = xc.shape[0]
        d_values_list = []
        mask = self.get_mask(xc, is_training) # [N, K]

        #mask = (mask * self.noise).cuda()
        qr, qd = self.get_slot_deform()
        #qrs, trans, axes_o = self.get_slot_rigid_deform()
        for state in [0, 1]:
            xt, rot = self.deform_pts(xc, mask, qr, qd, state)
            #xt, rot = rigid_transform(xc, mask, qrs, trans, axes_o, state)
            d_xyz = xt - xc
            d_rotation = rot.detach()
            d_values = {
                'd_xyz': d_xyz,
                'd_rotation': d_rotation,
                'xt': xt,
                'mask': mask.argmax(-1),
            }
            d_values_list.append(d_values)

        return d_values_list
    
    def interpolate(self, gaussians: GaussianModel, time_list):
        xc = gaussians._xyz.detach()
        mask = self.get_mask(xc, False) # [N, K]
        qr1, qd1 = self.get_slot_deform()
        qr0, qd0 = dual_quaternion_inverse((qr1, qd1))

        dx_list = []
        dr_list = []
        for t in time_list:
            slot_qr = (1 - t) * qr0 + t * qr1
            slot_qd = (1 - t) * qd0 + t * qd1
            gs_qr, gs_qd = self.slotdq_to_gsdq(slot_qr, slot_qd, mask)
            xt = dual_quaternion_apply((gs_qr, gs_qd), xc)
            dx_list.append(xt - xc)
            dr_list.append(gs_qr)
        return dx_list, dr_list

    def mask_interpolate(self, gaussians: GaussianModel, time_list):
        xc = gaussians._xyz.detach()
        mask = self.get_mask(xc, False) # [N, K]
        qr1, qd1 = self.get_slot_deform()
        qr0, qd0 = dual_quaternion_inverse((qr1, qd1))

        dx_list = []
        dr_list = []
        for t in time_list:
            slot_qr = (1 - t) * qr0 + t * qr1
            slot_qd = (1 - t) * qd0 + t * qd1
            gs_qr, gs_qd = self.slotdq_to_gsdq(slot_qr, slot_qd, mask)
            xt = dual_quaternion_apply((gs_qr, gs_qd), xc)
            dx_list.append(xt - xc)
            dr_list.append(gs_qr)
        return dx_list, dr_list, mask.argmax(-1)


    def update(self, iteration, *args, **kwargs):
        self.tau = self.cosine_anneal(iteration, self.tau_decay_steps, 0, 1.0, 0.1)
        self.seg_model.grid.update_step(global_step=iteration)

    def cosine_anneal(self, step, final_step, start_step=0, start_value=1.0, final_value=0.1):
        if start_value <= final_value or start_step >= final_step:
            return final_value
        
        if step < start_step:
            value = start_value
        elif step >= final_step:
            value = final_value
        else:
            a = 0.5 * (start_value - final_value)
            b = 0.5 * (start_value + final_value)
            progress = (step - start_step) / (final_step - start_step)
            value = a * math.cos(math.pi * progress) + b
        return value

