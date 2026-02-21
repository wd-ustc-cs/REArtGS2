import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn


def linear(in_features, out_features, bias=True, weight_init='xavier', gain=1., nonlinearity='relu'):
    
    m = nn.Linear(in_features, out_features, bias)
    
    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)
    else:
        nn.init.xavier_uniform_(m.weight, gain)
    
    if bias:
        nn.init.zeros_(m.bias)
    
    return m


def gru_cell(input_size, hidden_size, bias=True):
    
    m = nn.GRUCell(input_size, hidden_size, bias)
    
    nn.init.xavier_uniform_(m.weight_ih)
    nn.init.orthogonal_(m.weight_hh)
    
    if bias:
        nn.init.zeros_(m.bias_ih)
        nn.init.zeros_(m.bias_hh)
    
    return m


class FeatMLP(nn.Module):
    def __init__(self, d_in, d_out, d_hid, num_layers):
        super().__init__()
        self.net = nn.Sequential(*[nn.Sequential(
            linear(d_in if i == 0 else d_hid, d_out if i == num_layers - 1 else d_hid, weight_init='kaiming'),
            nn.ReLU()
        ) for i in range(num_layers)])


    def forward(self, x):
        return self.net(x)
    

def gumbel_softmax(logits, tau=1., hard=True, dim=-1, is_training=False):
    # modified from torch.nn.functional.gumbel_softmax
    if is_training:
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        logits = (logits + gumbels)
    logits = logits / tau
    y_soft = logits.softmax(dim)
    index = y_soft.max(dim, keepdim=True)[1]
    if hard:
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        # Straight through estimator.
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft
    return y, index.squeeze(dim)


class SlotAttention(nn.Module):
    def __init__(
        self,
        feature_size,
        slot_size,
        num_slots, 
        gumbel=False,
        mlp_ratio=2,
    ):
        super().__init__()
        self.slot_size = slot_size 
        self.epsilon = 1.0

        self.norm_feature = nn.LayerNorm(feature_size)
        self.norm_mlp = nn.LayerNorm(slot_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        
        self.project_q = linear(slot_size, slot_size, bias=False)
        self.project_k = linear(feature_size, slot_size, bias=False)
        self.project_v = linear(feature_size, slot_size, bias=False)
        self.scale = slot_size ** -0.5
        self.gru = gru_cell(slot_size, slot_size)

        self.mlp = nn.Sequential(
            linear(slot_size, slot_size * mlp_ratio, weight_init='kaiming'),
            nn.ReLU(),
            linear(slot_size * mlp_ratio, slot_size),
        )

        self.slots_init = nn.Parameter(torch.zeros(num_slots, slot_size))
        nn.init.normal_(self.slots_init, std=0.02)
        
        self.gumbel = gumbel
        self.sigma = 1

    def forward(self, features, slots=None, shift=None, num_iter=2, tau=1.0, is_training=False):
        # features: [batch_size, num_feature, inputs_size]
        features = self.norm_feature(features)  
        k = self.project_k(features)  # Shape: [B, num_features, slot_size]
        v = self.project_v(features) 
        if slots == None:
            if is_training and self.sigma > 0:
                mu = self.slots_init
                z = torch.randn_like(mu).type_as(mu)
                slots = mu + z * self.sigma * mu.detach()
            else:
                slots = self.slots_init
        # Multiple rounds of attention.
        for i in range(num_iter - 1):
            slots, attn = self.iter(slots, k, v, 1.0, shift=shift)
            slots = slots.detach() + self.slots_init - self.slots_init.detach()
        # slots, attn = self.iter(slots, k, v, 1.0)
        return slots, self.get_attn(slots, k, tau, self.gumbel, shift=shift, is_training=is_training)

    def iter(self, slots, k, v, tau=1.0, gumbel=False, shift=None, is_training=False):
        K, D = slots.shape
        slots_prev = slots
        slots = self.norm_slots(slots)
        q = self.project_q(slots)

        # Attention
        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale # [K, Nf]
        if shift is not None:
            logits = logits + shift

        # attn, _ = gumbel_softmax(logits, tau=tau, hard=(tau - 0.1) < 1e-3 and gumbel, dim=0, is_training=is_training)
        if gumbel:
            attn, _ = gumbel_softmax(logits, tau=tau, hard=(tau - 0.1) < 1e-3, dim=0, is_training=is_training)
        else:
            attn = F.softmax(logits, dim=0)

        # # Weighted mean
        attn_sum = torch.sum(attn, dim=-1, keepdim=True) + self.epsilon
        attn_wm = attn / attn_sum 
        updates = torch.einsum('ij,jd->id', attn_wm, v)            

        # Update slots
        slots = self.gru(
            updates.reshape(-1, D),
            slots_prev.reshape(-1, D)
        )
        # slots = slots_prev + updates
        slots = slots.reshape(-1, D)
        slots = slots + self.mlp(self.norm_mlp(slots))
        return slots, attn
    
    def get_attn(self, slots, k, tau=1.0, gumbel=False, shift=None, is_training=False):
        slots = self.norm_slots(slots)
        q = self.project_q(slots)

        # Attention
        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale # [B, K, Nf]
        if shift is not None:
            logits = logits + shift
        if gumbel:
            attn, _ = gumbel_softmax(logits, tau=tau, hard=(tau - 0.1) < 1e-3, dim=0, is_training=is_training)
        else:
            attn = F.softmax(logits, dim=0)
        return attn.transpose(0, 1)


def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


class ProgressiveBandHashGrid(nn.Module):
    def __init__(self, in_channels, start_level=6, n_levels=12, start_step=1000, update_steps=1000, dtype=torch.float32):
        super().__init__()

        encoding_config = {
            "otype": "Grid",
            "type": "Hash",
            "n_levels": n_levels,  # 16 for complex motions
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": 16,
            "per_level_scale": 2.0,
            "interpolation": "Linear",
            "start_level": start_level,
            "start_step": start_step,
            "update_steps": update_steps,
        }

        self.n_input_dims = in_channels
        with torch.cuda.device(get_rank()):
            self.encoding = tcnn.Encoding(in_channels, encoding_config, dtype=dtype)
        self.n_output_dims = self.encoding.n_output_dims
        self.n_level = encoding_config["n_levels"]
        self.n_features_per_level = encoding_config["n_features_per_level"]
        self.start_level, self.start_step, self.update_steps = (
            encoding_config["start_level"],
            encoding_config["start_step"],
            encoding_config["update_steps"],
        )
        self.current_level = self.start_level
        self.mask = torch.zeros(
            self.n_level * self.n_features_per_level,
            dtype=torch.float32,
            device=get_rank(),
        )
        self.mask[: self.current_level * self.n_features_per_level] = 1.0

    def forward(self, x):
        enc = self.encoding(x)
        enc = enc * self.mask + enc.detach() * (1 - self.mask)
        return enc

    def update_step(self, global_step):
        current_level = min(
            self.start_level
            + max(global_step - self.start_step, 0) // self.update_steps,
            self.n_level,
        )
        if current_level > self.current_level:
            print(f"Update current level of HashGrid to {current_level}")
            self.current_level = current_level
            self.mask[: self.current_level * self.n_features_per_level] = 1.0