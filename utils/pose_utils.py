import torch
import numpy as np

from utils.graphics_utils import fov2focal
from utils.dual_quaternion import quaternion_to_axis_angle, matrix_to_quaternion
import torch.nn.functional as F


trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def rodrigues_mat_to_rot(R):
    eps = 1e-16
    trc = np.trace(R)
    trc2 = (trc - 1.) / 2.
    # sinacostrc2 = np.sqrt(1 - trc2 * trc2)
    s = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if (1 - trc2 * trc2) >= eps:
        tHeta = np.arccos(trc2)
        tHetaf = tHeta / (2 * (np.sin(tHeta)))
    else:
        tHeta = np.real(np.arccos(trc2))
        tHetaf = 0.5 / (1 - tHeta / 6)
    omega = tHetaf * s
    return omega


def rodrigues_rot_to_mat(r):
    wx, wy, wz = r
    theta = np.sqrt(wx * wx + wy * wy + wz * wz)
    a = np.cos(theta)
    b = (1 - np.cos(theta)) / (theta * theta)
    c = np.sin(theta) / theta
    R = np.zeros([3, 3])
    R[0, 0] = a + b * (wx * wx)
    R[0, 1] = b * wx * wy - c * wz
    R[0, 2] = b * wx * wz + c * wy
    R[1, 0] = b * wx * wy + c * wz
    R[1, 1] = a + b * (wy * wy)
    R[1, 2] = b * wy * wz - c * wx
    R[2, 0] = b * wx * wz - c * wy
    R[2, 1] = b * wz * wy + c * wx
    R[2, 2] = a + b * (wz * wz)
    return R


def normalize(x):
    return x / np.linalg.norm(x)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = viewmatrix(vec2, up, center)
    return c2w

def render_path_spiral(c2ws, focal, zrate=.1, rots=3, N=300):
    c2w = poses_avg(c2ws)
    up = normalize(c2ws[:, :3, 1].sum(0))
    tt = c2ws[:,:3,3]
    rads = np.percentile(np.abs(tt), 90, 0)
    rads[:] = rads.max() * .05
    
    render_poses = []
    rads = np.array(list(rads) + [1.])
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        # c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        # z = normalize(c2w[:3, 2])
        render_poses.append(viewmatrix(z, up, c))
    render_poses = np.stack(render_poses, axis=0)
    render_poses = np.concatenate([render_poses, np.zeros_like(render_poses[..., :1, :])], axis=1)
    render_poses[..., 3, 3] = 1
    render_poses = np.array(render_poses, dtype=np.float32)
    return render_poses

def render_wander_path(view):
    focal_length = fov2focal(view.FoVy, view.image_height)
    R = view.R
    R[:, 1] = -R[:, 1]
    R[:, 2] = -R[:, 2]
    T = -view.T.reshape(-1, 1)
    pose = np.concatenate([R, T], -1)

    num_frames = 60
    max_disp = 5000.0  # 64 , 48

    max_trans = max_disp / focal_length  # Maximum camera translation to satisfy max_disp parameter
    output_poses = []

    for i in range(num_frames):
        x_trans = max_trans * np.sin(2.0 * np.pi * float(i) / float(num_frames))
        y_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(num_frames)) / 3.0  # * 3.0 / 4.0
        z_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(num_frames)) / 3.0

        i_pose = np.concatenate([
            np.concatenate(
                [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]], axis=1),
            np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]
        ], axis=0)  # [np.newaxis, :, :]

        i_pose = np.linalg.inv(i_pose)  # torch.tensor(np.linalg.inv(i_pose)).float()

        ref_pose = np.concatenate([pose, np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0)

        render_pose = np.dot(ref_pose, i_pose)
        output_poses.append(torch.Tensor(render_pose))

    return output_poses

def R_from_quaternions(quaternions: torch.tensor):
    quaternions = F.normalize(quaternions, p=2., dim=0)

    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3)).to(quaternions)

def R_from_axis_angle(k: torch.tensor, theta: torch.tensor):
    if torch.norm(k) == 0.:
        return torch.eye(3)
    k = F.normalize(k, p=2., dim=0)
    kx, ky, kz = k[0], k[1], k[2]
    cos, sin = torch.cos(theta), torch.sin(theta)
    R = torch.zeros((3, 3)).to(k)
    R[0, 0] = cos + (kx**2) * (1 - cos)
    R[0, 1] = kx * ky * (1 - cos) - kz * sin
    R[0, 2] = kx * kz * (1 - cos) + ky * sin
    R[1, 0] = kx * ky * (1 - cos) + kz * sin
    R[1, 1] = cos + (ky**2) * (1 - cos)
    R[1, 2] = ky * kz * (1 - cos) - kx * sin
    R[2, 0] = kx * kz * (1 - cos) - ky * sin
    R[2, 1] = ky * kz * (1 - cos) + kx * sin
    R[2, 2] = cos + (kz**2) * (1 - cos)

    # batch_size = k.size(0)
    # if batch_size == 0:
    #     return torch.eye(3).expand(batch_size, 3, 3)
    # k = F.normalize(k, p=2., dim=1)
    # R = exp_so3(k, theta)
    return  R


def R_from_axis_angle_batch(k: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    将轴角表示转换为旋转矩阵，支持批量计算。

    Args:
        k (torch.Tensor): 旋转轴向量，维度为 [b, 3]，其中 b 是 batch_size。
                          每个向量不需要预先归一化。
        theta (torch.Tensor): 旋转角度（弧度），维度为 [b, 1]。

    Returns:
        torch.Tensor: 旋转矩阵，维度为 [b, 3, 3]。
    """
    batch_size = k.shape[0]

    # 初始化所有旋转矩阵为单位矩阵 (处理 k 模为零的情况)
    R = torch.eye(3, device=k.device, dtype=k.dtype).unsqueeze(0).repeat(batch_size, 1, 1)

    # 计算 k 的 L2 范数
    norm_k = torch.norm(k, p=2, dim=1, keepdim=True)  # 维度 [b, 1]

    # 创建一个 mask，标记 k 向量模不为零的批次元素
    # 使用一个小 epsilon 进行比较，以避免浮点数精度问题
    non_zero_norm_mask = (norm_k.squeeze(1) > 1e-8)  # 维度 [b]

    # 如果所有 k 向量的模都为零，直接返回单位矩阵批次
    if not non_zero_norm_mask.any():
        return R

    # 仅对模不为零的 k 向量进行处理
    k_valid = k[non_zero_norm_mask]  # 维度 [b_valid, 3]
    theta_valid = theta[non_zero_norm_mask]  # 维度 [b_valid, 1]

    # 归一化 k 向量
    k_normalized = F.normalize(k_valid, p=2., dim=1)  # 维度 [b_valid, 3]

    # 提取分量
    kx, ky, kz = k_normalized[:, 0:1], k_normalized[:, 1:2], k_normalized[:, 2:3]  # 维度 [b_valid, 1]

    # 计算 cos(theta) 和 sin(theta)
    cos_theta = torch.cos(theta_valid)  # 维度 [b_valid, 1]
    sin_theta = torch.sin(theta_valid)  # 维度 [b_valid, 1]
    one_minus_cos_theta = 1.0 - cos_theta  # 维度 [b_valid, 1]

    # 构建旋转矩阵的各个元素
    # 注意：这里使用广播机制，kx, ky, kz, cos_theta, sin_theta 都是 [b_valid, 1]
    # 这样可以直接进行元素级乘法，结果仍是 [b_valid, 1]
    # 然后通过 .squeeze(-1) 或直接使用索引去除维度为 1 的尾部维度，以便后续赋值

    # R_valid 应该是一个 [b_valid, 3, 3] 的张量
    R_valid = torch.zeros((k_valid.shape[0], 3, 3), device=k.device, dtype=k.dtype)

    # 第一行
    R_valid[:, 0, 0] = (cos_theta + (kx ** 2) * one_minus_cos_theta).squeeze(-1)
    R_valid[:, 0, 1] = (kx * ky * one_minus_cos_theta - kz * sin_theta).squeeze(-1)
    R_valid[:, 0, 2] = (kx * kz * one_minus_cos_theta + ky * sin_theta).squeeze(-1)

    # 第二行
    R_valid[:, 1, 0] = (kx * ky * one_minus_cos_theta + kz * sin_theta).squeeze(-1)
    R_valid[:, 1, 1] = (cos_theta + (ky ** 2) * one_minus_cos_theta).squeeze(-1)
    R_valid[:, 1, 2] = (ky * kz * one_minus_cos_theta - kx * sin_theta).squeeze(-1)

    # 第三行
    R_valid[:, 2, 0] = (kx * kz * one_minus_cos_theta - ky * sin_theta).squeeze(-1)
    R_valid[:, 2, 1] = (ky * kz * one_minus_cos_theta + kx * sin_theta).squeeze(-1)
    R_valid[:, 2, 2] = (cos_theta + (kz ** 2) * one_minus_cos_theta).squeeze(-1)

    # 将计算出的有效旋转矩阵赋值回总的 R 张量中
    R[non_zero_norm_mask] = R_valid

    return R


# --- 推荐的更简洁且数值稳定的实现方式 (使用 exp_so3) ---

def exp_so3_batch(log_rot: torch.Tensor) -> torch.Tensor:
    """
    将 SO(3) 元素（李代数表示的旋转向量，即 k * theta）指数映射到 SO(3) 元素（旋转矩阵）。
    此函数处理批量输入，并且对小角度情况进行数值稳定处理。

    Args:
        log_rot (torch.Tensor): 旋转向量，形状为 [b, n, 3]。
                                等同于 k * theta，其中 k 是归一化的轴向量，theta 是角度。

    Returns:
        torch.Tensor: 旋转矩阵，形状为 [b, n, 3, 3]。
    """
    b, n, _ = log_rot.shape
    device = log_rot.device
    dtype = log_rot.dtype

    # 计算旋转角度 theta 的平方
    theta_sq = torch.sum(log_rot ** 2, dim=-1, keepdim=True)  # 形状 [b, n, 1]

    # 计算旋转角度 theta (范数)
    theta = torch.sqrt(theta_sq)  # 形状 [b, n, 1]

    # kx, ky, kz 向量
    k_vec = log_rot  # 形状 [b, n, 3]

    # 生成一个批次的单位矩阵
    identity = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)  # 形状 [1, 1, 3, 3]
    identity = identity.expand(b, n, -1, -1)  # 扩展到 [b, n, 3, 3]

    # 对于 theta 接近零的情况，使用泰勒展开进行数值稳定处理
    # 当 theta -> 0 时，sin(theta)/theta -> 1, (1-cos(theta))/theta^2 -> 0.5
    small_angle_mask = (theta < 1e-6)  # 形状 [b, n, 1]

    # 计算 sin(theta)/theta 和 (1-cos(theta))/theta^2
    # 对于小角度，直接使用其极限值
    sin_theta_over_theta = torch.where(
        small_angle_mask,
        torch.ones_like(theta),  # 极限值 1
        torch.sin(theta) / theta
    )  # 形状 [b, n, 1]

    one_minus_cos_theta_over_theta_sq = torch.where(
        small_angle_mask,
        0.5 * torch.ones_like(theta),  # 极限值 0.5
        (1.0 - torch.cos(theta)) / theta_sq
    )  # 形状 [b, n, 1]

    # 构建叉积矩阵 K = [0, -kz, ky; kz, 0, -kx; -ky, kx, 0]
    # k_vec_x = k_vec[..., 0].unsqueeze(-1) # [b, n, 1]
    # k_vec_y = k_vec[..., 1].unsqueeze(-1) # [b, n, 1]
    # k_vec_z = k_vec[..., 2].unsqueeze(-1) # [b, n, 1]

    # K 矩阵的批次构建
    K = torch.zeros(b, n, 3, 3, device=device, dtype=dtype)
    K[..., 0, 1] = -k_vec[..., 2]
    K[..., 0, 2] = k_vec[..., 1]
    K[..., 1, 0] = k_vec[..., 2]
    K[..., 1, 2] = -k_vec[..., 0]
    K[..., 2, 0] = -k_vec[..., 1]
    K[..., 2, 1] = k_vec[..., 0]

    # 罗德里格斯公式的矩阵形式: R = I + (sin(theta)/theta) * K + ((1-cos(theta))/theta^2) * K^2
    # K_sq = torch.matmul(K, K) # K^2
    # K_sq 可以直接计算为 k * k^T - ||k||^2 * I
    # 然而，由于 K 中的 k 已经包含了 theta，所以 K^2 中的 theta 也是平方。
    # 更直接的方式是 K^2 的元素是 k_i k_j - delta_ij ||k||^2
    # 这里 k 实际上是 theta * k_normalized
    # 所以 K_sq 的元素是 (theta k_i)(theta k_j) - delta_ij (theta^2)
    # 也就是 (theta^2) * (k_i_norm k_j_norm - delta_ij)

    # 我们可以直接使用 K * K 来计算 K_sq
    K_sq = torch.matmul(K, K)

    # 最终的旋转矩阵 R
    R = identity + sin_theta_over_theta.unsqueeze(-1) * K + \
        one_minus_cos_theta_over_theta_sq.unsqueeze(-1) * K_sq

    return R


def R_from_axis_angle_exp_so3(k: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    使用指数映射 exp_so3_batch 将轴角表示 (k, theta) 转换为旋转矩阵 R。
    支持批量计算。

    Args:
        k (torch.Tensor): 轴向量，形状为 [b, n, 3]。
        theta (torch.Tensor): 旋转角度（弧度），形状为 [b, n, 1]。

    Returns:
        torch.Tensor: 旋转矩阵，形状为 [b, n, 3, 3]。
    """
    # 归一化 k，并处理零向量
    k_norm = torch.norm(k, p=2, dim=-1, keepdim=True)

    # 对于零向量，归一化结果为 NaN。我们需要将其设为0，这样 k_normalized * theta 结果也是0
    # 避免在 log_rot 中引入 NaN。
    # k_normalized = F.normalize(k, p=2, dim=-1) # 这会在 k_norm=0 时产生 NaN

    # 安全的归一化：当范数为0时，归一化结果为0
    k_normalized = torch.where(
        k_norm < 1e-6,
        torch.zeros_like(k),
        k / k_norm
    )

    # 计算旋转向量 log_rot = k_normalized * theta
    log_rot = k_normalized * theta  # 形状 [b, n, 3]

    return exp_so3_batch(log_rot)


def rigid_transform(xyz: torch.Tensor, mask, qrs, trans, axes_o, state, canonical=0.5):
    '''
    Perform the rigid transformation: R @ (x - c) + c + T

    Transform the positions from canonical state=0.5 to state=0 or state=1
    '''

    # if gaussians.use_canonical:
    #     scaling = (gaussians.canonical - state) / gaussians.canonical
    # else:
    #     scaling = state
    scaling = (canonical - state) / canonical

    positions = xyz.unsqueeze(1).repeat(1,axes_o.shape[0],1) - axes_o

    if scaling == 1.:
        R = R_from_quaternions(qrs)

        #positions = torch.matmul(R, positions.T).T
    elif scaling == -1.:
        inv_sc = torch.tensor([1., -1., -1., -1]).to(qrs)
        inv_q = inv_sc * qrs
        R = R_from_quaternions(inv_q)
        #positions = torch.matmul(R, positions.T).T
    else:
        axis, angles = quaternion_to_axis_angle(qrs)  # the angle means from t=0 to t=0.5
        tgt_angles = scaling * angles
        R = R_from_axis_angle_batch(axis, tgt_angles)
        #positions = torch.matmul(R, positions.T).T

    # rotation_matrix = torch.eye(3).unsqueeze(dim=0).repeat(gaussians.get_xyz.shape[0],1,1).cuda()
    # rotation_matrix[gaussians.dynamic_part_mask] = R
    #rotation_matrix = torch.einsum('nk, kl->nl', mask, R)
    rotation_matrix = torch.einsum('nk,kij->nkij', mask, R)
    rot = torch.einsum('nk,kij->nij', mask, R)
    quaternions = matrix_to_quaternion(rot)
    #positions = torch.einsum('nkij,nka->na',rotation_matrix, positions)
    positions = torch.matmul(rotation_matrix, positions.unsqueeze(dim=-1)).squeeze(dim=-1)
    #positions = torch.matmul(rotation_matrix, positions.unsqueeze(dim=-1)).squeeze(dim=-1)
    positions = (positions + axes_o + trans).sum(dim=1)
    return positions, quaternions