import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
from scipy.spatial import KDTree
from sklearn.cluster import SpectralClustering
from pytorch3d.loss import chamfer_distance

def generate_realistic_test_data(N=2048, k=4, device='cuda'):
    """
    生成更真实的测试数据，使得同一part的点在空间上聚集
    """
    points_list = []
    labels_list = []

    # 为每个part生成一个聚类中心
    for part_id in range(k):
        n_points_per_part = N // k
        # 随机生成聚类中心
        center = torch.randn(1, 3) * 2.0
        # 在中心周围生成点
        part_points = center + torch.randn(n_points_per_part, 3) * 0.3
        part_labels = torch.full((n_points_per_part,), part_id, dtype=torch.long)

        points_list.append(part_points)
        labels_list.append(part_labels)

    # 合并所有点
    points = torch.cat(points_list, dim=0).to(device)
    true_labels = torch.cat(labels_list, dim=0).to(device)

    # 生成part_mask：在真实标签基础上添加噪声
    part_mask = torch.zeros(N, k, device=device)
    for i in range(N):
        # 主要概率给真实标签
        part_mask[i, true_labels[i]] = 0.7 + torch.rand(1).item() * 0.2
        # 其他part分配小概率
        for j in range(k):
            if j != true_labels[i]:
                part_mask[i, j] = torch.rand(1).item() * 0.1

    # 归一化
    part_mask = part_mask / part_mask.sum(dim=1, keepdim=True)
    part_mask.requires_grad = True

    return points, part_mask, true_labels


def find_knn_indices_np(points, k_neighbors, kd_tree=None):
    """
    为点云查找k近邻的索引。
    """
    if kd_tree is None:
        kd_tree = KDTree(points)
    _, indices = kd_tree.query(points, k=k_neighbors + 1)
    return indices[:, 1:] if indices.shape[1] > k_neighbors else indices # 移除点本身

def identify_point_boundaries(points, masks, knn_indices, k_neighbors, boundary_ratio_threshold=0.1):
    """
    识别点云中点级别的边界点。
    """
    num_points = points.shape[0]
    is_point_boundary = np.zeros(num_points, dtype=bool)

    if num_points == 0:
        return is_point_boundary

    for i in range(num_points):
        current_mask = masks[i]
        neighbor_idx = knn_indices[i]
        neighbor_masks = masks[neighbor_idx]

        diff_label_count = np.sum(neighbor_masks != current_mask)
        if diff_label_count / k_neighbors > boundary_ratio_threshold:
            is_point_boundary[i] = True

    return is_point_boundary


class BoundaryPointVotingModule(nn.Module):
    """
    边界点局部投票模块，用于关节物体part分割的边界一致性优化
    """

    def __init__(self,
                 neighbor_k=16,  # K近邻数量用于边界检测
                 different_label_threshold=0.3,  # 边界判定阈值（新增）
                 boundary_neighbor_radius=0.02,
                 dbscan_eps=0.05,
                 dbscan_min_samples=5,  # 5
                 voting_radius=0.03,
                 voting_k=16,
                 boundary_threshold=0.6, #0.3
                 boundary_radius = 3):  # 1
        super(BoundaryPointVotingModule, self).__init__()
        self.neighbor_k = neighbor_k
        self.different_label_threshold = different_label_threshold  # 新增
        self.boundary_neighbor_radius = boundary_neighbor_radius
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.voting_radius = voting_radius
        self.voting_k = voting_k
        self.boundary_threshold = boundary_threshold
        self.boundary_radius = boundary_radius

    # def extract_boundary_points(self, points, part_mask):
    #     """
    #     提取两个part间的重叠边界点：
    #     1. 每个点分配到概率最大的part
    #     2. 找出K近邻中存在足够比例不同part标签的点作为边界点
    #
    #     Args:
    #         points: (N, 3) 点云坐标
    #         part_mask: (N, k) 每个点属于各part的概率
    #     Returns:
    #         boundary_mask: (N,) boolean tensor，True表示边界点
    #         boundary_indices: (M,) 边界点的索引
    #         part_labels: (N,) 每个点的part标签
    #     """
    #     N, k = part_mask.shape
    #     device = part_mask.device
    #
    #     # 1. 将每个点分配到概率最大的part
    #     part_labels = torch.argmax(part_mask, dim=1)  # (N,)
    #
    #     # 2. 使用K近邻查找边界点
    #     points_np = points.detach().cpu().numpy()
    #
    #     # 构建KNN搜索
    #     nbrs = NearestNeighbors(n_neighbors=self.neighbor_k + 1,
    #                             algorithm='kd_tree').fit(points_np)
    #     distances, indices = nbrs.kneighbors(points_np)
    #
    #     # 转换回torch tensor
    #     neighbor_indices = torch.from_numpy(indices[:, 1:]).long().to(device)  # (N, K) 排除自己
    #
    #     # 3. 检测边界点：邻域中不同part标签的点达到一定比例
    #     boundary_mask = torch.zeros(N, dtype=torch.bool, device=device)
    #
    #     # 设置阈值：至少30%的邻居属于不同part才认为是边界点
    #     different_label_threshold = 0.3
    #
    #     for i in range(N):
    #         # 获取当前点的part标签
    #         current_label = part_labels[i]
    #         # 获取邻居点的part标签
    #         neighbor_labels = part_labels[neighbor_indices[i]]  # (K,)
    #         # 计算不同标签的比例
    #         different_ratio = (neighbor_labels != current_label).float().mean().item()
    #
    #         # 如果不同标签比例超过阈值，则为边界点
    #         if different_ratio >= different_label_threshold:
    #             boundary_mask[i] = True
    #
    #     boundary_indices = torch.where(boundary_mask)[0]
    #
    #     return boundary_mask, boundary_indices, part_labels
    def extract_boundary_points(self, points, part_mask,
                                different_label_threshold=0.3):
        """
        提取两个part间的重叠边界点（向量化版本，更快）

        Args:
            points: (N, 3) 点云坐标
            part_mask: (N, k) 每个点属于各part的概率
            different_label_threshold: float, 邻域内不同标签比例阈值
        Returns:
            boundary_mask: (N,) boolean tensor，True表示边界点
            boundary_indices: (M,) 边界点的索引
            part_labels: (N,) 每个点的part标签
        """
        N, k = part_mask.shape
        device = part_mask.device

        # 1. 将每个点分配到概率最大的part
        part_labels = torch.argmax(part_mask, dim=1)  # (N,)

        # 2. 使用K近邻查找
        points_np = points.detach().cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=self.neighbor_k + 1, radius= 3, # 1
                                algorithm='kd_tree').fit(points_np)
        distances, indices = nbrs.kneighbors(points_np)

        # 转换回torch tensor
        neighbor_indices = torch.from_numpy(indices[:, 1:]).long().to(device)  # (N, K)

        # 3. 向量化计算边界点
        # 获取所有邻居的标签
        neighbor_labels = part_labels[neighbor_indices]  # (N, K)

        # 计算每个点的标签与其邻居标签的差异
        current_labels_expanded = part_labels.unsqueeze(1).expand(-1, self.neighbor_k)  # (N, K)
        different_labels = (neighbor_labels != current_labels_expanded).float()  # (N, K)

        # 计算不同标签的比例
        different_ratio = different_labels.mean(dim=1)  # (N,)

        # 边界点：不同标签比例超过阈值
        boundary_mask = different_ratio >= different_label_threshold
        boundary_indices = torch.where(boundary_mask)[0]

        return boundary_mask, boundary_indices

    # def extract_boundary_points(self, part_mask):
    #     """
    #     提取边界点：计算每个点的part不确定性
    #     Args:
    #         part_mask: (N, k) 每个点属于各part的概率
    #     Returns:
    #         boundary_mask: (N,) boolean tensor，True表示边界点
    #         boundary_indices: (M,) 边界点的索引
    #     """
    #     N, k = part_mask.shape
    #
    #     # 方法1: 基于熵的边界检测
    #     # 熵越大，说明点在多个part间越不确定
    #     epsilon = 1e-8
    #     entropy = -torch.sum(part_mask * torch.log(part_mask + epsilon), dim=1)  # (N,)
    #     max_entropy = torch.log(torch.tensor(k, dtype=torch.float32, device=part_mask.device))
    #     normalized_entropy = entropy / max_entropy
    #
    #     # 方法2: 基于最大概率与次大概率差值的边界检测
    #     sorted_probs, _ = torch.sort(part_mask, dim=1, descending=True)
    #     prob_diff = sorted_probs[:, 0] - sorted_probs[:, 1]  # (N,)
    #
    #     # 综合判断：高熵或小概率差
    #     boundary_mask = (normalized_entropy > self.boundary_threshold) | (prob_diff < self.boundary_threshold)
    #     boundary_indices = torch.where(boundary_mask)[0]
    #
    #     return boundary_mask, boundary_indices

    def spatial_clustering(self, points, boundary_indices):
        """
        对边界点进行空间聚类
        Args:
            points: (N, 3) 点云坐标
            boundary_indices: (M,) 边界点索引
        Returns:
            cluster_labels: (M,) 每个边界点的聚类标签，-1表示噪声点
        """
        if len(boundary_indices) == 0:
            return torch.tensor([], dtype=torch.long, device=points.device)

        # 提取边界点坐标
        boundary_points = points[boundary_indices]  # (M, 3)

        # 转换为numpy进行DBSCAN聚类（sklearn不支持GPU）
        boundary_points_np = boundary_points.detach().cpu().numpy()

        # DBSCAN聚类
        # clustering = DBSCAN(eps=self.dbscan_eps,
        #                     min_samples=self.dbscan_min_samples,
        #                     metric='euclidean')
        # cluster_labels_np = clustering.fit_predict(boundary_points_np)
        # KNN

        cluster = SpectralClustering(10, assign_labels='discretize', random_state=0)
        labels = cluster.fit_predict(boundary_points_np)

        # 转回torch tensor
        cluster_labels = torch.from_numpy(labels).long().to(points.device)

        return cluster_labels

    def local_voting(self, points, part_mask, boundary_indices, cluster_labels):
        """
        对每个聚类的边界点进行局部投票
        Args:
            points: (N, 3) 点云坐标
            part_mask: (N, k) part概率
            boundary_indices: (M,) 边界点索引
            cluster_labels: (M,) 聚类标签
        Returns:
            voting_mask: (M, k) 投票后的边界点part概率
        """
        M = len(boundary_indices)
        k = part_mask.shape[1]
        device = points.device

        if M == 0:
            return torch.zeros((0, k), device=device)

        voting_mask = torch.zeros((M, k), device=device)

        # 获取唯一的聚类标签（排除-1噪声点）
        unique_clusters = torch.unique(cluster_labels)
        unique_clusters = unique_clusters[unique_clusters >= 0]

        boundary_points = points[boundary_indices]

        for cluster_id in unique_clusters:
            # 获取当前聚类的点
            cluster_mask = cluster_labels == cluster_id
            cluster_point_indices = boundary_indices[cluster_mask]
            cluster_points = points[cluster_point_indices]  # (C, 3)

            # 对聚类中的每个点进行局部投票
            for i, point_idx in enumerate(cluster_point_indices):
                point = points[point_idx:point_idx + 1]  # (1, 3)

                # 计算与所有点的距离
                dists = torch.norm(points - point, dim=1)  # (N,)

                # 方法1: 基于半径的邻域
                neighbor_mask = dists < self.voting_radius

                # 方法2: K近邻（可选）
                if neighbor_mask.sum() < self.voting_k:
                    _, knn_indices = torch.topk(dists, k=min(self.voting_k, len(dists)),
                                                largest=False)
                    neighbor_mask = torch.zeros_like(neighbor_mask)
                    neighbor_mask[knn_indices] = True

                # 获取邻域点的part概率
                neighbor_probs = part_mask[neighbor_mask]  # (K, k)

                if len(neighbor_probs) > 0:
                    # 距离加权投票
                    neighbor_dists = dists[neighbor_mask].unsqueeze(1)  # (K, 1)
                    weights = torch.exp(-neighbor_dists / self.voting_radius)  # (K, 1)
                    weights = weights / (weights.sum() + 1e-8)

                    # 加权平均得到投票结果
                    voted_prob = (neighbor_probs * weights).sum(dim=0)  # (k,)
                    voting_mask[cluster_mask][i] = voted_prob
                else:
                    # 如果没有邻域点，保持原始概率
                    voting_mask[cluster_mask][i] = part_mask[point_idx]

        # 对噪声点（cluster_id == -1），保持原始概率
        noise_mask = cluster_labels == -1
        if noise_mask.any():
            voting_mask[noise_mask] = part_mask[boundary_indices[noise_mask]]

        # 归一化
        voting_mask = F.softmax(voting_mask, dim=1)

        return voting_mask

    def forward(self, points, part_mask):
        """
        前向传播
        Args:
            points: (N, 3) 点云坐标，GPU tensor
            part_mask: (N, k) part概率，GPU tensor
        Returns:
            boundary_indices: 边界点索引
            voting_mask: 投票后的边界点概率
            cluster_labels: 聚类标签
        """
        # 1. 提取边界点
        #boundary_mask, boundary_indices = self.extract_boundary_points(points, part_mask)
        boundary_mask, boundary_indices = self.extract_boundary_points(points, part_mask, self.different_label_threshold)
        boundary_mask = boundary_mask.detach() * 1


        # 2. 空间聚类
        cluster_labels = self.spatial_clustering(points, boundary_indices)

        # 3. 局部投票
        voting_mask = self.local_voting(points, part_mask, boundary_indices, cluster_labels)
        cluster_labels += 1
        boundary_mask[boundary_indices] = cluster_labels
        return boundary_mask, boundary_indices, voting_mask


class LocalConsistencyVotingLoss(nn.Module):
    """
    局部一致性投票损失
    """

    def __init__(self, loss_type='kl', temperature=1.0):
        super(LocalConsistencyVotingLoss, self).__init__()
        self.loss_type = loss_type
        self.temperature = temperature

    def forward(self, part_mask, voting_mask, boundary_indices):
        """
        计算边界点处原始mask与投票mask之间的一致性损失
        Args:
            part_mask: (N, k) 原始part概率
            voting_mask: (M, k) 边界点投票概率
            boundary_indices: (M,) 边界点索引
        Returns:
            loss: scalar tensor
        """
        if len(boundary_indices) == 0:
            return torch.tensor(0.0, device=part_mask.device, requires_grad=True)

        # 提取边界点的原始概率
        boundary_part_mask = part_mask[boundary_indices]  # (M, k)

        if self.loss_type == 'kl':
            # KL散度损失
            log_boundary_mask = torch.log(boundary_part_mask + 1e-8)
            loss = F.kl_div(log_boundary_mask, voting_mask, reduction='batchmean')

        elif self.loss_type == 'mse':
            # MSE损失
            loss = F.mse_loss(boundary_part_mask, voting_mask)

        elif self.loss_type == 'cosine':
            # 余弦相似度损失
            cos_sim = F.cosine_similarity(boundary_part_mask, voting_mask, dim=1)
            loss = 1.0 - cos_sim.mean()

        elif self.loss_type == 'smooth_l1':
            # Smooth L1损失
            loss = F.smooth_l1_loss(boundary_part_mask, voting_mask)

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss

def visualize_boundary_points(points, part_labels, boundary_indices,
                              colors=None, point_size=2.0):
    """
    使用Open3D可视化边界点

    Args:
        points: (N, 3) numpy array or torch tensor，点云坐标
        part_labels: (N,) numpy array or torch tensor，part标签
        boundary_indices: (M,) numpy array or torch tensor，边界点索引
        colors: optional, list of RGB colors for each part
        point_size: float, 点的显示大小
    """
    # 转换为numpy
    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
    if torch.is_tensor(part_labels):
        part_labels = part_labels.detach().cpu().numpy()
    if torch.is_tensor(boundary_indices):
        boundary_indices = boundary_indices.detach().cpu().numpy()

    N = len(points)
    k = int(part_labels.max()) + 1

    # 默认颜色方案
    if colors is None:
        colors = [
            [1.0, 0.0, 0.0],  # 红色
            [0.0, 1.0, 0.0],  # 绿色
            [0.0, 0.0, 1.0],  # 蓝色
            [1.0, 1.0, 0.0],  # 黄色
            [1.0, 0.0, 1.0],  # 品红
            [0.0, 1.0, 1.0],  # 青色
        ]
        # 如果part数量超过默认颜色，生成更多颜色
        while len(colors) < k:
            colors.append(np.random.rand(3).tolist())

    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 为每个点分配颜色
    point_colors = np.zeros((N, 3))
    for i in range(N):
        point_colors[i] = colors[part_labels[i]]

    # 将边界点标记为白色或特殊颜色
    boundary_color = [1.0, 1.0, 1.0]  # 白色
    point_colors[boundary_indices] = boundary_color

    pcd.colors = o3d.utility.Vector3dVector(point_colors)

    # 可视化
    print(f"Total points: {N}")
    print(f"Boundary points: {len(boundary_indices)}")
    print(f"Number of parts: {k}")
    print("\nVisualization:")
    print("- Non-boundary points: colored by part label")
    print("- Boundary points: white color")
    print("\nPress 'Q' to close the window")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Boundary Points Visualization")
    vis.add_geometry(pcd)

    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.background_color = np.array([0.1, 0.1, 0.1])

    vis.run()
    vis.destroy_window()


def visualize_boundary_points_separate(points, part_labels, boundary_indices,
                                       colors=None, point_size=2.0):
    """
    分别可视化：左侧显示所有点，右侧仅显示边界点

    Args:
        points: (N, 3) numpy array or torch tensor
        part_labels: (N,) numpy array or torch tensor
        boundary_indices: (M,) numpy array or torch tensor
        colors: optional, list of RGB colors for each part
        point_size: float
    """
    # 转换为numpy
    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
    if torch.is_tensor(part_labels):
        part_labels = part_labels.detach().cpu().numpy()
    if torch.is_tensor(boundary_indices):
        boundary_indices = boundary_indices.detach().cpu().numpy()

    N = len(points)
    k = int(part_labels.max()) + 1

    # 默认颜色方案
    if colors is None:
        colors = [
            [1.0, 0.0, 0.0],  # 红色
            [0.0, 1.0, 0.0],  # 绿色
            [0.0, 0.0, 1.0],  # 蓝色
            [1.0, 1.0, 0.0],  # 黄色
            [1.0, 0.0, 1.0],  # 品红
            [0.0, 1.0, 1.0],  # 青色
        ]
        while len(colors) < k:
            colors.append(np.random.rand(3).tolist())

    # 创建完整点云
    pcd_full = o3d.geometry.PointCloud()
    pcd_full.points = o3d.utility.Vector3dVector(points)
    point_colors = np.array([colors[label] for label in part_labels])
    pcd_full.colors = o3d.utility.Vector3dVector(point_colors)

    # 创建边界点云
    boundary_points = points[boundary_indices]
    pcd_boundary = o3d.geometry.PointCloud()
    pcd_boundary.points = o3d.utility.Vector3dVector(boundary_points)
    boundary_labels = part_labels[boundary_indices]
    boundary_colors = np.array([colors[label] for label in boundary_labels])
    pcd_boundary.colors = o3d.utility.Vector3dVector(boundary_colors)

    # 平移边界点云以便并排显示
    bbox = pcd_full.get_axis_aligned_bounding_box()
    offset = bbox.get_max_bound() - bbox.get_min_bound()
    pcd_boundary.translate([offset[0] * 1.5, 0, 0])

    print(f"Total points: {N}")
    print(f"Boundary points: {len(boundary_indices)} ({len(boundary_indices) / N * 100:.2f}%)")
    print(f"Number of parts: {k}")
    print("\nVisualization:")
    print("- Left: All points colored by part")
    print("- Right: Boundary points only")

    # 可视化
    o3d.visualization.draw_geometries(
        [pcd_full, pcd_boundary],
        window_name="Left: Full Point Cloud | Right: Boundary Points",
        width=1600,
        height=900,
        point_show_normal=False
    )


# 修改后的使用示例
def example_usage_with_visualization():
    """
    使用示例（含可视化），使用真实数据
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 生成更真实的测试数据
    N = 2048
    k = 4
    points, part_mask, true_labels = generate_realistic_test_data(N, k, device)

    print(f"Generated {N} points with {k} parts")

    # 创建模块，设置合适的阈值
    voting_module = BoundaryPointVotingModule(
        neighbor_k=20,  # 增加K近邻数量
        different_label_threshold=0.25,  # 调整阈值，25%的邻居不同即为边界
        boundary_neighbor_radius=0.02,
        dbscan_eps=0.08,
        dbscan_min_samples=5,
        voting_radius=0.05,
        voting_k=16
    ).to(device)

    # 提取边界点
    boundary_mask, boundary_indices, part_labels = voting_module.extract_boundary_points(
        points, part_mask
    )

    print(f"\nBoundary Detection Results:")
    print(f"Total points: {N}")
    print(f"Boundary points: {len(boundary_indices)} ({len(boundary_indices) / N * 100:.2f}%)")
    print(f"Non-boundary points: {N - len(boundary_indices)} ({(N - len(boundary_indices)) / N * 100:.2f}%)")

    # 统计每个part的边界点数量
    for part_id in range(k):
        part_boundary_count = ((part_labels[boundary_indices] == part_id).sum().item())
        part_total_count = (part_labels == part_id).sum().item()
        print(f"Part {part_id}: {part_boundary_count}/{part_total_count} boundary points")

    # 可视化
    visualize_boundary_points_separate(
        points=points,
        part_labels=part_labels,
        boundary_indices=boundary_indices,
        point_size=3.0
    )

    return voting_module


# 使用示例
def example_usage():
    """
    使用示例
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 生成模拟数据
    N = 2048  # 点数
    k = 4  # part数量

    # 点云坐标 (N, 3)
    points = torch.randn(N, 3, device=device, requires_grad=False)

    # part概率 (N, k) - 需要梯度
    part_logits = torch.randn(N, k, device=device, requires_grad=True)
    part_mask = F.softmax(part_logits, dim=1)

    # 创建模块
    voting_module = BoundaryPointVotingModule(
        boundary_threshold=0.3,
        dbscan_eps=0.05,
        dbscan_min_samples=5,
        voting_radius=0.03,
        voting_k=16
    ).to(device)

    loss_fn = LocalConsistencyVotingLoss(loss_type='kl').to(device)

    # 前向传播
    boundary_indices, voting_mask, cluster_labels = voting_module(points, part_mask)

    print(f"Total points: {N}")
    print(f"Boundary points: {len(boundary_indices)}")
    print(f"Number of clusters: {len(torch.unique(cluster_labels[cluster_labels >= 0]))}")
    print(f"Voting mask shape: {voting_mask.shape}")

    # 计算损失
    loss = loss_fn(part_mask, voting_mask, boundary_indices)
    print(f"Loss: {loss.item():.6f}")

    # 反向传播测试
    loss.backward()
    print(f"Gradient exists: {part_logits.grad is not None}")
    print(f"Gradient norm: {part_logits.grad.norm().item():.6f}")

    return voting_module, loss_fn


if __name__ == "__main__":
    #example_usage()
    example_usage_with_visualization()