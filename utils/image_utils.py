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
import torch.nn.functional as F
import numpy as np
import torchvision
import cv2
import os

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def dilate(bin_img, ksize=5):
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out

def erode(bin_img, ksize=5):
    out = 1 - dilate(1 - bin_img, ksize)
    return out

def normal_tensor_to_img(normal_tensor, render_normal_path="/media/wd/work/REArtGS2/method", img_name = "normal.png"):
    max = 1
    min = 0
    tensor_max = normal_tensor.max()
    tensor_min = normal_tensor.min()
    normal_tensor = normal_tensor.permute(1,2,0)
    normal = normal_tensor / (normal_tensor.norm(dim=-1, keepdim=True) + 1.0e-8)
    normal = normal.detach().cpu().numpy()
    normal = ((normal + 1) * 127.5).astype(np.uint8).clip(0, 255)
    cv2.imwrite(os.path.join(render_normal_path, img_name), normal)
    #torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))

def depth_tensor_to_img(depth_tensor, render_depth_path="/media/wd/work/REArtGS2/method", img_name = "depth.png"):
    max = 1
    min = 0
    depth = depth_tensor.squeeze().detach().cpu().numpy()
    depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
    depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(render_depth_path, img_name), depth_color)


import cv2
import numpy as np


def remove_isolated_noise(image_path, output_path=None, method='connected_components',
                          min_area=10, kernel_size=3, min_neighbors=2):
    """
    检测并移除图像中的孤立噪声点，将其填充为白色背景

    参数:
    - image_path: 输入图像路径
    - output_path: 输出图像路径（可选）
    - method: 检测方法 ('connected_components', 'morphology', 'custom_filter')
    - min_area: 最小面积阈值（仅用于connected_components方法）
    - kernel_size: 形态学操作核大小
    - min_neighbors: 最小邻居数量（仅用于custom_filter方法）
    """

    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图像，请检查文件路径")

    # 创建图像的副本，用于处理
    result = img.copy()

    # 转换为灰度图用于处理
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 二值化图像
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    if method == 'connected_components':
        # 方法1: 使用连通组件分析
        noise_mask = np.zeros_like(gray)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        for i in range(1, num_labels):  # 跳过背景(标签0)
            if stats[i, cv2.CC_STAT_AREA] <= min_area:
                # 标记小区域为噪声
                noise_mask[labels == i] = 255

    elif method == 'morphology':
        # 方法2: 使用形态学操作
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # 开运算去除小点，与原图比较找出被去除的点
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        noise_mask = cv2.subtract(binary, opened)

    elif method == 'custom_filter':
        # 方法3: 使用自定义滤波器检测孤立点
        noise_mask = np.zeros_like(gray)
        height, width = binary.shape

        # 遍历每个像素
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if binary[y, x] > 0:  # 如果当前点是前景
                    # 计算8邻域中前景点的数量
                    neighbors = np.sum(binary[y - 1:y + 2, x - 1:x + 2] > 0) - 1  # 减去中心点
                    if neighbors <= min_neighbors:
                        noise_mask[y, x] = 255

    else:
        raise ValueError("不支持的检测方法，请选择 'connected_components', 'morphology' 或 'custom_filter'")

    # 将噪声点填充为白色
    if len(result.shape) > 2:  # 彩色图像
        result[noise_mask == 255] = [255, 255, 255]
    else:  # 灰度图像
        result[noise_mask == 255] = 255

    # 统计噪声点数量
    noise_count = np.sum(noise_mask == 255)
    print(f"检测到 {noise_count} 个噪声点并已填充为白色")

    # 保存结果图像
    if output_path:
        cv2.imwrite(output_path, result)
        print(f"结果已保存至: {output_path}")

    return result, noise_mask


def visualize_comparison(original, processed, noise_mask):
    """可视化原始图像、处理后的图像和噪声掩模"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    if len(original.shape) > 2:
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[1].imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    else:
        axes[0].imshow(original, cmap='gray')
        axes[1].imshow(processed, cmap='gray')

    axes[2].imshow(noise_mask, cmap='gray')

    axes[0].set_title('原始图像')
    axes[1].set_title('处理后图像')
    axes[2].set_title('检测到的噪声点')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# 使用示例
if __name__ == "__main__":
    # 示例1: 使用连通组件分析方法
    result1, noise_mask1 = remove_isolated_noise(
        'input_image.jpg',
        'output_connected.jpg',
        method='connected_components',
        min_area=15  # 面积小于15像素的区域视为噪声
    )

    # 示例2: 使用形态学方法
    result2, noise_mask2 = remove_isolated_noise(
        'input_image.jpg',
        'output_morphology.jpg',
        method='morphology',
        kernel_size=3
    )

    # 示例3: 使用自定义滤波器方法
    result3, noise_mask3 = remove_isolated_noise(
        'input_image.jpg',
        'output_custom.jpg',
        method='custom_filter',
        min_neighbors=2  # 周围少于2个邻居的点视为噪声
    )

    # 可视化比较（需要matplotlib）
    # visualize_comparison(cv2.imread('input_image.jpg'), result1, noise_mask1)