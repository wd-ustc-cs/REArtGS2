import json
import os
from argparse import ArgumentParser

import numpy as np
import open3d as o3d
import pandas as pd
import seaborn as sns
import torch
from pytorch_lightning import seed_everything
from scipy.optimize import linear_sum_assignment

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from scene import DeformModel, GaussianModel, Scene
from utils.general_utils import safe_state
from utils.metrics import (
    compute_recon_error,
    eval_CD_2states,
    eval_axis_and_state,
    read_gt,
)
from utils.system_utils import searchForMaxIteration


def get_rotation_axis_angle(k, theta):
    if np.linalg.norm(k) == 0.0:
        return np.eye(3)
    k = k / np.linalg.norm(k)
    kx, ky, kz = k[0], k[1], k[2]
    cos, sin = np.cos(theta), np.sin(theta)
    R = np.zeros((3, 3))
    R[0, 0] = cos + (kx ** 2) * (1 - cos)
    R[0, 1] = kx * ky * (1 - cos) - kz * sin
    R[0, 2] = kx * kz * (1 - cos) + ky * sin
    R[1, 0] = kx * ky * (1 - cos) + kz * sin
    R[1, 1] = cos + (ky ** 2) * (1 - cos)
    R[1, 2] = ky * kz * (1 - cos) - kx * sin
    R[2, 0] = kx * kz * (1 - cos) - ky * sin
    R[2, 1] = ky * kz * (1 - cos) + kx * sin
    R[2, 2] = cos + (kz ** 2) * (1 - cos)
    return R


def save_axis_mesh(k, center, filepath):
    axis = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.02,
        cone_radius=0.04,
        cylinder_height=0.7,
        cone_height=0.04,
    )
    arrow = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    n = np.cross(arrow, k)
    rad = np.arccos(np.clip(np.dot(arrow, k), -1.0, 1.0))
    R_arrow = get_rotation_axis_angle(n, rad)
    axis.rotate(R_arrow, center=(0, 0, 0))
    axis.translate(center[:3])
    o3d.io.write_triangle_mesh(filepath, axis)


joint_type_dict = {
    "r": "hinge",
    "p": "slider",
}


def export_joint_info_json(pred_joint_list, mesh_files, exp_dir):
    meta_info = []
    for i, joint_info in enumerate(pred_joint_list):
        if i == 0:
            entry = {
                "id": i,
                "parent": -1,
                "name": "root",
                "joint": "heavy",
                "jointData": {},
                "visuals": [mesh_files[i]],
            }
        else:
            entry = {
                "id": i,
                "parent": 0,
                "name": f"joint_{i}",
                "joint": joint_type_dict[joint_info["type"]],
                "jointData": {
                    "axis": {
                        "origin": joint_info["axis_position"].tolist(),
                        "direction": joint_info["axis_direction"].tolist(),
                    },
                    "limit": {},
                },
                "visuals": [mesh_files[i]],
            }
        meta_info.append(entry)
    with open(os.path.join(exp_dir, "joint_info.json"), "w") as f:
        json.dump(meta_info, f, indent=4)


def resolve_iteration(model_path, iteration):
    iteration = int(iteration)
    if iteration == -1:
        return searchForMaxIteration(os.path.join(model_path, "deform"))
    return iteration


def resolve_gt_path(args):
    if args.gt_path:
        return os.path.abspath(args.gt_path)
    return os.path.join(args.source_path, "gt")


def resolve_mesh_path(args, iteration):
    if args.mesh_path:
        return os.path.abspath(args.mesh_path)
    return os.path.join(args.model_path, "train", f"ours_{iteration}", "meshes")


def resolve_save_dir(args, mesh_path):
    if args.save_dir:
        return os.path.abspath(args.save_dir)
    return os.path.dirname(mesh_path)


def ensure_required_meshes(mesh_path, dynamic_ids):
    required_files = [
        os.path.join(mesh_path, "start_-1.ply"),
        os.path.join(mesh_path, "end_-1.ply"),
        os.path.join(mesh_path, "start_0.ply"),
        os.path.join(mesh_path, "end_0.ply"),
    ]
    for dynamic_id in dynamic_ids:
        required_files.extend(
            [
                os.path.join(mesh_path, f"start_{dynamic_id}.ply"),
                os.path.join(mesh_path, f"end_{dynamic_id}.ply"),
            ]
        )

    missing_files = [path for path in required_files if not os.path.exists(path)]
    if missing_files:
        raise FileNotFoundError(
            "Missing mesh files for evaluation:\n" + "\n".join(missing_files)
        )


def resolve_eval_joint_types(args, deform):
    configured_types = [
        joint_type.strip()
        for joint_type in getattr(args, "joint_types", "").split(",")
        if joint_type.strip()
    ]
    if configured_types and configured_types[0] != "s":
        configured_types = ["s"] + configured_types

    if configured_types and len(configured_types) == deform.deform.num_slots:
        deform.deform.joint_types = configured_types
        deform.deform.use_art_type_prior = True
    elif configured_types:
        print(
            "Configured joint_types length does not match num_slots, "
            f"use loaded model types instead: joint_types={configured_types}, "
            f"num_slots={deform.deform.num_slots}"
        )

    return deform.deform.joint_types[1:]


def get_gt_dynamic_mesh_path(gt_path, state, gt_idx, num_gt_dynamic):
    suffix = "" if num_gt_dynamic == 1 else f"_{gt_idx}"
    return os.path.join(gt_path, state, f"{state}_dynamic{suffix}_rotate.ply")


def ensure_gt_dynamic_meshes(gt_path, num_gt_dynamic):
    required_files = []
    for state in ["start", "end"]:
        for gt_idx in range(num_gt_dynamic):
            required_files.append(
                get_gt_dynamic_mesh_path(gt_path, state, gt_idx, num_gt_dynamic)
            )

    missing_files = [path for path in required_files if not os.path.exists(path)]
    if missing_files:
        raise FileNotFoundError(
            "Missing GT dynamic mesh files for matching:\n" + "\n".join(missing_files)
        )


def compute_part_matching_cost(
    gt_path,
    mesh_path,
    pred_dynamic_ids,
    num_gt_dynamic,
    states=("start", "end"),
    n_trials=1,
):
    cost = np.zeros((num_gt_dynamic, len(pred_dynamic_ids)), dtype=np.float64)
    for seed in range(n_trials):
        seed_everything(seed)
        for gt_idx in range(num_gt_dynamic):
            for pred_col, pred_id in enumerate(pred_dynamic_ids):
                pair_cost = 0.0
                for state in states:
                    gt_mesh = get_gt_dynamic_mesh_path(gt_path, state, gt_idx, num_gt_dynamic)
                    pred_mesh = os.path.join(mesh_path, f"{state}_{pred_id}.ply")
                    pair_cost += compute_recon_error(
                        pred_mesh,
                        gt_mesh,
                        n_samples=10000,
                        vis=False,
                    ) * 1000
                cost[gt_idx, pred_col] += pair_cost / len(states)
    return cost / n_trials


def match_parts_by_mesh(
    gt_path,
    mesh_path,
    num_gt_dynamic,
    num_pred_dynamic,
    n_trials=1,
):
    if n_trials < 1:
        raise ValueError(f"match_n_trials must be >= 1, got {n_trials}.")
    if num_gt_dynamic != num_pred_dynamic:
        raise ValueError(
            "The number of GT dynamic parts and predicted dynamic parts must match "
            f"for one-to-one evaluation, got GT={num_gt_dynamic}, pred={num_pred_dynamic}."
        )

    pred_dynamic_ids = list(range(1, 1 + num_pred_dynamic))
    ensure_required_meshes(mesh_path, pred_dynamic_ids)
    ensure_gt_dynamic_meshes(gt_path, num_gt_dynamic)

    cost = compute_part_matching_cost(
        gt_path,
        mesh_path,
        pred_dynamic_ids,
        num_gt_dynamic,
        n_trials=n_trials,
    )
    gt_indices, pred_cols = linear_sum_assignment(cost)
    gt_to_pred_dynamic_id = [None] * num_gt_dynamic
    gt_to_pred_joint_idx = [None] * num_gt_dynamic
    for gt_idx, pred_col in zip(gt_indices, pred_cols):
        pred_id = pred_dynamic_ids[pred_col]
        gt_to_pred_dynamic_id[gt_idx] = pred_id
        gt_to_pred_joint_idx[gt_idx] = pred_id - 1

    if any(pred_id is None for pred_id in gt_to_pred_dynamic_id):
        raise ValueError("Part mesh matching did not assign every GT dynamic part.")

    return gt_to_pred_dynamic_id, gt_to_pred_joint_idx, cost


def evaluate_joints_with_part_assignment(
    pred_joint_list,
    gt_info_list,
    gt_to_pred_joint_idx,
):
    gt_joint_types = [
        "r" if gt_info["type"] == "revolute" else "p" for gt_info in gt_info_list
    ]
    results = []
    for gt_idx, pred_idx in enumerate(gt_to_pred_joint_idx):
        pred_joint = pred_joint_list[pred_idx]
        gt_joint = {key: value for key, value in gt_info_list[gt_idx].items()}
        angle, distance, theta_diff = eval_axis_and_state(
            pred_joint,
            gt_joint,
            gt_joint_types[gt_idx],
        )
        results.append((angle, distance, theta_diff))
    return results


def load_gaussians_for_visualization(dataset, iteration):
    point_cloud_path = os.path.join(
        dataset.model_path,
        "point_cloud",
        f"iteration_{iteration}",
        "point_cloud.ply",
    )
    if not os.path.exists(point_cloud_path):
        raise FileNotFoundError(
            "Cannot visualize mesh segmentation without the Gaussian point cloud:\n"
            f"{point_cloud_path}"
        )

    gaussians = GaussianModel(dataset.sh_degree)
    Scene(dataset, gaussians, load_iteration=iteration)
    return gaussians


def get_axis_visual_position(joint_info, joint_type, center):
    pos = np.asarray(joint_info["axis_position"], dtype=np.float64).copy()
    if joint_type.strip() == "p":
        return np.asarray(center, dtype=np.float64).copy()

    direction = np.asarray(joint_info["axis_direction"], dtype=np.float64)
    return pos + direction * np.dot(direction, np.asarray(center) - pos)


def save_predicted_axis_meshes(
    pred_joint_list,
    pred_joint_types,
    centers,
    mesh_path,
    gt_info_list=None,
):
    for i, joint_info in enumerate(pred_joint_list):
        pos = get_axis_visual_position(joint_info, pred_joint_types[i], centers[i])
        save_axis_mesh(
            joint_info["axis_direction"],
            pos,
            os.path.join(mesh_path, f"axis_{i}_{pred_joint_types[i]}.ply"),
        )

        if gt_info_list is None or i >= len(gt_info_list):
            continue

        gt_joint = gt_info_list[i]
        gt_joint_type = "r" if gt_joint["type"] == "revolute" else "p"
        gt_pos = get_axis_visual_position(gt_joint, gt_joint_type, centers[i])
        save_axis_mesh(
            gt_joint["axis_direction"],
            gt_pos,
            os.path.join(mesh_path, f"gt_axis_{i}_{gt_joint_type}.ply"),
        )


def visualize_mesh_segmentation(mesh_path, pred_joint_types, palette):
    num_d_joints = len(pred_joint_types)
    for state in ["start", "end"]:
        meshes = []
        for mesh_id in range(num_d_joints + 1):
            mesh_id_path = os.path.join(mesh_path, f"{state}_{mesh_id}.ply")
            if not os.path.exists(mesh_id_path):
                print(f"Mesh file not found, skip visualization: {mesh_id_path}")
                continue

            color = palette[mesh_id][None, ...]
            mesh = o3d.io.read_triangle_mesh(mesh_id_path)
            mesh.compute_vertex_normals()
            mesh.vertex_colors = o3d.utility.Vector3dVector(
                color.repeat(len(mesh.vertices), axis=0)
            )
            meshes.append(mesh)

            if mesh_id == 0:
                continue

            joint_idx = mesh_id - 1
            axis_path = os.path.join(
                mesh_path,
                f"axis_{joint_idx}_{pred_joint_types[joint_idx]}.ply",
            )
            if os.path.exists(axis_path):
                axis = o3d.io.read_triangle_mesh(axis_path)
                axis.paint_uniform_color([1, 0, 0])
                meshes.append(axis)

        if meshes:
            o3d.visualization.draw_geometries(
                meshes,
                window_name=f"{state} mesh segmentation",
            )


def render_mesh_segmentation(args, dataset, iteration, deform, pred_joint_types, pred_joint_list, mesh_path):
    gaussians = load_gaussians_for_visualization(dataset, iteration)

    centers = deform.deform.seg_model.center[1:].detach().cpu().numpy()
    gt_trans_path = os.path.join(args.source_path, "gt", "trans.json")
    gt_info_list = read_gt(gt_trans_path) if os.path.exists(gt_trans_path) else None
    save_predicted_axis_meshes(
        pred_joint_list,
        pred_joint_types,
        centers,
        mesh_path,
        gt_info_list=gt_info_list,
    )

    mask = deform.step(gaussians, is_training=False)[0]["mask"]
    num_colors = max(len(pred_joint_types) + 1, int(mask.max().item()) + 1)
    palette = np.array(sns.color_palette("hls", num_colors))
    palette[0] = np.array([0.737, 0.706, 0.663])
    visualize_mesh_segmentation(mesh_path, pred_joint_types, palette)


def evaluate_existing_meshes(args, dataset):
    iteration = resolve_iteration(dataset.model_path, args.iteration)
    gt_path = resolve_gt_path(args)
    mesh_path = resolve_mesh_path(args, iteration)
    save_dir = resolve_save_dir(args, mesh_path)

    if not os.path.isdir(gt_path):
        raise FileNotFoundError(f"Ground-truth path not found: {gt_path}")
    if not os.path.isdir(mesh_path):
        raise FileNotFoundError(f"Mesh path not found: {mesh_path}")

    os.makedirs(save_dir, exist_ok=True)
    print(f"Resolved iteration: {iteration}")
    print(f"GT path: {gt_path}")
    print(f"Mesh path: {mesh_path}")
    print(f"Save dir: {save_dir}")

    deform = DeformModel(dataset)
    loaded = deform.load_weights(dataset.model_path, iteration=iteration)
    if not loaded:
        raise ValueError(f"Failed to load weights from {dataset.model_path}")
    deform.update(30000)

    pred_joint_types = resolve_eval_joint_types(dataset, deform)
    num_d_joints = len(pred_joint_types)
    print(f"Evaluation joint types: {pred_joint_types}")
    pred_joint_list = deform.deform.get_joint_param(pred_joint_types)

    centers = deform.deform.seg_model.center[1:].detach().cpu().numpy()
    for i, joint_info in enumerate(pred_joint_list):
        pos = joint_info["axis_position"]
        if pred_joint_types[i] == "p":
            pos = centers[i]
        else:
            pos += joint_info["axis_direction"] * np.dot(
                joint_info["axis_direction"], centers[i] - pos
            )
        save_axis_mesh(
            joint_info["axis_direction"],
            pos,
            os.path.join(mesh_path, f"axis_{i}_{pred_joint_types[i]}.ply"),
        )

    gt_trans_path = os.path.join(gt_path, "trans.json")
    gt_info_list = None
    if os.path.exists(gt_trans_path):
        gt_info_list = read_gt(gt_trans_path)
        num_gt_dynamic = len(gt_info_list)
    else:
        num_gt_dynamic = num_d_joints

    print("Matching dynamic part meshes")
    dynamic_ids, gt_to_pred_joint_idx, part_matching_cost = match_parts_by_mesh(
        gt_path,
        mesh_path,
        num_gt_dynamic,
        num_d_joints,
        n_trials=args.match_n_trials,
    )
    print("Part matching cost matrix (rows: GT parts, cols: predicted parts):")
    print(part_matching_cost)
    print(f"Part matching GT->pred mesh ids: {dynamic_ids}")

    results_list = []
    if os.path.exists(gt_trans_path):
        results_list = evaluate_joints_with_part_assignment(
            pred_joint_list,
            gt_info_list,
            gt_to_pred_joint_idx,
        )
        print(f"Joint evaluation GT->pred joint indices: {gt_to_pred_joint_idx}")
        print(results_list)
    else:
        print(f"Ground-truth joint file not found, skip axis/state evaluation: {gt_trans_path}")

    pred_joint_meta = [{}] + pred_joint_list
    mesh_files = [
        os.path.relpath(os.path.join(mesh_path, f"start_{i}.ply"), save_dir)
        for i in range(len(pred_joint_meta))
    ]
    export_joint_info_json(pred_joint_meta, mesh_files, save_dir)

    ensure_required_meshes(mesh_path, dynamic_ids)

    output = pd.DataFrame()
    print("Evaluating CD")
    s, d_list, w = eval_CD_2states(gt_path, mesh_path, dynamic_ids, n_trials=args.n_trials)
    mean_s = (s["start"] + s["end"]) / 2
    mean_d_list = [(ds + de) / 2 for ds, de in zip(d_list["start"], d_list["end"])]
    mean_w = (w["start"] + w["end"]) / 2

    output["avg_CD_static"] = [mean_s]
    output["avg_CD_whole"] = [mean_w]
    if mean_d_list:
        output["avg_CD_dynamic"] = [sum(mean_d_list) / len(mean_d_list)]
    else:
        output["avg_CD_dynamic"] = [0.0]

    if results_list:
        output["avg_angle"] = [sum(result[0] for result in results_list) / len(results_list)]
        output["avg_distance"] = [sum(result[1] for result in results_list) / len(results_list)]
        output["avg_theta_diff"] = [sum(result[2] for result in results_list) / len(results_list)]
    else:
        output["avg_angle"] = [np.nan]
        output["avg_distance"] = [np.nan]
        output["avg_theta_diff"] = [np.nan]

    for i, d in enumerate(mean_d_list):
        output[f"CD_dynamic_{i}"] = [d]
        output[f"matched_pred_dynamic_id_{i}"] = [dynamic_ids[i]]
        if results_list:
            output[f"angle_{i}"] = [results_list[i][0]]
            output[f"distance_{i}"] = [results_list[i][1]]
            output[f"theta_diff_{i}"] = [results_list[i][2]]

    for state in ["start", "end"]:
        output[f"{state}_CD_static"] = [s[state]]
        output[f"{state}_CD_whole"] = [w[state]]
        for i, d in enumerate(d_list[state]):
            output[f"{state}_CD_dynamic_{i}"] = [d]

    output["PSNR"] = [-1]
    output["SSIM"] = [-1]
    output["LPIPS"] = [-1]
    output = output.transpose()
    print(output)
    output.to_csv(os.path.join(save_dir, "result.csv"), index=True)
    with open(os.path.join(save_dir, "part_matching.json"), "w") as f:
        json.dump(
            {
                "gt_to_pred_dynamic_id": dynamic_ids,
                "gt_to_pred_joint_idx": gt_to_pred_joint_idx,
                "cost_matrix": part_matching_cost.tolist(),
            },
            f,
            indent=4,
        )
    if args.render_mesh_seg:
        render_mesh_segmentation(
            args,
            dataset,
            iteration,
            deform,
            pred_joint_types,
            pred_joint_list,
            mesh_path,
        )

if __name__ == "__main__":
    parser = ArgumentParser(description="Mesh-only evaluation script parameters")
    model = ModelParams(parser, sentinel=True)
    PipelineParams(parser)
    OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_mesh_seg", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--eval_app", action="store_true")
    parser.add_argument("--mode", default="eval", choices=["render", "eval"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gt_path", default="")
    parser.add_argument("--mesh_path", default="")
    parser.add_argument("--save_dir", default="")
    parser.add_argument("--n_trials", type=int, default=3)
    parser.add_argument("--match_n_trials", type=int, default=1)

    args = get_combined_args(parser)
    args.source_path = f"/data1/wd/ArtGS-data/ArtGS_raw_data/{args.dataset}/{args.subset}/{args.scene_name}"

    print("Evaluating " + args.source_path + " with " + args.model_path)
    safe_state(args.quiet)
    seed_everything(args.seed)

    with torch.no_grad():
        evaluate_existing_meshes(args, model.extract(args))
#python eval1.py --dataset artgs --subset sapien --scene_name storage_45503 --model_path /data1/wd/REArtGS2/weights/weights/storage_45503 --resolution -1 --iteration 30000 --render_mesh_seg --skip_test
