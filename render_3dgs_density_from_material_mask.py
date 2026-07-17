#!/usr/bin/env python3
"""Lift a 2D material segmentation to 3DGS primitives and render material properties.

The pipeline follows the GaussianEditor-style mask-to-Gaussian idea:

1. Render/load the same 3DGS view used by the material segmentation.
2. Project Gaussian centers into that view and sample the 2D material label map.
3. Keep only visible projected Gaussians as segmentation seeds, then propagate
   labels to the remaining Gaussians by nearest-neighbor assignment in 3D.
4. Convert each Gaussian label to a material color and density value.
5. Render material segmentation and mass density from the labeled 3DGS field.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import DeformModel, Scene
from scene.cameras import Camera
from scripts.single_image_density_kernel_query import (
    colormap_rgba,
    load_density_table,
    load_label_entries,
    lookup_density,
    on_white,
    save_colorbar,
)
from utils.general_utils import safe_state


RAW_ROOT = Path("/data1/wd/ArtGS-data/ArtGS_raw_data/artgs/sapien")
FALLBACK_LABEL_COLORS: tuple[tuple[int, int, int], ...] = (
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
)


def _arg_value(argv: list[str], name: str) -> str | None:
    prefix = f"{name}="
    for idx, item in enumerate(argv):
        if item == name and idx + 1 < len(argv):
            return argv[idx + 1]
        if item.startswith(prefix):
            return item[len(prefix) :]
    return None


def _has_arg(argv: list[str], name: str) -> bool:
    prefix = f"{name}="
    return any(item == name or item.startswith(prefix) for item in argv)


def inject_default_model_path_from_scene(argv: list[str]) -> list[str]:
    scene = _arg_value(argv, "--scene")
    if scene and not _has_arg(argv, "--model_path"):
        return ["--model_path", str(ROOT / "weights" / "weights" / scene), *argv]
    return argv


def select_camera(views: list[Any], frame_id: str):
    matches = [view for view in views if str(view.image_name) == str(frame_id)]
    if not matches:
        available = ", ".join(sorted(str(view.image_name) for view in views[:20]))
        raise ValueError(f"frame_id={frame_id} not found; available examples: {available}")
    return matches[0]


def load_render_manifest(scene: str, path: Path | None) -> dict[str, Any]:
    manifest = path or (ROOT / "tmp" / scene / "rendering_rgb.json")
    if manifest.exists():
        return json.loads(manifest.read_text())
    return {
        "scene_name": scene,
        "state": "end",
        "frame_id": "0000",
        "iteration": 30000,
        "model_path": str(ROOT / "weights" / "weights" / scene),
        "source_path": str(RAW_ROOT / scene),
    }


def camera_from_manifest(payload: dict[str, Any], *, frame_id: str, time: float) -> Camera:
    """Build a render camera without requiring the original training dataset."""
    camera = payload.get("camera")
    if not isinstance(camera, dict):
        raise KeyError("render manifest does not contain a camera object")
    width = int(camera["width"])
    height = int(camera["height"])
    fovy = math.radians(float(camera["fovy_degrees"]))
    fovx = 2.0 * math.atan(math.tan(fovy * 0.5) * width / height)
    image = torch.zeros((3, height, width), dtype=torch.float32)
    return Camera(
        colmap_id=0,
        R=np.asarray(camera["R"], dtype=np.float32),
        T=np.asarray(camera["T"], dtype=np.float32),
        FoVx=fovx,
        FoVy=fovy,
        image=image,
        gt_alpha_mask=None,
        image_name=frame_id,
        uid=0,
        data_device="cuda",
        fid=float(time),
    )


def load_material_label_map(mask_path: Path, metadata_path: Path) -> tuple[np.ndarray, list[Any], dict[str, Any]]:
    payload = json.loads(metadata_path.read_text())
    entries = load_label_entries(metadata_path, prefer="classes")

    masks = np.load(mask_path)
    if "label_map" not in masks.files:
        raise KeyError(f"{mask_path} does not contain label_map")
    label_map = np.asarray(masks["label_map"], dtype=np.int32)

    if "foreground" in masks.files:
        foreground = np.asarray(masks["foreground"], dtype=bool)
        if foreground.shape == label_map.shape:
            for entry in entries:
                if entry.is_remaining and entry.label_id is not None and entry.label_id > 0:
                    label_map = label_map.copy()
                    label_map[foreground & (label_map == 0)] = int(entry.label_id)
                    break

    return label_map, entries, payload


def resize_label_map(label_map: np.ndarray, width: int, height: int) -> np.ndarray:
    if label_map.shape == (height, width):
        return label_map
    image = Image.fromarray(label_map.astype(np.uint16))
    image = image.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.int32)


def sample_projected_labels(label_map: np.ndarray, u: np.ndarray, v: np.ndarray, in_bounds: np.ndarray, radius: int) -> np.ndarray:
    height, width = label_map.shape
    labels = np.zeros(len(u), dtype=np.int32)
    idxs = np.where(in_bounds)[0]
    if len(idxs) == 0:
        return labels

    xs = np.rint(u[idxs]).astype(np.int64)
    ys = np.rint(v[idxs]).astype(np.int64)
    xs = np.clip(xs, 0, width - 1)
    ys = np.clip(ys, 0, height - 1)

    if radius <= 0:
        labels[idxs] = label_map[ys, xs]
        return labels

    max_label = int(max(label_map.max(), 0))
    counts = np.zeros((len(idxs), max_label + 1), dtype=np.uint16)
    offsets = [(0, 0)]
    r = int(radius)
    offsets.extend((dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if not (dy == 0 and dx == 0))
    rows = np.arange(len(idxs))
    for dy, dx in offsets:
        xx = np.clip(xs + dx, 0, width - 1)
        yy = np.clip(ys + dy, 0, height - 1)
        vals = label_map[yy, xx].astype(np.int64)
        np.add.at(counts, (rows, vals), 1)
    if counts.shape[1] > 0:
        counts[:, 0] = 0
    chosen = counts.argmax(axis=1).astype(np.int32)
    labels[idxs] = np.where(counts.max(axis=1) > 0, chosen, 0)
    return labels


def project_gaussians(view, xyz: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        pts_cam = xyz @ view.world_view_transform[:3, :3] + view.world_view_transform[3, :3]
        z = pts_cam[:, 2]
        u = pts_cam[:, 0] * float(view.Fx) / torch.clamp(z, min=1e-6) + float(view.Cx)
        v = pts_cam[:, 1] * float(view.Fy) / torch.clamp(z, min=1e-6) + float(view.Cy)
        in_bounds = (
            (z > 0.1)
            & (u >= 0)
            & (u < int(view.image_width))
            & (v >= 0)
            & (v < int(view.image_height))
        )
    return (
        u.detach().cpu().numpy(),
        v.detach().cpu().numpy(),
        z.detach().cpu().numpy(),
        in_bounds.detach().cpu().numpy().astype(bool),
    )


def visible_seed_mask(
    *,
    sampled_labels: np.ndarray,
    in_bounds: np.ndarray,
    radii: np.ndarray,
    depth: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    use_depth_filter: bool,
    depth_abs_tol: float,
    depth_rel_tol: float,
    min_visible_seeds: int,
) -> tuple[np.ndarray, bool]:
    seed = in_bounds & (radii > 0) & (sampled_labels > 0)
    relaxed = False
    if not use_depth_filter:
        return seed, relaxed

    height, width = depth.shape
    xs = np.clip(np.rint(u).astype(np.int64), 0, width - 1)
    ys = np.clip(np.rint(v).astype(np.int64), 0, height - 1)
    sampled_depth = depth[ys, xs]
    depth_ok = np.isfinite(sampled_depth) & (sampled_depth > 0)
    tol = float(depth_abs_tol) + float(depth_rel_tol) * np.maximum(np.abs(sampled_depth), np.abs(z))
    depth_ok &= np.abs(z - sampled_depth) <= tol
    strict = seed & depth_ok
    if int(np.count_nonzero(strict)) >= int(min_visible_seeds):
        return strict, relaxed
    return seed, True


def build_label_density_lookup(entries: list[Any], density_table: dict[str, Any]) -> tuple[np.ndarray, dict[int, dict[str, Any]]]:
    max_label = max([int(entry.label_id or 0) for entry in entries] + [0])
    label_to_density = np.zeros(max_label + 1, dtype=np.float32)
    class_info: dict[int, dict[str, Any]] = {}
    for entry in entries:
        if entry.label_id is None or int(entry.label_id) <= 0:
            continue
        label_id = int(entry.label_id)
        density = lookup_density(entry.name, density_table)
        if label_id >= len(label_to_density):
            grown = np.zeros(label_id + 1, dtype=np.float32)
            grown[: len(label_to_density)] = label_to_density
            label_to_density = grown
        label_to_density[label_id] = float(density.mean)
        class_info[label_id] = {
            "label_id": label_id,
            "name": entry.name,
            "color_rgb": list(entry.color_rgb) if entry.color_rgb else None,
            "density_name": density.name,
            "density_mean_kg_m3": float(density.mean),
            "source_confidence": float(entry.confidence),
            "is_remaining": bool(entry.is_remaining),
        }
    return label_to_density, class_info


def build_label_color_lookup(entries: list[Any]) -> dict[int, tuple[int, int, int]]:
    """Return the display color for each lifted material label.

    Metadata produced by the material segmentation step already stores the
    project-wide material colors. If a legacy metadata file has no color, keep
    the renderer usable with a small deterministic fallback palette.
    """
    label_to_color: dict[int, tuple[int, int, int]] = {}
    fallback_idx = 0
    for entry in entries:
        if entry.label_id is None or int(entry.label_id) <= 0:
            continue
        label_id = int(entry.label_id)
        if entry.color_rgb is not None:
            color = tuple(int(np.clip(v, 0, 255)) for v in entry.color_rgb[:3])
        else:
            color = FALLBACK_LABEL_COLORS[fallback_idx % len(FALLBACK_LABEL_COLORS)]
            fallback_idx += 1
        label_to_color[label_id] = color
    return label_to_color


def assign_gaussian_density(
    *,
    xyz: np.ndarray,
    sampled_labels: np.ndarray,
    seed: np.ndarray,
    label_to_density: np.ndarray,
    propagate_unassigned: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.zeros(len(xyz), dtype=np.int32)
    density = np.full(len(xyz), np.nan, dtype=np.float32)
    valid_label = seed & (sampled_labels >= 0) & (sampled_labels < len(label_to_density))
    labels[valid_label] = sampled_labels[valid_label].astype(np.int32)
    density[valid_label] = label_to_density[labels[valid_label]]
    seed = valid_label & np.isfinite(density) & (density > 0)

    if propagate_unassigned and np.any(seed):
        missing = ~seed
        tree = cKDTree(xyz[seed])
        try:
            _, nn = tree.query(xyz[missing], k=1, workers=-1)
        except TypeError:
            _, nn = tree.query(xyz[missing], k=1)
        seed_labels = labels[seed]
        seed_density = density[seed]
        labels[missing] = seed_labels[nn]
        density[missing] = seed_density[nn]

    return labels, density, seed


def write_augmented_ply(src: Path, dst: Path, labels: np.ndarray, density: np.ndarray) -> None:
    ply = PlyData.read(src)
    vertex = ply["vertex"].data
    if len(vertex) != len(labels):
        raise ValueError(f"PLY vertex count {len(vertex)} != labels {len(labels)}")

    new_dtype = list(vertex.dtype.descr)
    existing = set(vertex.dtype.names or [])
    for name, dtype in [
        ("material_label", "i4"),
        ("density", "f4"),
    ]:
        if name not in existing:
            new_dtype.append((name, dtype))
    out = np.empty(len(vertex), dtype=new_dtype)
    for name in vertex.dtype.names or []:
        out[name] = vertex[name]
    out["material_label"] = labels.astype(np.int32)
    out["density"] = np.nan_to_num(density.astype(np.float32), nan=0.0)

    elements = [PlyElement.describe(out, "vertex")]
    for element in ply.elements:
        if element.name != "vertex":
            elements.append(element)
    PlyData(elements, text=ply.text).write(dst)


def density_render_to_images(
    *,
    view,
    gaussians: GaussianModel,
    pipeline_args,
    d_values: dict[str, torch.Tensor],
    gaussian_density: np.ndarray,
    density_min: float,
    density_max: float,
    colormap: str,
    alpha_threshold: float,
) -> tuple[np.ndarray, np.ndarray, Image.Image, Image.Image]:
    finite_density = np.nan_to_num(gaussian_density.astype(np.float32), nan=float(density_min))
    norm = np.clip((finite_density - float(density_min)) / max(float(density_max) - float(density_min), 1e-6), 0.0, 1.0)
    colors = torch.from_numpy(np.repeat(norm[:, None], 3, axis=1)).float().cuda()
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        rendered = render(
            view,
            gaussians,
            pipeline_args,
            background,
            d_xyz=d_values["d_xyz"],
            d_rot=d_values["d_rotation"],
            override_color=colors,
        )
    accum = rendered["render"][0].detach().float().cpu().numpy()
    alpha = rendered["alpha"].detach().float().squeeze().cpu().numpy()
    valid = alpha > float(alpha_threshold)
    density_norm = np.zeros_like(accum, dtype=np.float32)
    density_norm[valid] = accum[valid] / np.maximum(alpha[valid], 1e-6)
    density_image = density_norm * (float(density_max) - float(density_min)) + float(density_min)
    density_image[~valid] = np.nan
    rgba = colormap_rgba(density_image, valid, cmap=colormap, vmin=density_min, vmax=density_max)
    white = on_white(rgba)
    return density_image.astype(np.float32), alpha.astype(np.float32), rgba, white


def label_alpha_density_to_images(
    *,
    view,
    gaussians: GaussianModel,
    pipeline_args,
    d_values: dict[str, torch.Tensor],
    gaussian_labels: np.ndarray,
    label_to_density: np.ndarray,
    label_map: np.ndarray,
    density_min: float,
    density_max: float,
    colormap: str,
    alpha_threshold: float,
    gate_dilation: int,
) -> tuple[np.ndarray, np.ndarray, Image.Image, Image.Image]:
    """Render density by combining per-material alpha maps.

    Rendering a high-density scalar directly as Gaussian color can create bright
    trails when tiny metal Gaussians have broad projected support. Here each
    material label is rendered as an alpha image and then gated by that label's
    2D segmentation mask in the current view before density is accumulated.
    """
    h, w = int(view.image_height), int(view.image_width)
    density_num = np.zeros((h, w), dtype=np.float32)
    alpha_sum = np.zeros((h, w), dtype=np.float32)
    ones = torch.ones((len(gaussian_labels), 3), dtype=torch.float32, device="cuda")
    background = torch.zeros(3, dtype=torch.float32, device="cuda")

    for label_id in sorted(int(v) for v in np.unique(gaussian_labels) if int(v) > 0):
        if label_id >= len(label_to_density) or label_to_density[label_id] <= 0:
            continue
        gs_mask_np = gaussian_labels == label_id
        if not np.any(gs_mask_np):
            continue
        gs_mask = torch.from_numpy(gs_mask_np).bool().cuda()
        with torch.no_grad():
            rendered = render(
                view,
                gaussians,
                pipeline_args,
                background,
                d_xyz=d_values["d_xyz"],
                d_rot=d_values["d_rotation"],
                override_color=ones,
                vis_mask=gs_mask,
            )
        alpha = rendered["alpha"].detach().float().squeeze().cpu().numpy()

        label_gate = label_map == label_id
        if gate_dilation > 0:
            label_gate = ndimage.binary_dilation(label_gate, iterations=int(gate_dilation))
        alpha = alpha * label_gate.astype(np.float32)
        density_num += alpha * float(label_to_density[label_id])
        alpha_sum += alpha

    valid = alpha_sum > float(alpha_threshold)
    density_image = np.full((h, w), np.nan, dtype=np.float32)
    density_image[valid] = density_num[valid] / np.maximum(alpha_sum[valid], 1e-6)
    rgba = colormap_rgba(density_image, valid, cmap=colormap, vmin=density_min, vmax=density_max)
    white = on_white(rgba)
    return density_image.astype(np.float32), np.clip(alpha_sum, 0.0, 1.0).astype(np.float32), rgba, white


def label_alpha_material_to_images(
    *,
    view,
    gaussians: GaussianModel,
    pipeline_args,
    d_values: dict[str, torch.Tensor],
    gaussian_labels: np.ndarray,
    label_to_color: dict[int, tuple[int, int, int]],
    label_map: np.ndarray,
    alpha_threshold: float,
    gate_dilation: int,
) -> tuple[np.ndarray, Image.Image, Image.Image]:
    """Render a material segmentation image from lifted Gaussian labels.

    This mirrors the density renderer's per-label alpha compositing path, so the
    segmentation panel and density panel share the same 3DGS visibility, object
    resolution, and optional 2D gate used to suppress off-mask splat trails.
    """
    h, w = int(view.image_height), int(view.image_width)
    color_num = np.zeros((h, w, 3), dtype=np.float32)
    alpha_sum = np.zeros((h, w), dtype=np.float32)
    ones = torch.ones((len(gaussian_labels), 3), dtype=torch.float32, device="cuda")
    background = torch.zeros(3, dtype=torch.float32, device="cuda")

    for label_id in sorted(int(v) for v in np.unique(gaussian_labels) if int(v) > 0):
        color = label_to_color.get(label_id)
        if color is None:
            continue
        gs_mask_np = gaussian_labels == label_id
        if not np.any(gs_mask_np):
            continue
        gs_mask = torch.from_numpy(gs_mask_np).bool().cuda()
        with torch.no_grad():
            rendered = render(
                view,
                gaussians,
                pipeline_args,
                background,
                d_xyz=d_values["d_xyz"],
                d_rot=d_values["d_rotation"],
                override_color=ones,
                vis_mask=gs_mask,
            )
        alpha = rendered["alpha"].detach().float().squeeze().cpu().numpy()

        label_gate = label_map == label_id
        if gate_dilation > 0:
            label_gate = ndimage.binary_dilation(label_gate, iterations=int(gate_dilation))
        alpha = alpha * label_gate.astype(np.float32)
        color_num += alpha[..., None] * (np.asarray(color, dtype=np.float32) / 255.0)
        alpha_sum += alpha

    valid = alpha_sum > float(alpha_threshold)
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[valid] = color_num[valid] / np.maximum(alpha_sum[valid, None], 1e-6)
    rgba_arr = np.zeros((h, w, 4), dtype=np.uint8)
    rgba_arr[..., :3] = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    rgba_arr[..., 3] = np.where(valid, 255, 0).astype(np.uint8)
    rgba = Image.fromarray(rgba_arr, mode="RGBA")
    white = on_white(rgba)
    return np.clip(alpha_sum, 0.0, 1.0).astype(np.float32), rgba, white


def build_arg_parser() -> tuple[argparse.ArgumentParser, ModelParams, PipelineParams]:
    parser = argparse.ArgumentParser(description="Render 3DGS mass density by lifting a 2D material mask to Gaussian primitives.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--scene", default=None, help="Scene name; when set, defaults come from tmp/<scene>/rendering_rgb.json.")
    parser.add_argument("--render-manifest", type=Path, default=None, help="Manifest produced by render_gt_view_gaussian.py.")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--state", choices=("start", "end"), default=None)
    parser.add_argument("--frame-id", default=None)
    parser.add_argument("--time", type=float, default=None, help="Optional deformation time, independent of the camera state/view set.")
    parser.add_argument("--mask", type=Path, default=None, help="sam3_masks.npz with label_map.")
    parser.add_argument("--metadata", type=Path, default=None, help="ours_material_segmentation_metadata.json.")
    parser.add_argument("--density-table", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--mask-sample-radius", type=int, default=1, help="Pixel radius for projected Gaussian label voting.")
    parser.add_argument("--depth-abs-tol", type=float, default=0.08, help="Absolute depth tolerance for visible Gaussian seeds.")
    parser.add_argument("--depth-rel-tol", type=float, default=0.08, help="Relative depth tolerance for visible Gaussian seeds.")
    parser.add_argument("--disable-depth-filter", action="store_true", help="Assign labels to projected Gaussians without depth visibility filtering.")
    parser.add_argument("--min-visible-seeds", type=int, default=100, help="If strict depth filtering finds fewer seeds, fall back to projected mask seeds.")
    parser.add_argument("--propagate-unassigned", action=argparse.BooleanOptionalAction, default=True, help="Nearest-neighbor propagate seed labels to all remaining Gaussians.")
    parser.add_argument("--density-min", type=float, default=500.0)
    parser.add_argument("--density-max", type=float, default=8000.0)
    parser.add_argument("--colormap", default="paper")
    parser.add_argument("--alpha-threshold", type=float, default=0.01)
    parser.add_argument("--render-mode", choices=("label-alpha-gated", "scalar-alpha"), default="label-alpha-gated", help="label-alpha-gated suppresses splat trails by rendering per-material alpha maps gated by the 2D mask; scalar-alpha renders normalized density as Gaussian color.")
    parser.add_argument("--gate-dilation", type=int, default=2, help="Pixel dilation for 2D material gates in label-alpha-gated mode.")
    parser.add_argument("--write-augmented-ply", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quiet", action="store_true")
    return parser, model, pipeline


def fill_missing_custom_defaults(args: argparse.Namespace) -> None:
    defaults = {
        "scene": None,
        "render_manifest": None,
        "state": None,
        "frame_id": None,
        "time": None,
        "mask": None,
        "metadata": None,
        "density_table": None,
        "out_dir": None,
        "mask_sample_radius": 1,
        "depth_abs_tol": 0.08,
        "depth_rel_tol": 0.08,
        "disable_depth_filter": False,
        "min_visible_seeds": 100,
        "propagate_unassigned": True,
        "density_min": 500.0,
        "density_max": 8000.0,
        "colormap": "paper",
        "alpha_threshold": 0.01,
        "render_mode": "label-alpha-gated",
        "gate_dilation": 2,
        "write_augmented_ply": True,
        "quiet": False,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)


def main() -> None:
    sys.argv = [sys.argv[0], *inject_default_model_path_from_scene(sys.argv[1:])]
    parser, model, pipeline = build_arg_parser()
    args = get_combined_args(parser)
    fill_missing_custom_defaults(args)

    scene_name = args.scene or args.scene_name
    manifest = load_render_manifest(scene_name, args.render_manifest)
    scene_name = str(manifest.get("scene_name") or scene_name)
    args.scene_name = scene_name
    args.model_path = str(Path(args.model_path).expanduser().resolve())
    args.source_path = str(Path(manifest.get("source_path") or RAW_ROOT / scene_name).expanduser().resolve())

    state = args.state or manifest.get("state") or "end"
    frame_id = str(args.frame_id or manifest.get("frame_id") or "0000")
    iteration = int(args.iteration or manifest.get("iteration") or 30000)
    deformation_time = args.time if args.time is not None else manifest.get("deformation_time")

    tmp_scene = ROOT / "tmp" / scene_name
    mask_path = args.mask or tmp_scene / "sam3_masks.npz"
    metadata_path = args.metadata or tmp_scene / "ours_material_segmentation_metadata.json"
    out_dir = args.out_dir or tmp_scene / "3dgs_density_from_material_mask"
    mass_density_dir = tmp_scene / "mass_density"
    out_dir.mkdir(parents=True, exist_ok=True)
    mass_density_dir.mkdir(parents=True, exist_ok=True)

    safe_state(args.quiet)
    dataset = model.extract(args)
    pipeline_args = pipeline.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    if isinstance(manifest.get("camera"), dict):
        ply_path = Path(dataset.model_path) / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        gaussians.load_ply(str(ply_path))
        view = camera_from_manifest(
            manifest,
            frame_id=frame_id,
            time=float(deformation_time if deformation_time is not None else 0.0),
        )
    else:
        scene = Scene(dataset, gaussians, load_iteration=iteration)
        views = scene.getTrainCameras_start() if state == "start" else scene.getTrainCameras_end()
        view = select_camera(views, frame_id)
    deform = DeformModel(dataset)
    has_deformation = deform.load_weights(dataset.model_path, iteration=iteration)
    if not has_deformation:
        d_values = {"d_xyz": None, "d_rotation": None}
    else:
        deform.update(iteration)
        if deformation_time is None:
            state_idx = 0 if state == "start" else 1
            d_values = deform.step(gaussians, is_training=False)[state_idx]
        else:
            d_xyz, d_rotation = deform.deform.interpolate_single_state(
                gaussians, torch.tensor(float(deformation_time), dtype=torch.float32, device="cuda")
            )
            d_values = {"d_xyz": d_xyz, "d_rotation": d_rotation}

    label_map, entries, metadata_payload = load_material_label_map(mask_path, metadata_path)
    label_map = resize_label_map(label_map, int(view.image_width), int(view.image_height))
    density_table = load_density_table(args.density_table, include_defaults=True)
    label_to_density, class_info = build_label_density_lookup(entries, density_table)
    label_to_color = build_label_color_lookup(entries)

    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        rgb_render = render(
            view,
            gaussians,
            pipeline_args,
            background,
            d_xyz=d_values["d_xyz"],
            d_rot=d_values["d_rotation"],
        )
    depth = rgb_render["depth"].detach().float().squeeze().cpu().numpy()
    radii = rgb_render["radii"].detach().float().cpu().numpy()

    xyz_deformed = gaussians.get_xyz if d_values["d_xyz"] is None else gaussians.get_xyz + d_values["d_xyz"]
    u, v, z, in_bounds = project_gaussians(view, xyz_deformed)
    sampled_labels = sample_projected_labels(label_map, u, v, in_bounds, args.mask_sample_radius)
    seed, depth_relaxed = visible_seed_mask(
        sampled_labels=sampled_labels,
        in_bounds=in_bounds,
        radii=radii,
        depth=depth,
        u=u,
        v=v,
        z=z,
        use_depth_filter=not bool(args.disable_depth_filter),
        depth_abs_tol=args.depth_abs_tol,
        depth_rel_tol=args.depth_rel_tol,
        min_visible_seeds=args.min_visible_seeds,
    )
    if int(np.count_nonzero(seed)) == 0:
        raise RuntimeError("No visible Gaussian seeds were assigned from the material mask.")

    xyz_cpu = xyz_deformed.detach().float().cpu().numpy()
    gaussian_labels, gaussian_density, seed = assign_gaussian_density(
        xyz=xyz_cpu,
        sampled_labels=sampled_labels,
        seed=seed,
        label_to_density=label_to_density,
        propagate_unassigned=args.propagate_unassigned,
    )
    material_alpha_image, material_rgba, material_white = label_alpha_material_to_images(
        view=view,
        gaussians=gaussians,
        pipeline_args=pipeline_args,
        d_values=d_values,
        gaussian_labels=gaussian_labels,
        label_to_color=label_to_color,
        label_map=label_map,
        alpha_threshold=args.alpha_threshold,
        gate_dilation=args.gate_dilation,
    )

    if args.render_mode == "scalar-alpha":
        density_image, alpha_image, rgba, white = density_render_to_images(
            view=view,
            gaussians=gaussians,
            pipeline_args=pipeline_args,
            d_values=d_values,
            gaussian_density=gaussian_density,
            density_min=args.density_min,
            density_max=args.density_max,
            colormap=args.colormap,
            alpha_threshold=args.alpha_threshold,
        )
    else:
        density_image, alpha_image, rgba, white = label_alpha_density_to_images(
            view=view,
            gaussians=gaussians,
            pipeline_args=pipeline_args,
            d_values=d_values,
            gaussian_labels=gaussian_labels,
            label_to_density=label_to_density,
            label_map=label_map,
            density_min=args.density_min,
            density_max=args.density_max,
            colormap=args.colormap,
            alpha_threshold=args.alpha_threshold,
            gate_dilation=args.gate_dilation,
        )

    np.save(out_dir / "gaussian_material_label.npy", gaussian_labels.astype(np.int32))
    np.save(out_dir / "gaussian_density.npy", gaussian_density.astype(np.float32))
    np.save(out_dir / "gaussian_seed_mask.npy", seed.astype(bool))
    np.save(out_dir / "rendered_material_alpha.npy", material_alpha_image.astype(np.float32))
    np.save(out_dir / "rendered_density.npy", density_image.astype(np.float32))
    np.save(out_dir / "rendered_alpha.npy", alpha_image.astype(np.float32))
    material_rgba.save(out_dir / "material_segmentation_3dgs.png")
    material_white.save(out_dir / "material_segmentation_3dgs_white.png")
    rgba.save(out_dir / "mass_density_3dgs.png")
    white.save(out_dir / "mass_density_3dgs_white.png")
    save_colorbar(out_dir / "density_colorbar.png", cmap=args.colormap, vmin=args.density_min, vmax=args.density_max)

    material_rgba.save(tmp_scene / "ours_material_segmentation_3dgs.png")
    material_white.save(tmp_scene / "ours_material_segmentation_3dgs_white.png")
    material_rgba.save(tmp_scene / "ours_material_segmentation_transparent.png")
    material_white.save(tmp_scene / "ours_material_segmentation_white.png")
    rgba.save(mass_density_dir / "ours_mass_density_3dgs.png")
    white.save(mass_density_dir / "ours_mass_density_3dgs_white.png")
    save_colorbar(mass_density_dir / "ours_mass_density_3dgs_colorbar.png", cmap=args.colormap, vmin=args.density_min, vmax=args.density_max)

    ply_out = None
    if args.write_augmented_ply:
        ply_src = Path(dataset.model_path) / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        ply_out = out_dir / "point_cloud_with_material_density.ply"
        write_augmented_ply(ply_src, ply_out, gaussian_labels, gaussian_density)

    label_counts = {
        str(label_id): int(np.count_nonzero(gaussian_labels == label_id))
        for label_id in sorted(int(v) for v in np.unique(gaussian_labels) if int(v) > 0)
    }
    seed_label_counts = {
        str(label_id): int(np.count_nonzero(gaussian_labels[seed] == label_id))
        for label_id in sorted(int(v) for v in np.unique(gaussian_labels[seed]) if int(v) > 0)
    }
    manifest_out = {
        "method": "3dgs_material_mask_lifted_density_render",
        "scene_name": scene_name,
        "state": state,
        "deformation_time": float(deformation_time) if deformation_time is not None else None,
        "frame_id": frame_id,
        "iteration": iteration,
        "inputs": {
            "render_manifest": str((args.render_manifest or (tmp_scene / "rendering_rgb.json")).resolve()),
            "mask": str(mask_path.resolve()),
            "metadata": str(metadata_path.resolve()),
            "model_path": str(Path(dataset.model_path).resolve()),
            "source_path": str(Path(dataset.source_path).resolve()),
            "density_table": str(args.density_table.resolve()) if args.density_table else "built_in_defaults",
        },
        "parameters": {
            "mask_sample_radius": int(args.mask_sample_radius),
            "depth_abs_tol": float(args.depth_abs_tol),
            "depth_rel_tol": float(args.depth_rel_tol),
            "depth_filter_enabled": not bool(args.disable_depth_filter),
            "depth_filter_relaxed": bool(depth_relaxed),
            "propagate_unassigned": bool(args.propagate_unassigned),
            "density_min": float(args.density_min),
            "density_max": float(args.density_max),
            "colormap": args.colormap,
            "alpha_threshold": float(args.alpha_threshold),
            "render_mode": args.render_mode,
            "gate_dilation": int(args.gate_dilation),
        },
        "camera_size": [int(view.image_width), int(view.image_height)],
        "classes": [class_info[k] for k in sorted(class_info)],
        "stats": {
            "num_gaussians": int(len(gaussian_labels)),
            "num_projected_in_bounds": int(np.count_nonzero(in_bounds)),
            "num_render_visible_radii": int(np.count_nonzero(radii > 0)),
            "num_seed_gaussians": int(np.count_nonzero(seed)),
            "num_assigned_gaussians": int(np.count_nonzero(gaussian_labels > 0)),
            "label_counts": label_counts,
            "seed_label_counts": seed_label_counts,
            "render_material_alpha_pixels": int(np.count_nonzero(material_alpha_image > args.alpha_threshold)),
            "render_alpha_pixels": int(np.count_nonzero(alpha_image > args.alpha_threshold)),
        },
        "outputs": {
            "gaussian_material_label": str(out_dir / "gaussian_material_label.npy"),
            "gaussian_density": str(out_dir / "gaussian_density.npy"),
            "gaussian_seed_mask": str(out_dir / "gaussian_seed_mask.npy"),
            "rendered_material_alpha": str(out_dir / "rendered_material_alpha.npy"),
            "material_segmentation_3dgs": str(out_dir / "material_segmentation_3dgs.png"),
            "material_segmentation_3dgs_white": str(out_dir / "material_segmentation_3dgs_white.png"),
            "rendered_density": str(out_dir / "rendered_density.npy"),
            "rendered_alpha": str(out_dir / "rendered_alpha.npy"),
            "mass_density_3dgs": str(out_dir / "mass_density_3dgs.png"),
            "mass_density_3dgs_white": str(out_dir / "mass_density_3dgs_white.png"),
            "density_colorbar": str(out_dir / "density_colorbar.png"),
            "augmented_ply": str(ply_out) if ply_out else None,
            "paper_material_panel_white": str(tmp_scene / "ours_material_segmentation_3dgs_white.png"),
            "paper_panel_white": str(mass_density_dir / "ours_mass_density_3dgs_white.png"),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest_out, indent=2, ensure_ascii=False) + "\n")
    print(out_dir / "manifest.json")


if __name__ == "__main__":
    main()
