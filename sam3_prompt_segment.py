#!/usr/bin/env python3
"""Standalone SAM3 image segmentation with text and/or point prompts.

Examples:
  Text prompt only:
    python scripts/sam3_prompt_segment.py --image image.png --text "metal handle" --output-dir out

  Point prompt only:
    python scripts/sam3_prompt_segment.py --image image.png --pos-point 420,310 --neg-point 300,250 --output-dir out

  Text + point prompt:
    python scripts/sam3_prompt_segment.py --image image.png --text "drawer handle" --pos-point 420,310 --output-dir out

  Semantic coloring for all text-prompted instances:
    python scripts/sam3_prompt_segment.py --image image.png --text "person" --semantic --output-dir out

  Multi-class semantic segmentation with foreground remainder:
    python scripts/sam3_prompt_segment.py --image image.png --text "metal handle, painted table legs" --output-dir out
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_SAM3_ROOT = Path("/data1/wd/sam3")
DEFAULT_CHECKPOINT = Path("/data1/wd/sam3/sam3.1_model/sam3.1_multiplex.pt")
DEFAULT_BPE = Path("/data1/wd/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# High-separation segmentation palette. The first few colors are deliberately
# vivid because they are the most common semantic classes in quick prompt tests.
PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (255, 0, 0),      # red
    (0, 204, 255),    # cyan
    (255, 230, 0),    # yellow
    (0, 255, 90),     # green
    (255, 0, 255),    # magenta
    (0, 80, 255),     # blue
    (255, 128, 0),    # orange
    (145, 0, 255),    # violet
    (0, 255, 220),    # aqua
    (255, 80, 150),   # pink
    (160, 255, 0),    # lime
    (130, 80, 40),    # brown
)


@dataclass
class PointPrompt:
    x: float
    y: float
    label: int


@dataclass
class SemanticClass:
    label_id: int
    name: str
    prompt: Optional[str]
    mask: np.ndarray
    score: float
    box_xyxy: Tuple[float, float, float, float]
    is_remaining: bool = False


def parse_point(value: str, default_label: Optional[int] = None) -> PointPrompt:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError("point must be x,y or x,y,label")
    try:
        x = float(parts[0])
        y = float(parts[1])
        label = default_label if len(parts) == 2 else int(parts[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid point {value!r}") from exc
    if label is None:
        label = 1
    if label not in (0, 1):
        raise argparse.ArgumentTypeError("point label must be 1 foreground or 0 background")
    return PointPrompt(x=x, y=y, label=int(label))


def parse_class_point(value: str, default_label: Optional[int] = None) -> Tuple[str, PointPrompt]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("class point must be class_selector:x,y or class_selector:x,y,label")
    selector, point_text = value.split(":", 1)
    selector = selector.strip()
    if not selector:
        raise argparse.ArgumentTypeError("class point selector cannot be empty")
    return selector, parse_point(point_text, default_label=default_label)


def parse_class_box(value: str) -> Tuple[str, Tuple[float, float, float, float]]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("class box must be class_selector:x1,y1,x2,y2")
    selector, box_text = value.split(":", 1)
    selector = selector.strip()
    parts = [part.strip() for part in box_text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("class box must be class_selector:x1,y1,x2,y2")
    try:
        x1, y1, x2, y2 = [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid class box {value!r}") from exc
    if not selector:
        raise argparse.ArgumentTypeError("class box selector cannot be empty")
    return selector, (x1, y1, x2, y2)


def parse_class_polygon(value: str) -> Tuple[str, List[Tuple[float, float]]]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("class polygon must be class_selector:x1,y1,x2,y2,x3,y3,...")
    selector, polygon_text = value.split(":", 1)
    selector = selector.strip()
    parts = [part.strip() for part in polygon_text.split(",")]
    if len(parts) < 6 or len(parts) % 2 != 0:
        raise argparse.ArgumentTypeError("class polygon must contain at least three x,y points")
    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid class polygon {value!r}") from exc
    if not selector:
        raise argparse.ArgumentTypeError("class polygon selector cannot be empty")
    points = [(values[idx], values[idx + 1]) for idx in range(0, len(values), 2)]
    return selector, points


def parse_class_name(value: str) -> Tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("class name must be class_selector:display_name")
    selector, name = value.split(":", 1)
    selector = selector.strip()
    name = name.strip()
    if not selector or not name:
        raise argparse.ArgumentTypeError("class name selector and display name cannot be empty")
    return selector, name


def parse_class_min_area(value: str) -> Tuple[str, int]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("class min area must be class_selector:pixels")
    selector, area_text = value.split(":", 1)
    selector = selector.strip()
    try:
        area = int(area_text.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid class min area {value!r}") from exc
    if not selector:
        raise argparse.ArgumentTypeError("class min area selector cannot be empty")
    if area < 0:
        raise argparse.ArgumentTypeError("class min area must be non-negative")
    return selector, area


def parse_class_min_component_area(value: str) -> Tuple[str, int]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("class min component area must be class_selector:pixels")
    selector, area_text = value.split(":", 1)
    selector = selector.strip()
    try:
        area = int(area_text.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid class min component area {value!r}") from exc
    if not selector:
        raise argparse.ArgumentTypeError("class min component area selector cannot be empty")
    if area < 0:
        raise argparse.ArgumentTypeError("class min component area must be non-negative")
    return selector, area


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
        if str(getattr(value, "dtype", "")).endswith(("bfloat16", "float16")):
            value = value.float()
        value = value.cpu().numpy()
    return np.asarray(value)


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def split_text_prompts(text: Optional[str]) -> List[str]:
    if not text:
        return []
    return [item.strip() for item in re.split(r"[,;，；]", text) if item.strip()]


def color_for_index(index: int, *, semantic: bool, semantic_color: Optional[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    if semantic and semantic_color is not None:
        return semantic_color
    return PALETTE[(index - 1) % len(PALETTE)]


def ensure_mask_shape(masks: np.ndarray, height: int, width: int) -> np.ndarray:
    masks = np.asarray(masks)
    if masks.size == 0:
        return np.zeros((0, height, width), dtype=bool)
    if masks.ndim == 4:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3:
        raise ValueError(f"unexpected masks shape: {masks.shape}")
    return masks.astype(bool)


def score_array(scores: np.ndarray, count: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.size == count:
        return scores
    out = np.zeros((count,), dtype=np.float32)
    out[: min(count, scores.size)] = scores[: min(count, scores.size)]
    return out


def boxes_array(boxes: np.ndarray, count: int) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim == 1 and boxes.size == 4:
        boxes = boxes[None]
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        return np.zeros((count, 4), dtype=np.float32)
    out = np.zeros((count, 4), dtype=np.float32)
    out[: min(count, boxes.shape[0])] = boxes[: min(count, boxes.shape[0])]
    return out


def add_points_to_grounding_state(state: Dict[str, Any], points: Sequence[PointPrompt], width: int, height: int, device: str) -> None:
    if not points:
        return
    import torch

    if "geometric_prompt" not in state:
        state["geometric_prompt"] = state["_model"]._get_dummy_prompt()
    coords = []
    labels = []
    for point in points:
        x = min(max(point.x / max(1, width), 0.0), 1.0)
        y = min(max(point.y / max(1, height), 0.0), 1.0)
        coords.append([x, y])
        labels.append(point.label)
    point_tensor = torch.tensor(coords, device=device, dtype=torch.float32).view(len(coords), 1, 2)
    label_tensor = torch.tensor(labels, device=device, dtype=torch.long).view(len(labels), 1)
    state["geometric_prompt"].append_points(point_tensor, label_tensor)


def build_model(args: argparse.Namespace):
    if str(args.sam3_root) not in sys.path:
        sys.path.insert(0, str(args.sam3_root))
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(
        bpe_path=str(args.bpe),
        checkpoint_path=str(args.checkpoint),
        load_from_HF=False,
        device=device,
        eval_mode=True,
        enable_inst_interactivity=False,
    )
    processor = Sam3Processor(
        model,
        device=device,
        confidence_threshold=args.confidence_threshold,
    )
    return model, processor, device


def run_grounding_prompt(
    image: Image.Image,
    text: str,
    points: Sequence[PointPrompt],
    model: Any,
    processor: Any,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    autocast_enabled = str(device).startswith("cuda")
    context = torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled)
    with torch.inference_mode(), context:
        state = processor.set_image(image)
        state["_model"] = model
        if points:
            dummy = model.backbone.forward_text([text], device=device)
            state["backbone_out"].update(dummy)
            if "geometric_prompt" not in state:
                state["geometric_prompt"] = model._get_dummy_prompt()
            add_points_to_grounding_state(state, points, image.width, image.height, device)
            state = processor._forward_grounding(state)
        else:
            state = processor.set_text_prompt(prompt=text, state=state)
    masks = ensure_mask_shape(to_numpy(state.get("masks", np.zeros((0, image.height, image.width)))), image.height, image.width)
    scores = score_array(to_numpy(state.get("scores", np.zeros((0,), dtype=np.float32))), masks.shape[0])
    boxes = boxes_array(to_numpy(state.get("boxes", np.zeros((0, 4), dtype=np.float32))), masks.shape[0])
    return masks, scores, boxes


def run_point_prompt(
    image: Image.Image,
    points: Sequence[PointPrompt],
    model: Any,
    processor: Any,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not points:
        raise ValueError("point prompt mode requires at least one point")
    return run_grounding_prompt(image, "visual", points, model, processor, device)


def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    boxes = np.zeros((masks.shape[0], 4), dtype=np.float32)
    for idx, mask in enumerate(masks):
        ys, xs = np.nonzero(mask)
        if xs.size == 0 or ys.size == 0:
            continue
        boxes[idx] = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
    return boxes


def filter_small_masks(masks: np.ndarray, scores: np.ndarray, boxes: np.ndarray, min_area: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if min_area <= 0 or masks.shape[0] == 0:
        return masks, scores, boxes
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    keep = areas >= min_area
    return masks[keep], scores[keep], boxes[keep]


def polygons_to_mask(polygons: Sequence[Sequence[Tuple[float, float]]], width: int, height: int) -> np.ndarray:
    mask_image = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(mask_image)
    for polygon in polygons:
        if len(polygon) >= 3:
            draw.polygon([(float(x), float(y)) for x, y in polygon], fill=1)
    return np.asarray(mask_image, dtype=bool)


def foreground_mask_from_image(image_path: Path, image: Image.Image) -> np.ndarray:
    try:
        source = Image.open(image_path)
        if "A" in source.getbands():
            alpha = np.asarray(source.getchannel("A"))
            if alpha.shape == (image.height, image.width) and int((alpha > 8).sum()) > 0:
                return alpha > 8
    except Exception:
        pass
    arr = np.asarray(image.convert("RGB"), dtype=np.int16)
    patch_h = max(1, image.height // 12)
    patch_w = max(1, image.width // 12)
    corners = np.concatenate(
        [
            arr[:patch_h, :patch_w].reshape(-1, 3),
            arr[:patch_h, -patch_w:].reshape(-1, 3),
            arr[-patch_h:, :patch_w].reshape(-1, 3),
            arr[-patch_h:, -patch_w:].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(corners, axis=0)
    delta = np.linalg.norm(arr - bg[None, None, :], axis=2)
    mask = delta > 18.0
    if int(mask.sum()) < image.width * image.height * 0.005:
        mask = np.any(arr < 245, axis=2)
    return mask


def compose_label_map(masks: np.ndarray, scores: np.ndarray, semantic: bool) -> np.ndarray:
    if masks.shape[0] == 0:
        return np.zeros(masks.shape[1:], dtype=np.uint16)
    order = np.argsort(scores)
    label_map = np.zeros(masks.shape[1:], dtype=np.uint16)
    for rank, idx in enumerate(order, start=1):
        label = 1 if semantic else int(idx) + 1
        label_map[masks[idx]] = label
    return label_map


def compose_multiclass_semantic(
    image: Image.Image,
    image_path: Path,
    text_prompts: Sequence[str],
    points: Sequence[PointPrompt],
    class_points: Optional[Dict[int, Sequence[PointPrompt]]],
    class_keep_boxes: Optional[Dict[int, Sequence[Tuple[float, float, float, float]]]],
    class_fill_polygons: Optional[Dict[int, Sequence[Sequence[Tuple[float, float]]]]],
    class_min_areas: Optional[Dict[int, int]],
    class_min_component_areas: Optional[Dict[int, int]],
    model: Any,
    processor: Any,
    device: str,
    min_area: int,
    include_remaining_foreground: bool,
) -> Tuple[np.ndarray, List[SemanticClass], np.ndarray]:
    label_map = np.zeros((image.height, image.width), dtype=np.uint16)
    occupied = np.zeros((image.height, image.width), dtype=bool)
    classes: List[SemanticClass] = []
    for label_id, prompt in enumerate(text_prompts, start=1):
        class_min_area = (class_min_areas or {}).get(label_id, min_area)
        prompt_points = [*points, *((class_points or {}).get(label_id, []))]
        masks, scores, boxes = run_grounding_prompt(image, prompt, prompt_points, model, processor, device)
        masks, scores, boxes = filter_small_masks(masks, scores, boxes, class_min_area)
        if masks.shape[0] == 0:
            class_mask = np.zeros((image.height, image.width), dtype=bool)
            score = 0.0
        else:
            order = np.argsort(scores)[::-1]
            class_mask = np.zeros((image.height, image.width), dtype=bool)
            for idx in order:
                class_mask |= masks[idx]
            class_mask &= ~occupied
            score = float(np.max(scores)) if scores.size else 0.0
        keep_boxes = (class_keep_boxes or {}).get(label_id, [])
        if keep_boxes:
            keep = np.zeros_like(class_mask)
            for x1, y1, x2, y2 in keep_boxes:
                x_lo = max(0, min(image.width, int(np.floor(min(x1, x2)))))
                x_hi = max(0, min(image.width, int(np.ceil(max(x1, x2)))))
                y_lo = max(0, min(image.height, int(np.floor(min(y1, y2)))))
                y_hi = max(0, min(image.height, int(np.ceil(max(y1, y2)))))
                if x_hi > x_lo and y_hi > y_lo:
                    keep[y_lo:y_hi, x_lo:x_hi] = True
            class_mask &= keep
        fill_polygons = (class_fill_polygons or {}).get(label_id, [])
        if fill_polygons:
            class_mask |= polygons_to_mask(fill_polygons, image.width, image.height)
            class_mask &= ~occupied
        class_min_component_area = (class_min_component_areas or {}).get(label_id, 0)
        class_mask = filter_small_components(class_mask, class_min_component_area)
        if class_min_area > 0 and int(class_mask.sum()) < class_min_area:
            class_mask[:] = False
        occupied |= class_mask
        label_map[class_mask] = label_id
        classes.append(
            SemanticClass(
                label_id=label_id,
                name=prompt,
                prompt=prompt,
                mask=class_mask,
                score=score,
                box_xyxy=mask_to_box(class_mask),
            )
        )
    foreground = foreground_mask_from_image(image_path, image)
    if not include_remaining_foreground:
        return label_map, classes, foreground
    remaining = foreground & ~occupied
    if min_area > 0 and int(remaining.sum()) < min_area:
        remaining[:] = False
    remaining_id = len(text_prompts) + 1
    label_map[remaining] = remaining_id
    classes.append(
        SemanticClass(
            label_id=remaining_id,
            name="remaining foreground",
            prompt=None,
            mask=remaining,
            score=1.0 if int(remaining.sum()) > 0 else 0.0,
            box_xyxy=mask_to_box(remaining),
            is_remaining=True,
        )
    )
    return label_map, classes, foreground


def mask_to_box(mask: np.ndarray) -> Tuple[float, float, float, float]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def connected_components_summary(mask: np.ndarray) -> List[Dict[str, Any]]:
    mask = np.asarray(mask, dtype=bool)
    visited = np.zeros(mask.shape, dtype=bool)
    components: List[Dict[str, Any]] = []
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    for start_y, start_x in zip(ys, xs):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        pixels: List[Tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        py = np.asarray([pixel[0] for pixel in pixels], dtype=np.int32)
        px = np.asarray([pixel[1] for pixel in pixels], dtype=np.int32)
        components.append(
            {
                "area_pixels": int(len(pixels)),
                "box_xyxy": [float(px.min()), float(py.min()), float(px.max() + 1), float(py.max() + 1)],
            }
        )
    components.sort(key=lambda item: int(item["area_pixels"]), reverse=True)
    return components


def filter_small_components(mask: np.ndarray, min_component_area: int) -> np.ndarray:
    if min_component_area <= 0:
        return np.asarray(mask, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    visited = np.zeros(mask.shape, dtype=bool)
    out = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    for start_y, start_x in zip(ys, xs):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        pixels: List[Tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(pixels) >= min_component_area:
            for y, x in pixels:
                out[y, x] = True
    return out


def make_overlay(
    image: Image.Image,
    label_map: np.ndarray,
    instance_count: int,
    semantic: bool,
    semantic_color: Optional[Tuple[int, int, int]],
) -> Image.Image:
    base = np.asarray(image.convert("RGB")).copy()
    overlay = base.copy()
    labels = [1] if semantic and instance_count else list(range(1, instance_count + 1))
    for label in labels:
        mask = label_map == label
        if not mask.any():
            continue
        color = np.asarray(color_for_index(label, semantic=semantic, semantic_color=semantic_color), dtype=np.uint8)
        overlay[mask] = (overlay[mask].astype(np.float32) * 0.35 + color.astype(np.float32) * 0.65).astype(np.uint8)
    return Image.fromarray(overlay)


def make_mask_preview(
    label_map: np.ndarray,
    instance_count: int,
    semantic: bool,
    semantic_color: Optional[Tuple[int, int, int]],
) -> Image.Image:
    preview = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    labels = [1] if semantic and instance_count else list(range(1, instance_count + 1))
    for label in labels:
        preview[label_map == label] = color_for_index(label, semantic=semantic, semantic_color=semantic_color)
    return Image.fromarray(preview)


def make_semantic_overlay(image: Image.Image, label_map: np.ndarray, classes: Sequence[SemanticClass]) -> Image.Image:
    base = np.asarray(image.convert("RGB")).copy()
    overlay = base.copy()
    for cls in classes:
        mask = label_map == cls.label_id
        if not mask.any():
            continue
        color = np.asarray(PALETTE[(cls.label_id - 1) % len(PALETTE)], dtype=np.uint8)
        overlay[mask] = (overlay[mask].astype(np.float32) * 0.35 + color.astype(np.float32) * 0.65).astype(np.uint8)
    return Image.fromarray(overlay)


def make_semantic_mask_preview(label_map: np.ndarray, classes: Sequence[SemanticClass]) -> Image.Image:
    preview = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for cls in classes:
        preview[label_map == cls.label_id] = PALETTE[(cls.label_id - 1) % len(PALETTE)]
    return Image.fromarray(preview)


def draw_points(image: Image.Image, points: Sequence[PointPrompt]) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for point in points:
        color = (20, 180, 40) if point.label == 1 else (220, 45, 45)
        radius = 7
        x, y = point.x, point.y
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(255, 255, 255), width=2)
    return out


def add_title(image: Image.Image, title: str) -> Image.Image:
    font = load_font(26)
    width, height = image.size
    title_h = 44
    canvas = Image.new("RGB", (width, height + title_h), "white")
    canvas.paste(image.convert("RGB"), (0, title_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title, fill=(0, 0, 0), font=font)
    return canvas


def write_summary(
    image: Image.Image,
    overlay: Image.Image,
    mask_preview: Image.Image,
    points: Sequence[PointPrompt],
    output_path: Path,
    title: str,
) -> None:
    panel_w = 640
    panel_h = int(round(panel_w * image.height / max(1, image.width)))
    panels = [
        add_title(draw_points(image.resize((panel_w, panel_h), Image.Resampling.LANCZOS), scaled_points(points, image.size, (panel_w, panel_h))), "Input + points"),
        add_title(overlay.resize((panel_w, panel_h), Image.Resampling.LANCZOS), "SAM3 overlay"),
        add_title(mask_preview.resize((panel_w, panel_h), Image.Resampling.NEAREST), "Label map"),
    ]
    pad = 20
    header_h = 54
    canvas = Image.new("RGB", (panel_w * 3 + pad * 4, panel_h + 44 + header_h + pad * 2), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 12), title, fill=(0, 0, 0), font=load_font(32))
    x = pad
    for panel in panels:
        canvas.paste(panel, (x, header_h + pad))
        x += panel_w + pad
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> List[str]:
    words = text.replace("_", " ").split()
    if not words:
        return [text]
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def write_semantic_summary(
    image: Image.Image,
    overlay: Image.Image,
    mask_preview: Image.Image,
    classes: Sequence[SemanticClass],
    points: Sequence[PointPrompt],
    output_path: Path,
) -> None:
    panel_w = 560
    panel_h = int(round(panel_w * image.height / max(1, image.width)))
    panels = [
        add_title(draw_points(image.resize((panel_w, panel_h), Image.Resampling.LANCZOS), scaled_points(points, image.size, (panel_w, panel_h))), "Input + points"),
        add_title(overlay.resize((panel_w, panel_h), Image.Resampling.LANCZOS), "SAM3 semantic overlay"),
        add_title(mask_preview.resize((panel_w, panel_h), Image.Resampling.NEAREST), "Semantic label map"),
    ]
    pad = 20
    header_h = 58
    legend_w = 520
    row_h = 82
    legend_h = 48 + max(1, len(classes)) * row_h
    canvas_w = panel_w * 3 + legend_w + pad * 5
    canvas_h = max(panel_h + 44 + header_h + pad * 2, header_h + pad + legend_h + pad)
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 12), "SAM3 multi-class semantic segmentation", fill=(0, 0, 0), font=load_font(32))
    x = pad
    for panel in panels:
        canvas.paste(panel, (x, header_h + pad))
        x += panel_w + pad
    legend_x = x
    y = header_h + pad
    title_font = load_font(28)
    font = load_font(20)
    small_font = load_font(18)
    draw.text((legend_x, y), "Semantic Classes", fill=(0, 0, 0), font=title_font)
    y += 44
    for cls in classes:
        color = PALETTE[(cls.label_id - 1) % len(PALETTE)]
        draw.rectangle((legend_x, y + 7, legend_x + 38, y + 45), fill=color, outline=(0, 0, 0))
        suffix = "remaining" if cls.is_remaining else f"prompt: {cls.prompt}"
        label = f"{cls.label_id}. {cls.name} | {suffix} | pixels {int(cls.mask.sum())}"
        lines = wrap_text(draw, label, font, legend_w - 54)[:3]
        for line_no, line in enumerate(lines):
            draw.text((legend_x + 52, y + line_no * 23), line, fill=(0, 0, 0), font=font if line_no == 0 else small_font)
        y += max(row_h, len(lines) * 23 + 16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def scaled_points(points: Sequence[PointPrompt], old_size: Tuple[int, int], new_size: Tuple[int, int]) -> List[PointPrompt]:
    old_w, old_h = old_size
    new_w, new_h = new_size
    sx = new_w / max(1, old_w)
    sy = new_h / max(1, old_h)
    return [PointPrompt(point.x * sx, point.y * sy, point.label) for point in points]


def write_outputs(
    args: argparse.Namespace,
    image: Image.Image,
    image_path: Path,
    masks: np.ndarray,
    scores: np.ndarray,
    boxes: np.ndarray,
    points: Sequence[PointPrompt],
    mode: str,
    class_names: Optional[Dict[int, str]] = None,
    output_prefix: str = "sam3",
) -> Dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    semantic_color = PALETTE[0]
    label_map = compose_label_map(masks, scores, args.semantic)
    overlay = make_overlay(image, label_map, masks.shape[0], args.semantic, semantic_color)
    mask_preview = make_mask_preview(label_map, masks.shape[0], args.semantic, semantic_color)

    overlay_path = args.output_dir / f"{output_prefix}_overlay.png"
    label_map_path = args.output_dir / f"{output_prefix}_label_map.png"
    summary_path = args.output_dir / f"{output_prefix}_prompt_segmentation_summary.png"
    masks_npz_path = args.output_dir / f"{output_prefix}_masks.npz"
    metadata_path = args.output_dir / f"{output_prefix}_prompt_segmentation.json"

    overlay.save(overlay_path)
    mask_preview.save(label_map_path)
    write_summary(
        image,
        overlay,
        mask_preview,
        points,
        summary_path,
        title=f"SAM3 prompt segmentation ({mode})",
    )
    records = []
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1) if masks.shape[0] else np.zeros((0,), dtype=np.int64)
    for idx in range(masks.shape[0]):
        label_id = 1 if args.semantic else idx + 1
        records.append(
            {
                "instance_index": idx + 1,
                "label_id": int(label_id),
                "score": float(scores[idx]) if scores.size > idx and math.isfinite(float(scores[idx])) else 0.0,
                "area_pixels": int(areas[idx]),
                "box_xyxy": [float(v) for v in boxes[idx].tolist()],
                "color_rgb": list(color_for_index(label_id, semantic=args.semantic, semantic_color=semantic_color)),
            }
        )
    class_records = []
    positive_labels = sorted(int(label) for label in np.unique(label_map) if int(label) > 0)
    for label_id in positive_labels:
        class_mask = label_map == label_id
        prompt = args.text if args.text else None
        if class_names and label_id in class_names:
            name = class_names[label_id]
        elif args.semantic and prompt:
            name = prompt
        elif prompt:
            name = f"{prompt}_{label_id}"
        else:
            name = f"point_mask_{label_id}"
        matching_scores = [
            float(record["score"])
            for record in records
            if int(record.get("label_id") or 0) == label_id and math.isfinite(float(record.get("score") or 0.0))
        ]
        components = connected_components_summary(class_mask)
        class_records.append(
            {
                "label_id": int(label_id),
                "name": str(name),
                "prompt": prompt,
                "is_remaining": False,
                "score": float(max(matching_scores)) if matching_scores else 0.0,
                "area_pixels": int(class_mask.sum()),
                "box_xyxy": [float(v) for v in mask_to_box(class_mask)],
                "component_count": len(components),
                "components": components,
                "color_rgb": list(color_for_index(label_id, semantic=args.semantic, semantic_color=semantic_color)),
            }
        )
    np.savez_compressed(
        masks_npz_path,
        masks=masks.astype(np.bool_),
        scores=scores.astype(np.float32),
        boxes_xyxy=boxes.astype(np.float32),
        label_map=label_map.astype(np.uint16),
        label_names=np.asarray([cls["name"] for cls in class_records]),
        label_ids=np.asarray([cls["label_id"] for cls in class_records], dtype=np.uint16),
        prompt_texts=np.asarray([cls.get("prompt") or "" for cls in class_records]),
        semantic=np.asarray([args.semantic], dtype=np.bool_),
    )
    metadata = {
        "image": str(image_path),
        "mode": mode,
        "text_prompt": args.text,
        "semantic": bool(args.semantic),
        "point_prompts": [{"x": p.x, "y": p.y, "label": p.label} for p in points],
        "instance_count": int(masks.shape[0]),
        "class_count": int(len(class_records)),
        "outputs": {
            "overlay": str(overlay_path),
            "label_map": str(label_map_path),
            "summary": str(summary_path),
            "masks_npz": str(masks_npz_path),
        },
        "instances": records,
        "classes": class_records,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def write_semantic_outputs(
    args: argparse.Namespace,
    image: Image.Image,
    image_path: Path,
    label_map: np.ndarray,
    classes: Sequence[SemanticClass],
    foreground: np.ndarray,
    points: Sequence[PointPrompt],
    text_prompts: Sequence[str],
    output_prefix: str = "sam3",
) -> Dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay = make_semantic_overlay(image, label_map, classes)
    mask_preview = make_semantic_mask_preview(label_map, classes)

    overlay_path = args.output_dir / f"{output_prefix}_overlay.png"
    label_map_path = args.output_dir / f"{output_prefix}_label_map.png"
    summary_path = args.output_dir / f"{output_prefix}_prompt_segmentation_summary.png"
    masks_npz_path = args.output_dir / f"{output_prefix}_masks.npz"
    metadata_path = args.output_dir / f"{output_prefix}_prompt_segmentation.json"

    overlay.save(overlay_path)
    mask_preview.save(label_map_path)
    write_semantic_summary(image, overlay, mask_preview, classes, points, summary_path)
    class_masks = np.stack([cls.mask for cls in classes], axis=0) if classes else np.zeros((0, image.height, image.width), dtype=bool)
    np.savez_compressed(
        masks_npz_path,
        masks=class_masks.astype(np.bool_),
        label_map=label_map.astype(np.uint16),
        label_names=np.asarray([cls.name for cls in classes]),
        label_ids=np.asarray([cls.label_id for cls in classes], dtype=np.uint16),
        prompt_texts=np.asarray([cls.prompt or "" for cls in classes]),
        semantic=np.asarray([True], dtype=np.bool_),
        foreground=foreground.astype(np.bool_),
    )

    class_records = []
    for cls in classes:
        color = PALETTE[(cls.label_id - 1) % len(PALETTE)]
        components = connected_components_summary(cls.mask)
        class_records.append(
            {
                "label_id": int(cls.label_id),
                "name": cls.name,
                "prompt": cls.prompt,
                "is_remaining": bool(cls.is_remaining),
                "score": float(cls.score) if math.isfinite(float(cls.score)) else 0.0,
                "area_pixels": int(cls.mask.sum()),
                "box_xyxy": [float(v) for v in cls.box_xyxy],
                "component_count": len(components),
                "components": components,
                "color_rgb": list(color),
            }
        )
    metadata = {
        "image": str(image_path),
        "mode": "multi-class-semantic",
        "text_prompt": args.text,
        "text_prompts": list(text_prompts),
        "semantic": True,
        "multi_class_semantic": True,
        "include_remaining_foreground": not args.no_remaining_foreground,
        "remaining_class": "remaining foreground" if not args.no_remaining_foreground else None,
        "class_min_areas": {
            str(idx): int(getattr(args, "class_min_area_values", {}).get(idx, args.min_area))
            for idx in range(1, len(text_prompts) + 1)
        },
        "class_min_component_areas": {
            str(idx): int(getattr(args, "class_min_component_area_values", {}).get(idx, 0))
            for idx in range(1, len(text_prompts) + 1)
        },
        "class_fill_polygon_counts": {
            str(idx): len(getattr(args, "class_fill_polygon_values", {}).get(idx, []))
            for idx in range(1, len(text_prompts) + 1)
        },
        "point_prompts": [{"x": p.x, "y": p.y, "label": p.label} for p in points],
        "class_count": int(len(classes)),
        "foreground_pixel_count": int(foreground.sum()),
        "outputs": {
            "overlay": str(overlay_path),
            "label_map": str(label_map_path),
            "summary": str(summary_path),
            "masks_npz": str(masks_npz_path),
        },
        "classes": class_records,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def collect_points(args: argparse.Namespace) -> List[PointPrompt]:
    points: List[PointPrompt] = []
    for value in args.point or []:
        points.append(parse_point(value))
    for value in args.pos_point or []:
        points.append(parse_point(value, default_label=1))
    for value in args.neg_point or []:
        points.append(parse_point(value, default_label=0))
    return points


def collect_class_points(args: argparse.Namespace, text_prompts: Sequence[str]) -> Dict[int, List[PointPrompt]]:
    by_label: Dict[int, List[PointPrompt]] = {}

    for value in args.class_point or []:
        selector, point = parse_class_point(value)
        by_label.setdefault(selector_to_label(selector, text_prompts), []).append(point)
    for value in args.class_pos_point or []:
        selector, point = parse_class_point(value, default_label=1)
        by_label.setdefault(selector_to_label(selector, text_prompts), []).append(point)
    for value in args.class_neg_point or []:
        selector, point = parse_class_point(value, default_label=0)
        by_label.setdefault(selector_to_label(selector, text_prompts), []).append(point)
    return by_label


def selector_to_label(selector: str, text_prompts: Sequence[str]) -> int:
    prompt_to_label = {prompt.lower(): idx for idx, prompt in enumerate(text_prompts, start=1)}
    key = selector.strip().lower()
    if key.isdigit():
        label = int(key)
        if 1 <= label <= len(text_prompts):
            return label
    if key in prompt_to_label:
        return prompt_to_label[key]
    matches = [idx for idx, prompt in enumerate(text_prompts, start=1) if key and key in prompt.lower()]
    if len(matches) == 1:
        return matches[0]
    raise argparse.ArgumentTypeError(f"class selector {selector!r} does not match exactly one prompt")


def collect_class_keep_boxes(args: argparse.Namespace, text_prompts: Sequence[str]) -> Dict[int, List[Tuple[float, float, float, float]]]:
    by_label: Dict[int, List[Tuple[float, float, float, float]]] = {}
    for value in args.class_keep_box or []:
        selector, box = parse_class_box(value)
        by_label.setdefault(selector_to_label(selector, text_prompts), []).append(box)
    return by_label


def collect_class_fill_polygons(args: argparse.Namespace, text_prompts: Sequence[str]) -> Dict[int, List[List[Tuple[float, float]]]]:
    by_label: Dict[int, List[List[Tuple[float, float]]]] = {}
    for value in args.class_fill_polygon or []:
        selector, polygon = parse_class_polygon(value)
        by_label.setdefault(selector_to_label(selector, text_prompts), []).append(polygon)
    return by_label


def collect_class_min_areas(args: argparse.Namespace, text_prompts: Sequence[str]) -> Dict[int, int]:
    by_label: Dict[int, int] = {}
    for value in args.class_min_area or []:
        selector, area = parse_class_min_area(value)
        by_label[selector_to_label(selector, text_prompts)] = area
    return by_label


def collect_class_min_component_areas(args: argparse.Namespace, text_prompts: Sequence[str]) -> Dict[int, int]:
    by_label: Dict[int, int] = {}
    for value in args.class_min_component_area or []:
        selector, area = parse_class_min_component_area(value)
        by_label[selector_to_label(selector, text_prompts)] = area
    return by_label


def collect_class_names(args: argparse.Namespace, text_prompts: Sequence[str]) -> Dict[int, str]:
    names: Dict[int, str] = {}
    for value in args.class_name or []:
        selector, name = parse_class_name(value)
        names[selector_to_label(selector, text_prompts)] = name
    return names


def discover_images(image_dir: Path) -> List[Path]:
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def segment_one_image(
    args: argparse.Namespace,
    image_path: Path,
    model: Any,
    processor: Any,
    device: str,
    points: Sequence[PointPrompt],
    class_points: Dict[int, Sequence[PointPrompt]],
    class_keep_boxes: Dict[int, Sequence[Tuple[float, float, float, float]]],
    class_fill_polygons: Dict[int, Sequence[Sequence[Tuple[float, float]]]],
    class_min_areas: Dict[int, int],
    class_min_component_areas: Dict[int, int],
    class_names: Dict[int, str],
    text_prompts: Sequence[str],
    output_prefix: str = "sam3",
) -> Dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    display_points = [*points, *[point for pts in class_points.values() for point in pts]]
    if len(text_prompts) > 1:
        label_map, classes, foreground = compose_multiclass_semantic(
            image,
            image_path,
            text_prompts,
            points,
            class_points,
            class_keep_boxes,
            class_fill_polygons,
            class_min_areas,
            class_min_component_areas,
            model,
            processor,
            device,
            args.min_area,
            not args.no_remaining_foreground,
        )
        for cls in classes:
            if cls.label_id in class_names:
                cls.name = class_names[cls.label_id]
        return write_semantic_outputs(args, image, image_path, label_map, classes, foreground, display_points, text_prompts, output_prefix=output_prefix)
    if args.text:
        mode = "text+point" if points else "text"
        masks, scores, boxes = run_grounding_prompt(image, args.text, points, model, processor, device)
    else:
        mode = "point"
        masks, scores, boxes = run_point_prompt(image, points, model, processor, device)
    masks, scores, boxes = filter_small_masks(masks, scores, boxes, args.min_area)
    return write_outputs(args, image, image_path, masks, scores, boxes, points, mode, class_names=class_names, output_prefix=output_prefix)


def run_image_dir_batch(
    args: argparse.Namespace,
    model: Any,
    processor: Any,
    device: str,
    points: Sequence[PointPrompt],
    class_points: Dict[int, Sequence[PointPrompt]],
    class_keep_boxes: Dict[int, Sequence[Tuple[float, float, float, float]]],
    class_fill_polygons: Dict[int, Sequence[Sequence[Tuple[float, float]]]],
    class_min_areas: Dict[int, int],
    class_min_component_areas: Dict[int, int],
    class_names: Dict[int, str],
    text_prompts: Sequence[str],
) -> int:
    frames = discover_images(args.image_dir)
    if not frames:
        raise FileNotFoundError(f"no images found in {args.image_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for index, image_path in enumerate(frames, start=1):
        prefix = image_path.stem
        try:
            metadata = segment_one_image(args, image_path, model, processor, device, points, class_points, class_keep_boxes, class_fill_polygons, class_min_areas, class_min_component_areas, class_names, text_prompts, output_prefix=prefix)
            if metadata.get("multi_class_semantic"):
                count = metadata.get("class_count", 0)
                areas = {item["name"]: item["area_pixels"] for item in metadata.get("classes", [])}
            else:
                count = metadata.get("instance_count", 0)
                areas = {f"instance_{item['instance_index']}": item["area_pixels"] for item in metadata.get("instances", [])}
            records.append(
                {
                    "image": str(image_path),
                    "status": "ok",
                    "count": int(count),
                    "areas": areas,
                    "overlay": metadata["outputs"]["overlay"],
                    "label_map": metadata["outputs"]["label_map"],
                    "masks_npz": metadata["outputs"]["masks_npz"],
                    "metadata": str(args.output_dir / f"{prefix}_prompt_segmentation.json"),
                    "error": "",
                }
            )
            print(f"[{index}/{len(frames)}] ok {image_path.name} count={count}", flush=True)
        except Exception as exc:  # noqa: BLE001
            records.append(
                {
                    "image": str(image_path),
                    "status": "failed",
                    "count": 0,
                    "areas": {},
                    "overlay": "",
                    "label_map": "",
                    "masks_npz": "",
                    "metadata": "",
                    "error": str(exc),
                }
            )
            print(f"[{index}/{len(frames)}] failed {image_path.name}: {exc}", flush=True)
            if not args.keep_going:
                break
    summary = {
        "image_dir": str(args.image_dir),
        "output_dir": str(args.output_dir),
        "text_prompt": args.text,
        "text_prompts": list(text_prompts),
        "semantic_batch": len(text_prompts) > 1,
        "total": len(frames),
        "ok": sum(1 for item in records if item["status"] == "ok"),
        "failed": sum(1 for item in records if item["status"] != "ok"),
        "records": records,
    }
    summary_path = args.output_dir / "batch_summary.json"
    csv_path = args.output_dir / "batch_summary.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "status", "count", "overlay", "label_map", "masks_npz", "metadata", "error"])
        writer.writeheader()
        for item in records:
            writer.writerow({key: item.get(key, "") for key in writer.fieldnames})
    print(json.dumps({"status": "ok", "summary": str(summary_path), "csv": str(csv_path), "ok": summary["ok"], "failed": summary["failed"]}, indent=2))
    return 0 if summary["failed"] == 0 else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone SAM3 image segmentation with text and/or point prompts.")
    parser.add_argument("--image", type=Path, default=None, help="Input image path.")
    parser.add_argument("--image-dir", type=Path, default=None, help="Input image directory for batch propagation.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for overlay, label map, masks npz, and metadata.")
    parser.add_argument("--text", default=None, help="Text prompt for open-vocabulary SAM3 segmentation. Comma-separated prompts create separate semantic classes plus a remaining-foreground class.")
    parser.add_argument("--point", action="append", default=[], help="Point prompt as x,y,label where label is 1 foreground or 0 background. Can be repeated.")
    parser.add_argument("--pos-point", action="append", default=[], help="Foreground point prompt as x,y. Can be repeated.")
    parser.add_argument("--neg-point", action="append", default=[], help="Background point prompt as x,y. Can be repeated.")
    parser.add_argument("--class-point", action="append", default=[], help="Class-specific point as class_selector:x,y,label. Selector can be a 1-based class id or a unique prompt substring.")
    parser.add_argument("--class-pos-point", action="append", default=[], help="Class-specific foreground point as class_selector:x,y. Can be repeated.")
    parser.add_argument("--class-neg-point", action="append", default=[], help="Class-specific background point as class_selector:x,y. Can be repeated.")
    parser.add_argument("--class-keep-box", action="append", default=[], help="Post-filter one class mask to one or more boxes: class_selector:x1,y1,x2,y2. Useful for tiny repeated hardware.")
    parser.add_argument("--class-fill-polygon", action="append", default=[], help="Add a filled polygon to one class mask: class_selector:x1,y1,x2,y2,x3,y3,...")
    parser.add_argument("--class-min-area", action="append", default=[], help="Class-specific min-area filter as class_selector:pixels. Overrides --min-area for that semantic class.")
    parser.add_argument("--class-min-component-area", action="append", default=[], help="Drop connected components smaller than pixels for one class: class_selector:pixels.")
    parser.add_argument("--class-name", action="append", default=[], help="Rename a semantic class in outputs without changing the grounding prompt: class_selector:display_name.")
    parser.add_argument("--semantic", action="store_true", help="For a single text prompt, merge all detected instances into one semantic class. Multiple comma-separated prompts are semantic classes automatically.")
    parser.add_argument("--confidence-threshold", type=float, default=0.35, help="SAM3 text grounding confidence threshold.")
    parser.add_argument("--min-area", type=int, default=0, help="Drop masks smaller than this many pixels.")
    parser.add_argument("--no-remaining-foreground", action="store_true", help="In multi-class semantic mode, do not add the automatic remaining-foreground class.")
    parser.add_argument("--sam3-root", type=Path, default=DEFAULT_SAM3_ROOT, help="Local SAM3 repository root.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="SAM3/SAM3.1 checkpoint path.")
    parser.add_argument("--bpe", type=Path, default=DEFAULT_BPE, help="BPE vocabulary path used by SAM3 text encoder.")
    parser.add_argument("--device", default="auto", help="cuda, cpu, or auto.")
    parser.add_argument("--keep-going", action="store_true", help="In --image-dir mode, continue processing after a frame fails.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if bool(args.image) == bool(args.image_dir):
        parser.error("provide exactly one of --image or --image-dir")
    if args.image and not args.image.exists():
        parser.error(f"--image not found: {args.image}")
    if args.image_dir and not args.image_dir.exists():
        parser.error(f"--image-dir not found: {args.image_dir}")
    if args.image_dir and not args.image_dir.is_dir():
        parser.error(f"--image-dir is not a directory: {args.image_dir}")
    if not args.sam3_root.exists():
        parser.error(f"--sam3-root not found: {args.sam3_root}")
    if not args.checkpoint.exists():
        parser.error(f"--checkpoint not found: {args.checkpoint}")
    if not args.bpe.exists():
        parser.error(f"--bpe not found: {args.bpe}")
    points = collect_points(args)
    text_prompts = split_text_prompts(args.text)
    class_points = collect_class_points(args, text_prompts)
    class_keep_boxes = collect_class_keep_boxes(args, text_prompts)
    class_fill_polygons = collect_class_fill_polygons(args, text_prompts)
    args.class_fill_polygon_values = class_fill_polygons
    class_min_areas = collect_class_min_areas(args, text_prompts)
    args.class_min_area_values = class_min_areas
    class_min_component_areas = collect_class_min_component_areas(args, text_prompts)
    args.class_min_component_area_values = class_min_component_areas
    class_names = collect_class_names(args, text_prompts)
    if not args.text and not points:
        parser.error("provide --text, point prompts, or both")
    if args.semantic and not args.text:
        parser.error("--semantic is only meaningful with --text")

    model, processor, device = build_model(args)
    if args.image_dir:
        return run_image_dir_batch(args, model, processor, device, points, class_points, class_keep_boxes, class_fill_polygons, class_min_areas, class_min_component_areas, class_names, text_prompts)
    assert args.image is not None
    if len(text_prompts) > 1:
        metadata = segment_one_image(args, args.image, model, processor, device, points, class_points, class_keep_boxes, class_fill_polygons, class_min_areas, class_min_component_areas, class_names, text_prompts, output_prefix="sam3")
        print(json.dumps({"status": "ok", "class_count": metadata["class_count"], "outputs": metadata["outputs"]}, indent=2))
        return 0
    metadata = segment_one_image(args, args.image, model, processor, device, points, class_points, class_keep_boxes, class_fill_polygons, class_min_areas, class_min_component_areas, class_names, text_prompts, output_prefix="sam3")
    print(json.dumps({"status": "ok", "instance_count": metadata["instance_count"], "outputs": metadata["outputs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
