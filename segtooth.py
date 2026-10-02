# SPDX-License-Identifier: MIT
# ============================================================================
# SegmentAnyTooth Batch Processing Script
#
# Copyright (c) 2025 Khoa D. Nguyen (Original Code)
#
# This script processes a dataset of intraoral images, generates tooth
# segmentations and bounding boxes, and saves croptooth-style crops in a
# structured format. Its data-preparation logic mirrors tmp_process.py.
# ============================================================================

"""Prepare croptooth-style outputs plus tooth_bbox JSON for IntraoralAlignmentDataset.

Expected/updated output under each <date>_process folder:
  single_tooth/<case_id>/<view>/<fdi>.png
  sextant/<case_id>/<view>/<sextant_id>.png
  tooth_bbox/<case_id>/<view>.json
  mask/<case_id>/<view>.png
  preview/<case_id>/<view>.png

Padding summary:
- Single-tooth crops truncate the YOLO box with int(), clamp it to image bounds,
  and crop without extra padding.
- Sextant crops use directional padding: upper sextants expand mostly upward,
  lower sextants mostly downward, both sides expand by 0.1 of union width, and
  side views add an extra 0.1 toward the presumed posterior side.
- Crop coordinates are converted with int() truncation before clamping so the
  JSON bbox_padded matches the saved crop.

The mask PNG has the same height and width as the source image. Background is
0 and every detected tooth region is 255 (a binary uint8 mask). The tooth_bbox
JSON is always rewritten, while crops, masks, and previews are only generated
when missing (unless --overwrite_existing_masks is given).
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from sam import sam_load, sam_predict
from utils import suppress_stdout


REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_BASE_DIR = REPO_ROOT / ".datasets" / "intraoral" / "test"
DEFAULT_WEIGHT_DIR = REPO_ROOT / ".checkpoints" / "segtooth_model"

# View mapping: filename prefix -> view type required by predict().
# D (Down/Lower), U (Upper), F (Front), L (Left), R (Right)
VIEW_MAPPING: Dict[str, Literal["front", "upper", "lower", "left", "right"]] = {
    "F": "front",
    "U": "upper",
    "D": "lower",
    "L": "left",
    "R": "right",
}

# 定义每个视角下需要提取的牙齿 FDI 编号
TARGET_TEETH_PER_VIEW: Dict[str, List[int]] = {
    "front": [13, 12, 11, 21, 22, 23, 43, 42, 41, 31, 32, 33],
    "left": [24, 25, 26, 27, 28, 34, 35, 36, 37, 38],
    "right": [14, 15, 16, 17, 18, 44, 45, 46, 47, 48],
    "upper": [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28],
    "lower": [38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48],
}

# Sextant 定义: 每个区域包含哪些 FDI 牙位
# S1: 右上后牙, S2: 上前牙, S3: 左上后牙
# S4: 左下后牙, S5: 下前牙, S6: 右下后牙
SEXTANT_DEFINITIONS: Dict[str, List[int]] = {
    "S1": [18, 17, 16, 15, 14, 13],  # 包含13以辅助边界
    "S2": [13, 12, 11, 21, 22, 23],
    "S3": [23, 24, 25, 26, 27, 28],  # 包含23以辅助边界
    "S4": [38, 37, 36, 35, 34, 33],  # 包含33
    "S5": [33, 32, 31, 41, 42, 43],
    "S6": [43, 44, 45, 46, 47, 48],  # 包含43
}

# 视图与 Sextant 的对应关系 (即在这个视图下应该切哪些 Sextant)
VIEW_TO_SEXTANT_MAPPING: Dict[str, List[str]] = {
    "front": ["S2", "S5"],
    "right": ["S1", "S6"],  # 此时 Right View 显示的是患者右侧
    "left": ["S3", "S4"],
    "upper": [],  # 通常全口咬合面不切 Sextant
    "lower": [],  # 通常全口咬合面不切 Sextant
}

LEFT_CLASSES = [
    "le28", "le27", "le26", "le25", "le24", "le23", "le22", "le21",
    "le38", "le37", "le36", "le35", "le34", "le33", "le32", "le31",
    "le11", "le12", "le13", "le14", "le41", "le42", "le43", "le44",
]

GUM_PADDING_RATIO = 0.4
SIDE_PADDING_RATIO = 0.1
POSTERIOR_EXTRA_SIDE_RATIO = 0.1
UPPER_SMALL_OPPOSITE_GUM_RATIO = 0.05
LOWER_SMALL_OPPOSITE_GUM_RATIO = 0.05


@contextmanager
def load_torch_checkpoint_on_available_device() -> Iterable[None]:
    """Make third-party checkpoint loaders work on CPU-only hosts.

    ``segment_anything_hq`` calls ``torch.load`` internally without a
    ``map_location`` argument.  Its checkpoints may have been saved from a
    CUDA host, which otherwise makes loading fail before the model can run on
    CPU.  Keep the override tightly scoped to model construction and preserve
    the original behavior whenever CUDA is available.
    """
    import torch

    if torch.cuda.is_available():
        yield
        return

    original_torch_load = torch.load

    def torch_load_on_cpu(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("map_location", torch.device("cpu"))
        return original_torch_load(*args, **kwargs)

    torch.load = torch_load_on_cpu
    try:
        yield
    finally:
        torch.load = original_torch_load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite tooth_bbox JSON and fill missing crops, masks, and previews."
    )
    parser.add_argument("--base_dir", type=Path, default=DEFAULT_BASE_DIR,
                        help="Dataset root containing <date>_process folders.")
    parser.add_argument("--weight_dir", type=Path, default=DEFAULT_WEIGHT_DIR,
                        help="Directory containing SegmentAnyTooth YOLO weights.")
    parser.add_argument("--min_conf", type=float, default=0.35,
                        help="Minimum YOLO detection confidence.")
    parser.add_argument("--min_area", type=int, default=500,
                        help="Minimum YOLO box area in pixels.")
    parser.add_argument("--sam_batch_size", type=int, default=10,
                        help="SAM batch size used when a mask or preview must be generated.")
    parser.add_argument("--overwrite_existing_masks", action="store_true",
                        help="Regenerate existing mask PNGs (for example, to convert legacy label masks to binary).")
    parser.add_argument("--json_only", action="store_true",
                        help="Only rewrite tooth_bbox JSON; do not write crops, masks, or previews.")
    return parser.parse_args()


def get_model_path(
    model: Literal["upper", "lower", "left", "right", "front", "sam"],
    weight_dir: Path,
) -> str:
    """Return the file path to the model weights."""
    if model == "left":
        model = "right"  # Left view uses right model with flipping
    if model == "sam":
        name = "vit_tiny.pt"
    else:
        name = f"yolo11_{model}.pt"
    return str(weight_dir / f"segmentanytooth_{name}")


def class_name_for(names: Any, cls_id: int, use_left_classes: bool) -> str:
    """Resolve a YOLO class id to its class/left-view name."""
    if use_left_classes:
        return LEFT_CLASSES[cls_id]
    return str(names[cls_id])


def parse_fdi(class_name: str) -> int:
    """Parse the FDI tooth number from a YOLO class name (last two digits)."""
    digits = "".join(ch for ch in class_name if ch.isdigit())
    if not digits:
        raise ValueError(f"cannot parse FDI from class name: {class_name}")
    return int(digits[-2:])


def predict_boxes(
    image_path: Path,
    view: Literal["upper", "lower", "left", "right", "front"],
    weight_dir: Path,
    min_conf: float,
    min_area: int,
) -> List[Dict[str, Any]]:
    """Run YOLO detection and return one filtered, de-duplicated box per FDI."""
    try:
        from ultralytics import YOLO
        from ultralytics.utils import LOGGER
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for YOLO detection. Run this script in the "
            "same environment used by segtooth.py."
        ) from exc

    LOGGER.setLevel("ERROR")

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"could not read image: {image_path}")

    should_flip = view == "left"
    infer_image = cv2.flip(image, 1) if should_flip else image

    with suppress_stdout():
        yolo = YOLO(model=get_model_path(view, weight_dir))
        result = yolo.predict(
            infer_image,
            save=False,
            save_txt=False,
            save_conf=False,
            save_crop=False,
            project=None,
        )[0]

    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy().astype(np.int32)
    confs = result.boxes.conf.cpu().numpy()

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    valid_mask = (confs >= min_conf) & ((widths * heights) >= min_area)
    boxes = boxes[valid_mask]
    clss = clss[valid_mask]
    confs = confs[valid_mask]
    if len(boxes) == 0:
        return []

    fdis = np.array(
        [parse_fdi(class_name_for(result.names, int(cls_id), should_flip)) for cls_id in clss],
        dtype=np.int32,
    )

    # Keep the highest-confidence box for each FDI.
    final_indices: List[int] = []
    for fdi in np.unique(fdis):
        indices = np.where(fdis == fdi)[0]
        best_local_idx = np.argmax(confs[indices])
        final_indices.append(int(indices[best_local_idx]))

    boxes = boxes[final_indices]
    clss = clss[final_indices]
    confs = confs[final_indices]
    fdis = fdis[final_indices]

    order = np.argsort(clss)
    boxes = boxes[order]
    clss = clss[order]
    confs = confs[order]
    fdis = fdis[order]

    if should_flip:
        image_width = infer_image.shape[1]
        flipped_boxes = boxes.copy()
        flipped_boxes[:, [0, 2]] = image_width - flipped_boxes[:, [2, 0]]
        boxes = flipped_boxes

    records: List[Dict[str, Any]] = []
    for fdi, box, conf in zip(fdis, boxes, confs):
        records.append({
            "fdi_tooth_name": int(fdi),
            "box": [float(v) for v in box.tolist()],
            "confidence": float(conf),
        })
    return records


def predict_mask_with_sam(
    image: np.ndarray,
    boxes_data: List[Dict[str, Any]],
    weight_dir: Path,
    sam_batch_size: int,
) -> np.ndarray:
    """Return a full-resolution label mask where each tooth pixel is its FDI."""
    if not boxes_data:
        return np.zeros(image.shape[:2], dtype=np.uint8)

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required to generate masks or previews. Run this script in the "
            "same environment used by segtooth.py, or use --json_only if you only "
            "need tooth_bbox JSON."
        ) from exc

    with suppress_stdout(), load_torch_checkpoint_on_available_device():
        sam = sam_load(get_model_path("sam", weight_dir))
    # ``CUDA_VISIBLE_DEVICES`` renumbers the selected card to ``cuda:0``;
    # selecting plain ``cuda`` therefore uses the card assigned to this
    # process rather than a physical GPU index hard-coded in the script.
    sam.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    sam.eval()

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes_xyxy = np.array([item["box"] for item in boxes_data], dtype=np.float32)
    sam_masks = sam_predict(
        sam=sam,
        boxes_xyxy=boxes_xyxy,
        image=image_rgb,
        batch_size=sam_batch_size,
    )

    predict_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for item, current_mask in zip(boxes_data, sam_masks):
        fdi = int(item["fdi_tooth_name"])
        predict_mask[current_mask == 1] = fdi

    return predict_mask


def get_color_map() -> Dict[int, Tuple[int, int, int]]:
    """生成 FDI 标签到 RGB 颜色的映射"""
    import matplotlib.pyplot as plt

    fdi_labels = []
    for quadrant in [1, 2, 3, 4]:
        for tooth_num in range(1, 9):
            fdi_labels.append(quadrant * 10 + tooth_num)

    cmap = plt.get_cmap("gist_ncar", len(fdi_labels))
    color_map: Dict[int, Tuple[int, int, int]] = {0: (0, 0, 0)}
    for i, label in enumerate(fdi_labels):
        rgb_float = np.array(cmap(i)[:3])
        rgb_int = (rgb_float * 255).astype(np.uint8)
        color_map[label] = tuple(rgb_int.tolist())
    return color_map


def save_preview_original_style(
    image: np.ndarray,
    mask: np.ndarray,
    boxes_data: List[Dict[str, Any]],
    preview_path: Path,
) -> None:
    """Save the original segtooth-style visualization PNG (mask + boxes)."""
    if preview_path.exists():
        return

    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    color_map = get_color_map()
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in color_map.items():
        colored_mask[mask == label] = color

    fig, ax = plt.subplots(1, 1, figsize=(w / 100, h / 100), dpi=100)
    label_fontsize = 20
    text_color = "cyan"
    box_color = "yellow"
    mask_alpha = 0.5

    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.imshow(colored_mask, alpha=mask_alpha)

    for item in boxes_data:
        class_name = str(item["fdi_tooth_name"])
        box = item.get("box")
        if box is None or len(box) != 4:
            continue
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor=box_color,
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            xmin,
            ymin - 10,
            class_name,
            color=text_color,
            fontsize=label_fontsize,
            fontweight="bold",
            bbox=dict(facecolor="red", alpha=0.5, edgecolor="none", boxstyle="round,pad=0.1"),
        )

    ax.axis("off")
    fig.savefig(preview_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def clamp_int_box(values: Sequence[float], img_w: int, img_h: int) -> Tuple[List[int], bool]:
    """int()-truncate a box, clamp it to the image, and report if it was clamped."""
    x1, y1, x2, y2 = (int(v) for v in values)
    clamped = x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)
    return [x1, y1, x2, y2], clamped


def pad_sextant_box(
    union_box: Sequence[float],
    sextant_id: str,
    view_type: str,
    img_w: int,
    img_h: int,
) -> Tuple[List[int], bool, Dict[str, float]]:
    """Apply the directional sextant padding and return the padded box."""
    union_x1, union_y1, union_x2, union_y2 = union_box
    box_w = union_x2 - union_x1
    box_h = union_y2 - union_y1

    is_upper = sextant_id in {"S1", "S2", "S3"}
    if is_upper:
        pad_top = box_h * GUM_PADDING_RATIO
        pad_bottom = box_h * UPPER_SMALL_OPPOSITE_GUM_RATIO
    else:
        pad_top = box_h * LOWER_SMALL_OPPOSITE_GUM_RATIO
        pad_bottom = box_h * GUM_PADDING_RATIO

    pad_left = box_w * SIDE_PADDING_RATIO
    pad_right = box_w * SIDE_PADDING_RATIO
    extra_side_ratio = 0.0
    if view_type == "right":
        pad_left += box_w * POSTERIOR_EXTRA_SIDE_RATIO
        extra_side_ratio = POSTERIOR_EXTRA_SIDE_RATIO
    elif view_type == "left":
        pad_right += box_w * POSTERIOR_EXTRA_SIDE_RATIO
        extra_side_ratio = POSTERIOR_EXTRA_SIDE_RATIO

    padded, clamped = clamp_int_box(
        (union_x1 - pad_left, union_y1 - pad_top, union_x2 + pad_right, union_y2 + pad_bottom),
        img_w,
        img_h,
    )
    pad_params = {
        "gum_ratio": GUM_PADDING_RATIO,
        "side_ratio": SIDE_PADDING_RATIO,
        "extra_side_ratio": extra_side_ratio,
        "small_opposite_gum_ratio": (
            UPPER_SMALL_OPPOSITE_GUM_RATIO if is_upper else LOWER_SMALL_OPPOSITE_GUM_RATIO
        ),
    }
    return padded, clamped, pad_params


def valid_box(box: Sequence[int]) -> bool:
    return box[2] > box[0] and box[3] > box[1]


def rel_path(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def save_crop(image: np.ndarray, bbox: Sequence[int], save_path: Path) -> None:
    if save_path.exists():
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2].copy()
    cv2.imwrite(str(save_path), crop)


def save_binary_tooth_mask(mask: np.ndarray, save_path: Path, overwrite: bool = False) -> None:
    """Save an original-resolution binary tooth mask."""
    if save_path.exists() and not overwrite:
        return
    if mask.ndim != 2:
        raise ValueError(f"tooth mask must be HxW, got shape {mask.shape}")
    binary_mask = np.where(mask != 0, 255, 0).astype(np.uint8)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(save_path), binary_mask):
        raise IOError(f"failed to write tooth mask: {save_path}")


def load_annotation(json_path: Path) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def boxes_from_annotation(
    annotation: Dict[str, Any],
    json_path: Path,
) -> List[Dict[str, Any]]:
    """Recover YOLO/SAM-style box records from an existing annotation."""
    boxes_data: List[Dict[str, Any]] = []
    for index, tooth in enumerate(annotation.get("teeth", [])):
        fdi = tooth.get("fdi", tooth.get("fdi_tooth_name"))
        box = tooth.get("bbox_yolo", tooth.get("box"))
        if fdi is None or box is None or len(box) != 4:
            raise ValueError(
                f"invalid tooth record {index} in {json_path}: expected fdi and a 4-value bbox_yolo"
            )
        boxes_data.append({
            "fdi_tooth_name": int(fdi),
            "box": [float(value) for value in box],
            "confidence": float(tooth.get("confidence", 1.0)),
        })
    return boxes_data


def build_tooth_records(
    image: np.ndarray,
    boxes_data: List[Dict[str, Any]],
    date_path: Path,
    case_id: str,
    view_stem: str,
    view_type: str,
    write_outputs: bool,
) -> List[Dict[str, Any]]:
    img_h, img_w = image.shape[:2]
    target_fdis = set(TARGET_TEETH_PER_VIEW.get(view_type, []))
    records: List[Dict[str, Any]] = []

    for item in boxes_data:
        fdi = int(item["fdi_tooth_name"])
        if fdi not in target_fdis:
            continue

        crop_box, clamped = clamp_int_box(item["box"], img_w, img_h)
        if not valid_box(crop_box):
            continue

        crop_path = date_path / "single_tooth" / case_id / view_stem / f"{fdi}.png"
        if write_outputs:
            save_crop(image, crop_box, crop_path)
        records.append({
            "fdi": fdi,
            "bbox_yolo": item["box"],
            "confidence": item["confidence"],
            "pad_ratio": 0.0,
            "bbox_padded": crop_box,
            "clamped": clamped,
            "crop_path": rel_path(crop_path, date_path),
        })

    return records


def build_sextant_records(
    image: np.ndarray,
    boxes_data: List[Dict[str, Any]],
    date_path: Path,
    case_id: str,
    view_stem: str,
    view_type: str,
    write_outputs: bool,
) -> List[Dict[str, Any]]:
    img_h, img_w = image.shape[:2]
    target_sextants = VIEW_TO_SEXTANT_MAPPING.get(view_type, [])
    if not target_sextants:
        return []

    detection_map = {int(item["fdi_tooth_name"]): item for item in boxes_data}
    records: List[Dict[str, Any]] = []

    for sextant_id in target_sextants:
        member_fdis = [fdi for fdi in SEXTANT_DEFINITIONS[sextant_id] if fdi in detection_map]
        if not member_fdis:
            continue

        relevant_boxes = np.array(
            [detection_map[fdi]["box"] for fdi in member_fdis], dtype=np.float32
        )
        union_box = [
            float(np.min(relevant_boxes[:, 0])),
            float(np.min(relevant_boxes[:, 1])),
            float(np.max(relevant_boxes[:, 2])),
            float(np.max(relevant_boxes[:, 3])),
        ]

        padded, clamped, pad_params = pad_sextant_box(union_box, sextant_id, view_type, img_w, img_h)
        if not valid_box(padded):
            continue

        crop_path = date_path / "sextant" / case_id / view_stem / f"{sextant_id}.png"
        if write_outputs:
            save_crop(image, padded, crop_path)
        records.append({
            "id": sextant_id,
            "member_fdi_detected": member_fdis,
            "bbox_union_raw": union_box,
            "bbox_padded": padded,
            "pad_params": pad_params,
            "clamped": clamped,
            "crop_path": rel_path(crop_path, date_path),
        })

    return records


def write_annotation(
    date_path: Path,
    case_id: str,
    view_stem: str,
    image_path: Path,
    image: np.ndarray,
    tooth_records: List[Dict[str, Any]],
    sextant_records: List[Dict[str, Any]],
) -> Path:
    json_path = date_path / "tooth_bbox" / case_id / f"{view_stem}.json"

    img_h, img_w = image.shape[:2]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    annotation = {
        "case_id": case_id,
        "view": view_stem,
        "image_path": rel_path(image_path, date_path),
        "orig_size": [img_w, img_h],
        "teeth": tooth_records,
        "sextants": sextant_records,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, ensure_ascii=False, indent=2)
    return json_path


def iter_image_files(case_dir: Path) -> Iterable[Tuple[str, str, Path]]:
    valid_extensions = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]
    for stem, view_type in VIEW_MAPPING.items():
        for ext in valid_extensions:
            image_path = case_dir / f"{stem}{ext}"
            if image_path.exists():
                yield stem, view_type, image_path
                break


def process_view(
    image_path: Path,
    date_path: Path,
    case_id: str,
    view_stem: str,
    view_type: str,
    args: argparse.Namespace,
) -> None:
    json_path = date_path / "tooth_bbox" / case_id / f"{view_stem}.json"
    mask_path = date_path / "mask" / case_id / f"{view_stem}.png"
    preview_path = date_path / "preview" / case_id / f"{view_stem}.png"

    # Reuse the recorded boxes from an existing annotation so YOLO does not have
    # to run again. The JSON itself is always rewritten below.
    boxes_data: Optional[List[Dict[str, Any]]] = None
    if json_path.exists():
        annotation = load_annotation(json_path)
        if "teeth" in annotation:
            boxes_data = boxes_from_annotation(annotation, json_path)
        # A file without a "teeth" list uses an incompatible/legacy schema, so
        # fall through and recompute the boxes with YOLO rather than writing an
        # empty annotation over it.

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"could not read image: {image_path}")

    if boxes_data is None:
        boxes_data = predict_boxes(
            image_path=image_path,
            view=view_type,
            weight_dir=args.weight_dir.resolve(),
            min_conf=args.min_conf,
            min_area=args.min_area,
        )

    tooth_records = build_tooth_records(
        image=image,
        boxes_data=boxes_data,
        date_path=date_path,
        case_id=case_id,
        view_stem=view_stem,
        view_type=view_type,
        write_outputs=not args.json_only,
    )
    sextant_records = build_sextant_records(
        image=image,
        boxes_data=boxes_data,
        date_path=date_path,
        case_id=case_id,
        view_stem=view_stem,
        view_type=view_type,
        write_outputs=not args.json_only,
    )
    write_annotation(
        date_path=date_path,
        case_id=case_id,
        view_stem=view_stem,
        image_path=image_path,
        image=image,
        tooth_records=tooth_records,
        sextant_records=sextant_records,
    )

    if args.json_only:
        return

    needs_mask = args.overwrite_existing_masks or not mask_path.exists()
    needs_preview = not preview_path.exists()
    if needs_mask or needs_preview:
        mask = predict_mask_with_sam(
            image=image,
            boxes_data=boxes_data,
            weight_dir=args.weight_dir.resolve(),
            sam_batch_size=args.sam_batch_size,
        )
        if needs_mask:
            save_binary_tooth_mask(mask, mask_path, overwrite=args.overwrite_existing_masks)
        if needs_preview:
            save_preview_original_style(
                image=image,
                mask=mask,
                boxes_data=boxes_data,
                preview_path=preview_path,
            )


def main() -> None:
    args = parse_args()
    args.base_dir = Path(args.base_dir).expanduser().resolve()
    args.weight_dir = Path(args.weight_dir).expanduser().resolve()

    if not args.base_dir.exists():
        raise FileNotFoundError(f"base_dir does not exist: {args.base_dir}")
    if not args.weight_dir.exists():
        raise FileNotFoundError(f"weight_dir does not exist: {args.weight_dir}")

    error_log_path = args.base_dir / "process_errors.txt"
    date_folders = sorted(
        d for d in args.base_dir.iterdir() if d.is_dir() and d.name.endswith("_process")
    )
    if not date_folders:
        print(f"No '*_process' folders found under {args.base_dir}")
        return

    for date_path in tqdm(date_folders, desc="Dates", unit="date"):
        input_root = date_path / "process"
        if not input_root.exists():
            continue

        case_dirs = sorted(d for d in input_root.iterdir() if d.is_dir())
        for case_dir in tqdm(case_dirs, desc=f"  {date_path.name}", unit="case", leave=False):
            for view_stem, view_type, image_path in iter_image_files(case_dir):
                try:
                    process_view(
                        image_path=image_path,
                        date_path=date_path,
                        case_id=case_dir.name,
                        view_stem=view_stem,
                        view_type=view_type,
                        args=args,
                    )
                except Exception as exc:  # keep batch processing resilient
                    tqdm.write(f"Skipping {image_path} due to error: {exc}")
                    with open(error_log_path, "a", encoding="utf-8") as f:
                        f.write(f"ERROR | Path: {image_path} | Reason: {exc}\n")

    if error_log_path.exists():
        print(f"Processing finished. Some errors occurred, check: {error_log_path}")
    else:
        print("All tasks completed successfully without errors.")


if __name__ == "__main__":
    main()
