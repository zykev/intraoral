#!/usr/bin/env python3
"""Build the AlphaDent *single tooth* dataset from the full-mouth images + YOLO-seg labels.

Pipeline
--------
1. Read the view of every image from ``view_labels.json`` (produced by
   ``infer_view.py``).
2. Rotate the image to the orientation expected by the SegmentAnyTooth YOLO model of
   that view.  The rotation is picked by trying every candidate rotation and keeping
   the one that yields the most, most-confident tooth detections (the ``upper`` and
   ``lower`` models are orientation sensitive, and some AlphaDent photos are stored
   upside down).
3. Detect the teeth (same logic as ``segtooth.py``: per-view YOLO weights, optional
   horizontal flip for the left view, one box per FDI, FDI whitelist per view).
4. Assign every AlphaDent lesion polygon (caries / filling / crown / abrasion ...) to
   the detected tooth that contains it.
5. Crop each tooth (padded, and expanded so that no polygon is cut off) and convert
   the polygons into the crop coordinate frame.
6. Save the upright full-mouth image next to the crops as ``<stem>.jpg`` so every
   sample folder is self-contained (and so the full-image geometry can be checked).

Every detected tooth is exported, including the ones without any lesion polygon: their
crop is still written and their ``polygons`` / ``label_ids`` are empty, which makes them
usable as negative samples.  Pass ``--only_annotated`` to drop them.

Output layout (added to the original dataset, nothing is moved)::

    alphadent/
      images_tooth/<split>/<stem>/<stem>.jpg    # upright full-mouth image (same name as source)
      images_tooth/<split>/<stem>/<fdi>.png     # one crop per detected tooth
      labels_tooth/<split>/<stem>.json
      single_tooth_report.json                  # optional (--report)

All geometry is expressed in the *rotated* (upright) frame, never in the raw source
frame.  ``<stem>.jpg`` is the upright full image itself, so any of the numbers below can
be checked against it directly.

``labels_tooth/<split>/<stem>.json`` format::

    {
      "image": "images/train/p002_F_52_001.jpg",
      "upright_image": "images_tooth/train/p002_F_52_001/p002_F_52_001.jpg",
      "stem": "p002_F_52_001",
      "split": "train",
      "view": "F",
      "view_confidence": 1.0,
      "rotation": 0,
      "rotation_trials": {"0": 22, "90": 0, "180": 18, "270": 0},
      "orig_size": [5472, 3648],
      "rot_size": [3648, 5472],
      "num_labels": 24,
      "num_detections": 22,
      "num_orphan_polygons": 3,
      "teeth": [
        {
          "crop_path": "images_tooth/train/p002_F_52_001/11.png",
          "fdi": 11,
          "label_ids": [0, 2],
          "label_names": ["Abrasion", "Crown"],
          "crop_size": [883, 1230],                 # == the PNG size == crop_bbox size
          "crop_bbox": [2167, 655, 3050, 1885],     # xyxy, upright full image
          "bbox_xyxy": [2238.1, 740.6, 2990.4, 1800.6],   # xyxy, upright full image
          "confidence": 0.97,
          "clipped": false,
          "polygons": [
            {"label_id": 0, "label_name": "Abrasion",
             "points": [[12, 40], [45, 38], ...],   # pixels inside <fdi>.png, 0 <= x < crop_size[0]
             "area_ratio_inside": 1.0}
          ]
        },
        {
          "crop_path": "images_tooth/train/p002_F_52_001/12.png",
          "fdi": 12,
          "label_ids": [],                       # no lesion on this tooth
          "label_names": [],
          "polygons": [],                        # negative sample
          "...": "..."
        }
      ]
    }

``polygons[i].points`` is in the **single-tooth image** frame: the point ``(0, 0)`` is
the top-left pixel of ``<fdi>.png`` and every coordinate satisfies
``0 <= x <= crop_size[0] - 1`` and ``0 <= y <= crop_size[1] - 1``.  It is obtained by
subtracting ``crop_bbox[:2]`` from the upright full-image coordinate, with no scaling.
To go back to the upright full image: ``point + crop_bbox[:2]``.

Usage
-----
    # dry run on a few images, with debug overlays
    python process_public/alphadent/build_single_tooth.py --limit 4 --save_vis

    # full build
    python process_public/alphadent/build_single_tooth.py --splits train valid

    # check what is already there
    python process_public/alphadent/build_single_tooth.py --report
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Defaults / constants (ported from segtooth.py)
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_DIR = REPO_ROOT / ".datasets" / "intraoral_public" / "alphadent"
DEFAULT_VIEW_JSON = "view_labels.json"

# View letter -> model name used by SegmentAnyTooth.
VIEW_MAPPING: Dict[str, str] = {
    "F": "front",
    "U": "upper",
    "D": "lower",
    "L": "left",
    "R": "right",
}

# FDI whitelist per view.  ``segtooth.py`` restricts the front view to the 12
# anterior teeth and the lateral views to the posterior ones; AlphaDent lesions also
# appear on canines / first premolars / second molars, so the whitelists below are
# deliberately widened (the extra teeth are simply dropped when they carry no label).
TARGET_FDIS_PER_VIEW: Dict[str, List[int]] = {
    "front": [16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26,
              46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36],
    "left": [21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38],
    "right": [11, 12, 13, 14, 15, 16, 17, 18, 41, 42, 43, 44, 45, 46, 47, 48],
    "upper": [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28],
    "lower": [38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48],
}

# Class order of the YOLO flips for the left view (left view = mirrored right view).
LEFT_CLASSES = [
    "le28", "le27", "le26", "le25", "le24", "le23", "le22", "le21",
    "le38", "le37", "le36", "le35", "le34", "le33", "le32", "le31",
    "le11", "le12", "le13", "le14", "le41", "le42", "le43", "le44",
]

# AlphaDent YOLO-seg class names (``yolo_seg_train.yaml``).
ALPHADENT_CLASS_NAMES = [
    "Abrasion", "Filling", "Crown", "Caries 1", "Caries 2",
    "Caries 3", "Caries 4", "Caries 5", "Caries 6",
]

FDI_COLORS = [(0, 255, 0), (255, 128, 0), (0, 200, 255), (255, 0, 255)]


# --------------------------------------------------------------------------- #
# Small geometry helpers
# --------------------------------------------------------------------------- #

def rotate_image(image: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate clockwise by a multiple of 90 degrees (same convention as infer_view)."""
    degrees = int(degrees) % 360
    if degrees == 0:
        return image
    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"rotation must be a multiple of 90, got {degrees}")


def rotate_points(points: np.ndarray, degrees: int, width: int, height: int) -> np.ndarray:
    """Map ``points`` (Nx2, pixel coords) through the same rotation as ``rotate_image``.

    The formulas are the exact pixel mapping of ``cv2.rotate`` (a pixel at index
    ``x`` maps to ``W - 1 - x``), so ``rotate_points`` and its inverse are lossless.
    """
    degrees = int(degrees) % 360
    points = np.asarray(points, dtype=np.float32)
    x = points[:, 0]
    y = points[:, 1]
    if degrees == 0:
        return points.copy()
    if degrees == 90:   # new size = (height, width)
        return np.stack([height - 1 - y, x], axis=1)
    if degrees == 180:
        return np.stack([width - 1 - x, height - 1 - y], axis=1)
    if degrees == 270:  # new size = (height, width)
        return np.stack([y, width - 1 - x], axis=1)
    raise ValueError(f"rotation must be a multiple of 90, got {degrees}")


def rotated_size(width: int, height: int, degrees: int) -> Tuple[int, int]:
    """Width/height after a 90-degree rotation."""
    degrees = int(degrees) % 360
    return (height, width) if degrees in (90, 270) else (width, height)


def clamp_int_box(values: Sequence[float], img_w: int, img_h: int) -> Tuple[List[int], bool]:
    """int()-truncate a box, clamp it to the image, and report if it was clamped."""
    x1, y1, x2, y2 = (int(v) for v in values)
    clamped = x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    return [x1, y1, x2, y2], clamped


def valid_box(box: Sequence[int]) -> bool:
    return box[2] > box[0] and box[3] > box[1]


def polygon_area(points: np.ndarray) -> float:
    return float(abs(cv2.contourArea(np.asarray(points, dtype=np.float32))))


def relative_path(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


# --------------------------------------------------------------------------- #
# YOLO tooth detection (ported from segtooth.py)
# --------------------------------------------------------------------------- #

_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}


def weight_path(model: str, weight_dir: Path) -> Path:
    if model == "left":  # the left view reuses the right weights on a mirrored image
        model = "right"
    return weight_dir / f"segmentanytooth_yolo11_{model}.pt"


def get_model(model: str, weight_dir: Path) -> Any:
    key = (str(weight_dir), model)
    if key not in _MODEL_CACHE:
        from ultralytics import YOLO
        from ultralytics.utils import LOGGER

        LOGGER.setLevel("ERROR")
        path = weight_path(model, weight_dir)
        if not path.is_file():
            raise FileNotFoundError(f"missing YOLO weights: {path}")
        _MODEL_CACHE[key] = YOLO(str(path))
    return _MODEL_CACHE[key]


def class_name_for(names: Any, cls_id: int, use_left_classes: bool) -> str:
    if use_left_classes:
        return LEFT_CLASSES[cls_id]
    return str(names[cls_id])


def parse_fdi(class_name: str) -> int:
    digits = "".join(ch for ch in class_name if ch.isdigit())
    if not digits:
        raise ValueError(f"cannot parse FDI from class name: {class_name}")
    return int(digits[-2:])


def detect_teeth(
    image: np.ndarray,
    view: str,
    weight_dir: Path,
    min_conf: float,
    min_area: int,
) -> List[Dict[str, Any]]:
    """Detect one box per FDI in ``image`` (already in the correct orientation)."""
    model_name = VIEW_MAPPING[view]
    should_flip = model_name == "left"
    infer_image = cv2.flip(image, 1) if should_flip else image

    yolo = get_model(model_name, weight_dir)
    result = yolo.predict(
        infer_image,
        save=False, save_txt=False, save_conf=False, save_crop=False,
        project=None, verbose=False,
    )[0]

    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy().astype(np.int32)
    confs = result.boxes.conf.cpu().numpy()

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    keep = (confs >= min_conf) & ((widths * heights) >= min_area)
    boxes, clss, confs = boxes[keep], clss[keep], confs[keep]
    if len(boxes) == 0:
        return []

    fdis = np.array(
        [parse_fdi(class_name_for(result.names, int(c), should_flip)) for c in clss],
        dtype=np.int32,
    )

    # Keep the highest-confidence box per FDI.
    chosen: List[int] = []
    for fdi in np.unique(fdis):
        idx = np.where(fdis == fdi)[0]
        chosen.append(int(idx[int(np.argmax(confs[idx]))]))

    boxes, clss, confs, fdis = boxes[chosen], clss[chosen], confs[chosen], fdis[chosen]
    order = np.argsort(clss)
    boxes, clss, confs, fdis = boxes[order], clss[order], confs[order], fdis[order]

    if should_flip:  # mirror the boxes back into the un-flipped frame
        width = infer_image.shape[1]
        boxes = boxes.copy()
        boxes[:, [0, 2]] = width - boxes[:, [2, 0]]

    return [
        {"fdi": int(fdi), "box": [float(v) for v in box.tolist()], "confidence": float(conf)}
        for fdi, box, conf in zip(fdis, boxes, confs)
    ]


# --------------------------------------------------------------------------- #
# Rotation selection
# --------------------------------------------------------------------------- #

def handedness_ok(teeth: Sequence[Dict[str, Any]], view: str) -> bool:
    """For the frontal view check that FDI quadrants sit on the expected image side.

    On an upright frontal photo the patient's right side (quadrants 1 and 4) is on the
    viewer's left.  A 180-degree mis-rotation or a mirrored image breaks this, and the
    FDI numbers would silently be wrong, so it is used as a ranking criterion.
    """
    if view != "F":
        return True

    def centre_x(prefixes: Iterable[int]) -> Optional[float]:
        values = [np.mean([t["box"][0], t["box"][2]]) for t in teeth if t["fdi"] // 10 in prefixes]
        return float(np.mean(values)) if values else None

    upper_right, upper_left = centre_x({1}), centre_x({2})
    lower_right, lower_left = centre_x({4}), centre_x({3})
    checks = []
    if upper_right is not None and upper_left is not None:
        checks.append(upper_right < upper_left)
    if lower_right is not None and lower_left is not None:
        checks.append(lower_right < lower_left)
    return all(checks) if checks else True


def coverage_score(
    polygons: Sequence[Dict[str, Any]],
    teeth: Sequence[Dict[str, Any]],
    min_ratio: float,
) -> Tuple[int, float]:
    """How well the detected tooth boxes explain the ground-truth polygons.

    Returns ``(covered, mean_ratio)`` where ``covered`` counts the polygons whose best
    box covers at least ``min_ratio`` of their points (the same threshold used when
    assigning polygons to teeth) and ``mean_ratio`` is the average best coverage.

    This is a much stronger rotation signal than the raw detection count: the YOLO
    models happily fire on a wrongly-rotated image, but only the correct orientation
    makes the annotated lesions fall inside the predicted tooth boxes.
    """
    if not polygons:
        return (0, 0.0)
    if not teeth:
        return (0, 0.0)
    covered = 0
    ratios: List[float] = []
    for polygon in polygons:
        points = polygon["points"]
        best = max(polygons_inside_ratio(points, tooth["box"]) for tooth in teeth)
        ratios.append(best)
        if best >= min_ratio:
            covered += 1
    return (covered, float(np.mean(ratios)))


def score_rotation(
    teeth: Sequence[Dict[str, Any]],
    view: str,
    polygons: Sequence[Dict[str, Any]],
    min_ratio: float,
) -> Tuple[int, float, int, int, float]:
    """Ranking key for one candidate rotation (larger is better).

    Order of importance: polygons explained by the detections, mean polygon coverage,
    handedness (frontal views only), number of teeth, summed detection confidence.
    """
    covered, mean_ratio = coverage_score(polygons, teeth, min_ratio)
    return (
        covered,
        round(mean_ratio, 3),
        int(handedness_ok(teeth, view)),
        len(teeth),
        float(sum(t["confidence"] for t in teeth)),
    )


def choose_rotation(
    image: np.ndarray,
    view: str,
    target_fdis: Optional[set],
    rotations: Sequence[int],
    weight_dir: Path,
    min_conf: float,
    min_area: int,
    polygons: Sequence[Dict[str, Any]] = (),
    min_assign_ratio: float = 0.5,
) -> Tuple[int, List[Dict[str, Any]], Dict[int, Tuple[int, List[Dict[str, Any]]]]]:
    """Try every candidate rotation and keep the best ``(rotation, teeth)``.

    ``polygons`` must be in the ORIGINAL (un-rotated) image frame; they are rotated
    here for every trial so that the polygon coverage can be used as the ranking key.
    Ties are resolved in favour of the earlier entry of ``rotations`` (i.e. 0 first),
    so an already-upright photo is never rotated without evidence.

    Returns the winning rotation, its detections and, for every rotation that was
    tried, ``(number of teeth, detections)`` (useful for debugging / the report).
    """
    image_h, image_w = image.shape[:2]
    trials: Dict[int, Tuple[int, List[Dict[str, Any]]]] = {}
    best_rot = 0
    best_teeth: List[Dict[str, Any]] = []
    best_score: Optional[Tuple[int, float, int, int, float]] = None

    for rotation in rotations:
        rotated = rotate_image(image, rotation)
        teeth = detect_teeth(rotated, view, weight_dir, min_conf, min_area)
        if target_fdis is not None:
            teeth = [t for t in teeth if t["fdi"] in target_fdis]
        rotated_polygons = [
            {"points": rotate_points(p["points"], rotation, image_w, image_h)}
            for p in polygons
        ]
        score = score_rotation(teeth, view, rotated_polygons, min_assign_ratio)
        trials[int(rotation)] = (len(teeth), teeth)
        if best_score is None or score > best_score:
            best_score, best_rot, best_teeth = score, int(rotation), teeth

    return best_rot, best_teeth, trials


# --------------------------------------------------------------------------- #
# Label handling
# --------------------------------------------------------------------------- #

def load_yolo_seg_labels(label_path: Path, img_w: int, img_h: int) -> List[Dict[str, Any]]:
    """Read a YOLO-seg txt file into ``[{label_id, label_name, points (Nx2 px)}]``."""
    polygons: List[Dict[str, Any]] = []
    if not label_path.is_file():
        return polygons
    for line in label_path.read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if len(tokens) < 7:  # class + at least 3 points
            continue
        label_id = int(float(tokens[0]))
        coords = np.asarray([float(v) for v in tokens[1:]], dtype=np.float32)
        if coords.size % 2:
            coords = coords[:-1]
        points = coords.reshape(-1, 2) * np.array([img_w, img_h], dtype=np.float32)
        polygons.append({
            "label_id": label_id,
            "label_name": ALPHADENT_CLASS_NAMES[label_id] if 0 <= label_id < len(ALPHADENT_CLASS_NAMES)
                          else f"class_{label_id}",
            "points": points,
        })
    return polygons


def polygons_inside_ratio(points: np.ndarray, box: Sequence[float]) -> float:
    x1, y1, x2, y2 = box
    inside = ((points[:, 0] >= x1) & (points[:, 0] <= x2) &
              (points[:, 1] >= y1) & (points[:, 1] <= y2))
    return float(inside.mean())


def assign_polygons(
    polygons: Sequence[Dict[str, Any]],
    teeth: Sequence[Dict[str, Any]],
    min_ratio: float,
) -> Tuple[Dict[int, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Attach every polygon to the tooth that contains it best."""
    assigned: Dict[int, List[Dict[str, Any]]] = {index: [] for index in range(len(teeth))}
    orphans: List[Dict[str, Any]] = []

    for polygon in polygons:
        points = polygon["points"]
        centroid = points.mean(axis=0)
        best_index, best_ratio = None, -1.0
        for index, tooth in enumerate(teeth):
            x1, y1, x2, y2 = tooth["box"]
            ratio = polygons_inside_ratio(points, tooth["box"])
            inside_centroid = x1 <= centroid[0] <= x2 and y1 <= centroid[1] <= y2
            score = ratio + (0.25 if inside_centroid else 0.0)
            if score > best_ratio:
                best_index, best_ratio = index, score
        chosen_ratio = polygons_inside_ratio(points, teeth[best_index]["box"]) if best_index is not None else 0.0
        if best_index is None or chosen_ratio < min_ratio:
            orphans.append(polygon)
            continue
        assigned[best_index].append({**polygon, "area_ratio_inside": chosen_ratio})

    return assigned, orphans


def tooth_crop_box(
    tooth: Dict[str, Any],
    polygons: Sequence[Dict[str, Any]],
    pad_ratio: float,
    poly_margin_ratio: float,
    img_w: int,
    img_h: int,
) -> Tuple[List[int], bool]:
    """Padded tooth box, grown further so that every assigned polygon fits inside."""
    x1, y1, x2, y2 = tooth["box"]
    box_w, box_h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    pad_x, pad_y = box_w * pad_ratio, box_h * pad_ratio
    left, top, right, bottom = x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y

    margin = max(box_w, box_h) * poly_margin_ratio
    for polygon in polygons:
        points = polygon["points"]
        left = min(left, float(points[:, 0].min()) - margin)
        top = min(top, float(points[:, 1].min()) - margin)
        right = max(right, float(points[:, 0].max()) + margin)
        bottom = max(bottom, float(points[:, 1].max()) + margin)

    return clamp_int_box((left, top, right, bottom), img_w, img_h)


# --------------------------------------------------------------------------- #
# Per-image processing
# --------------------------------------------------------------------------- #

def process_image(
    split: str,
    stem: str,
    args: argparse.Namespace,
    view_info: Dict[str, Any],
) -> Dict[str, Any]:
    image_path = args.base_dir / "images" / split / f"{stem}.jpg"
    label_path = args.base_dir / "labels" / split / f"{stem}.txt"

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"could not read image: {image_path}")

    height, width = image.shape[:2]
    view = str(view_info["view"]).upper()
    if view not in VIEW_MAPPING:
        raise ValueError(f"invalid view {view!r} for {split}/{stem}")

    target_fdis = None if args.no_fdi_filter else set(TARGET_FDIS_PER_VIEW[VIEW_MAPPING[view]])

    # Ground-truth polygons are needed *before* the rotation search so that the
    # polygon coverage can be used to score the candidate rotations.
    polygons = load_yolo_seg_labels(label_path, width, height)

    if args.rotation_mode == "vlm":
        chosen_rotation = int(view_info.get("rotation", 0)) % 360
        rotated = rotate_image(image, chosen_rotation)
        teeth = detect_teeth(rotated, view, args.weight_dir, args.min_conf, args.min_area)
        if target_fdis is not None:
            teeth = [t for t in teeth if t["fdi"] in target_fdis]
        trials = {chosen_rotation: (len(teeth), teeth)}
    elif args.rotation_mode == "none":
        chosen_rotation = 0
        teeth = detect_teeth(image, view, args.weight_dir, args.min_conf, args.min_area)
        if target_fdis is not None:
            teeth = [t for t in teeth if t["fdi"] in target_fdis]
        trials = {0: (len(teeth), teeth)}
    else:
        chosen_rotation, teeth, trials = choose_rotation(
            image, view, target_fdis, args.rotations, args.weight_dir, args.min_conf, args.min_area,
            polygons=polygons, min_assign_ratio=args.min_assign_ratio,
        )

    rotated_image = rotate_image(image, chosen_rotation)
    rot_w, rot_h = rotated_size(width, height, chosen_rotation)

    for polygon in polygons:
        polygon["points"] = rotate_points(polygon["points"], chosen_rotation, width, height)

    assigned, orphans = assign_polygons(polygons, teeth, args.min_assign_ratio)

    record_teeth: List[Dict[str, Any]] = []
    image_dir = args.base_dir / "images_tooth" / split / stem

    # Keep the upright full-mouth image next to its tooth crops, under the original
    # stem name (so ``<stem>.jpg`` always sits beside ``<fdi>.png``).  This is the exact
    # frame that ``crop_bbox`` / ``bbox_xyxy`` and the polygon points refer to.  It is
    # written even when ``rotation == 0`` so that every folder looks the same.
    upright_path = image_dir / f"{stem}.jpg"
    if args.overwrite or not upright_path.exists():
        upright_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(upright_path), rotated_image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), args.upright_quality])

    for index, tooth in enumerate(teeth):
        tooth_polygons = assigned.get(index, [])
        if not tooth_polygons and not args.export_empty_teeth:
            continue
        crop_box, clamped = tooth_crop_box(
            tooth, tooth_polygons, args.pad_ratio, args.poly_margin_ratio, rot_w, rot_h,
        )
        if not valid_box(crop_box):
            continue

        x1, y1, x2, y2 = crop_box
        crop = rotated_image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_path = image_dir / f"{tooth['fdi']}.png"
        if args.overwrite or not crop_path.exists():
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(crop_path), crop)

        polygon_records = []
        clipped = clamped
        for polygon in tooth_polygons:
            points = polygon["points"] - np.array([x1, y1], dtype=np.float32)
            if (points[:, 0].min() < -0.5 or points[:, 1].min() < -0.5 or
                    points[:, 0].max() > crop.shape[1] + 0.5 or points[:, 1].max() > crop.shape[0] + 0.5):
                clipped = True
            points[:, 0] = np.clip(points[:, 0], 0, crop.shape[1] - 1)
            points[:, 1] = np.clip(points[:, 1], 0, crop.shape[0] - 1)
            polygon_records.append({
                "label_id": int(polygon["label_id"]),
                "label_name": polygon["label_name"],
                "points": [[round(float(px), 1), round(float(py), 1)] for px, py in points],
                "area_ratio_inside": round(float(polygon["area_ratio_inside"]), 4),
            })

        label_ids = sorted({p["label_id"] for p in polygon_records})
        bx1, by1, bx2, by2 = (round(float(v), 1) for v in tooth["box"])
        record_teeth.append({
            "crop_path": relative_path(crop_path, args.base_dir),
            "fdi": int(tooth["fdi"]),
            "label_ids": label_ids,
            "label_names": sorted({p["label_name"] for p in polygon_records}),
            "crop_size": [int(crop.shape[1]), int(crop.shape[0])],
            "crop_bbox": [int(x1), int(y1), int(x2), int(y2)],
            "bbox_xyxy": [bx1, by1, bx2, by2],
            "confidence": round(float(tooth["confidence"]), 4),
            "clipped": bool(clipped),
            "polygons": polygon_records,
        })

    if args.save_vis:
        save_visualisation(args, split, stem, rotated_image, teeth, assigned, record_teeth)

    return {
        "image": relative_path(image_path, args.base_dir),
        "upright_image": relative_path(upright_path, args.base_dir),
        "stem": stem,
        "split": split,
        "view": view,
        "view_confidence": float(view_info.get("confidence", 0.0)),
        "rotation": int(chosen_rotation),
        "rotation_trials": {str(k): v[0] for k, v in sorted(trials.items())},
        "orig_size": [int(width), int(height)],
        "rot_size": [int(rot_w), int(rot_h)],
        "num_labels": len(polygons),
        "num_detections": len(teeth),
        "num_orphan_polygons": len(orphans),
        "teeth": record_teeth,
    }


def save_visualisation(
    args: argparse.Namespace,
    split: str,
    stem: str,
    image: np.ndarray,
    teeth: Sequence[Dict[str, Any]],
    assigned: Dict[int, List[Dict[str, Any]]],
    record_teeth: Sequence[Dict[str, Any]],
) -> None:
    vis = image.copy()
    height, width = vis.shape[:2]
    thickness = max(3, int(round(max(height, width) / 500)))

    for index, tooth in enumerate(teeth):
        x1, y1, x2, y2 = (int(v) for v in tooth["box"])
        color = FDI_COLORS[index % len(FDI_COLORS)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(vis, str(tooth["fdi"]), (x1 + 4, max(y1 + 40, 40)), cv2.FONT_HERSHEY_SIMPLEX,
                    thickness * 1.6, color, thickness, cv2.LINE_AA)
        for polygon in assigned.get(index, []):
            cv2.polylines(vis, [polygon["points"].astype(np.int32)], True, (0, 0, 255), thickness)

    for record in record_teeth:
        x1, y1, x2, y2 = record["crop_bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), max(2, thickness // 2))

    args.vis_dir.mkdir(parents=True, exist_ok=True)
    scale = 1600 / max(height, width)
    if scale < 1:
        vis = cv2.resize(vis, (int(width * scale), int(height * scale)))
    cv2.imwrite(str(args.vis_dir / f"{split}__{stem}.jpg"), vis,
                [int(cv2.IMWRITE_JPEG_QUALITY), 85])


# --------------------------------------------------------------------------- #
# CLI / main
# --------------------------------------------------------------------------- #

def default_weight_dir() -> Path:
    finetuned = REPO_ROOT / "exp" / "segtooth_model"
    if (finetuned / "segmentanytooth_yolo11_front.pt").is_file():
        return finetuned
    return REPO_ROOT / ".checkpoints" / "segtooth_model"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the AlphaDent single-tooth dataset (images_tooth + labels_tooth).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base_dir", type=Path, default=DEFAULT_BASE_DIR, help="AlphaDent dataset root.")
    parser.add_argument("--view_json", type=Path, default=None,
                        help=f"View labels JSON (default: <base_dir>/{DEFAULT_VIEW_JSON}).")
    parser.add_argument("--splits", nargs="+", default=["train", "valid"],
                        help="Splits to process (test has no labels).")
    parser.add_argument("--weight_dir", type=Path, default=default_weight_dir(),
                        help="Directory containing the SegmentAnyTooth YOLO weights.")
    parser.add_argument("--min_conf", type=float, default=0.35, help="Minimum detection confidence.")
    parser.add_argument("--min_area", type=int, default=500, help="Minimum detection box area (px^2).")
    parser.add_argument("--pad_ratio", type=float, default=0.08,
                        help="Symmetric padding around the tooth box, relative to its size.")
    parser.add_argument("--poly_margin_ratio", type=float, default=0.02,
                        help="Extra margin kept around polygons when the crop is grown.")
    parser.add_argument("--min_assign_ratio", type=float, default=0.5,
                        help="Fraction of a polygon's points that must fall inside a tooth box "
                             "for the polygon to be assigned to that tooth.")
    parser.add_argument("--rotation_mode", choices=["search", "vlm", "none"], default="search",
                        help="search: try every rotation and keep the one whose detections best "
                             "cover the ground-truth polygons (recommended); vlm: use the rotation "
                             "stored in view_labels.json (unreliable); none: no rotation.")
    parser.add_argument("--rotations", default="0,90,180,270",
                        help="Comma separated rotations tried in search mode.")
    parser.add_argument("--only_annotated", action="store_true",
                        help="Only export teeth that have at least one label polygon; "
                             "by default every detected tooth is exported and the "
                             "unlabelled ones act as negative samples.")
    parser.add_argument("--no_fdi_filter", action="store_true", help="Do not restrict the FDI whitelist.")
    parser.add_argument("--upright_quality", type=int, default=95,
                        help="JPEG quality used for the saved upright full-mouth image.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing crops / JSON.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N images per split.")
    parser.add_argument("--save_vis", action="store_true", help="Write debug overlay images.")
    parser.add_argument("--vis_dir", type=Path, default=None,
                        help="Debug overlay output directory (default: <base_dir>/single_tooth_vis).")
    parser.add_argument("--report", action="store_true",
                        help="Only summarise the existing labels_tooth output, do not process images.")
    parser.add_argument("--report_path", type=Path, default=None,
                        help="Where to write the summary (default: <base_dir>/single_tooth_report.json).")
    args = parser.parse_args(argv)

    args.rotations = [int(v) % 360 for v in str(args.rotations).split(",") if str(v).strip() != ""]
    args.export_empty_teeth = not args.only_annotated
    if not args.rotations:
        args.rotations = [0]
    args.view_json = args.view_json or (args.base_dir / DEFAULT_VIEW_JSON)
    args.vis_dir = args.vis_dir or (args.base_dir / "single_tooth_vis")
    args.report_path = args.report_path or (args.base_dir / "single_tooth_report.json")
    return args


def write_report(args: argparse.Namespace, records: Sequence[Dict[str, Any]],
                 failures: Sequence[Dict[str, str]],
                 missing_views: Sequence[str] = ()) -> None:
    view_counts: Counter = Counter()
    label_counts: Counter = Counter()
    rotation_counts: Counter = Counter()
    teeth_total = 0
    teeth_annotated = 0
    for record in records:
        view_counts[record["view"]] += 1
        rotation_counts[record["rotation"]] += 1
        teeth_total += len(record["teeth"])
        for tooth in record["teeth"]:
            if tooth["polygons"]:
                teeth_annotated += 1
            for label_id in tooth["label_ids"]:
                label_counts[label_id] += 1

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_dir": str(args.base_dir),
        "splits": args.splits,
        "images": len(records),
        "images_without_teeth": sum(1 for r in records if not r["teeth"]),
        "teeth": teeth_total,
        "teeth_with_labels": teeth_annotated,
        "teeth_without_labels": teeth_total - teeth_annotated,
        "polygons": sum(len(t["polygons"]) for r in records for t in r["teeth"]),
        "views": dict(sorted(view_counts.items())),
        "rotations": dict(sorted(rotation_counts.items())),
        "labels": {ALPHADENT_CLASS_NAMES[k] if k < len(ALPHADENT_CLASS_NAMES) else str(k): v
                   for k, v in sorted(label_counts.items())},
        "missing_view_labels": list(missing_views),
        "failures": list(failures),
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"report written to {args.report_path}")
    print(f"  images={report['images']} teeth={report['teeth']} "
          f"annotated={report['teeth_with_labels']} "
          f"unannotated={report['teeth_without_labels']} "
          f"polygons={report['polygons']} empty={report['images_without_teeth']}")
    print(f"  views={report['views']}")
    print(f"  rotations={report['rotations']}")
    print(f"  labels={report['labels']}")
    if missing_views:
        print(f"  missing_view_labels={len(missing_views)}")
    if failures:
        print(f"  failures={len(failures)} (see report)")


def summarise_existing(args: argparse.Namespace) -> None:
    records: List[Dict[str, Any]] = []
    for split in args.splits:
        label_dir = args.base_dir / "labels_tooth" / split
        if not label_dir.is_dir():
            continue
        for path in sorted(label_dir.glob("*.json")):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
    if not records:
        print("no labels_tooth JSON found")
        return
    write_report(args, records, [])


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.report:
        summarise_existing(args)
        return 0

    if not args.view_json.is_file():
        print(f"[error] view JSON not found: {args.view_json}\n"
              f"        run infer_view.py first.", file=sys.stderr)
        return 1

    view_labels: Dict[str, Any] = json.loads(args.view_json.read_text(encoding="utf-8"))

    tasks: List[Tuple[str, str]] = []
    for split in args.splits:
        image_dir = args.base_dir / "images" / split
        if not image_dir.is_dir():
            print(f"[warn] missing split directory: {image_dir}")
            continue
        stems = sorted(p.stem for p in image_dir.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        stems = [s for s in stems if (args.base_dir / "labels" / split / f"{s}.txt").is_file()]
        if args.limit is not None:
            stems = stems[:args.limit]
        tasks.extend((split, stem) for stem in stems)

    if not tasks:
        print("no labelled images found")
        return 1
    print(f"{len(tasks)} labelled images, weights: {args.weight_dir}")
    print(f"rotation mode: {args.rotation_mode} {args.rotations if args.rotation_mode == 'search' else ''}")

    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    missing_views: List[str] = []
    skipped = 0

    for index, (split, stem) in enumerate(tasks, 1):
        out_path = args.base_dir / "labels_tooth" / split / f"{stem}.json"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        key = f"{split}/{stem}"
        view_info = view_labels.get(key)
        if not view_info or "view" not in view_info:
            missing_views.append(key)
            print(f"[{index}/{len(tasks)}] {key}: skip (no view label)")
            continue

        try:
            record = process_image(split, stem, args, view_info)
        except Exception as exc:  # noqa: BLE001
            failures.append({"image": key, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{index}/{len(tasks)}] {key}: FAILED {exc}")
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
        records.append(record)
        annotated = sum(1 for tooth in record["teeth"] if tooth["polygons"])
        print(f"[{index}/{len(tasks)}] {key} view={record['view']} rot={record['rotation']} "
              f"labels={record['num_labels']} teeth={len(record['teeth'])}/"
              f"{record['num_detections']} annotated={annotated} "
              f"orphans={record['num_orphan_polygons']} trials={record['rotation_trials']}")

    if skipped:
        print(f"{skipped} images already processed (use --overwrite to redo them)")
    if missing_views:
        print(f"{len(missing_views)} images have no view label in {args.view_json.name} "
              f"(run infer_view.py for them, then re-run this script)")

    # Report over everything that is on disk, not only what was just processed.
    all_records: List[Dict[str, Any]] = []
    for split in args.splits:
        label_dir = args.base_dir / "labels_tooth" / split
        if label_dir.is_dir():
            for path in sorted(label_dir.glob("*.json")):
                try:
                    all_records.append(json.loads(path.read_text(encoding="utf-8")))
                except json.JSONDecodeError:
                    continue
    write_report(args, all_records, failures, missing_views)

    if failures:
        print("\nfailures:")
        for failure in failures[:20]:
            print(f"  {failure['image']}: {failure['error']}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
