"""Dataset discovery and in-memory loading for tooth bounding boxes."""

from __future__ import annotations

import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
from PIL import Image
from ultralytics.data.dataset import YOLODataset


VIEW_TO_STEM = {
    "front": "F",
    "upper": "U",
    "lower": "D",
    "right": "R",
    "left": "L",
}
MODEL_VIEWS = ("front", "upper", "lower", "right")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
FDI_PATTERN = re.compile(r"(?<!\d)([1-4][1-8])(?!\d)")

# This ordering must remain identical to segtooth_new.py. During left-view
# inference, class ID i is interpreted as LEFT_CLASSES[i].
LEFT_CLASSES = [
    "le28", "le27", "le26", "le25", "le24", "le23", "le22", "le21",
    "le38", "le37", "le36", "le35", "le34", "le33", "le32", "le31",
    "le11", "le12", "le13", "le14", "le41", "le42", "le43", "le44",
]


@dataclass(frozen=True)
class ToothSample:
    """One source image and its manually revised bounding-box JSON."""

    image_path: Path
    annotation_path: Path
    subject_id: str
    source_view: str

    @property
    def flip_horizontal(self) -> bool:
        return self.source_view == "left"


def normalize_class_names(names: Mapping[int, str] | Sequence[str]) -> dict[int, str]:
    """Return checkpoint class names as a contiguous ``class_id -> name`` dict."""

    if isinstance(names, Mapping):
        normalized = {int(index): str(name) for index, name in names.items()}
    else:
        normalized = {index: str(name) for index, name in enumerate(names)}

    expected = list(range(len(normalized)))
    if sorted(normalized) != expected:
        raise ValueError(
            "YOLO class indices must be contiguous and start at 0; "
            f"received {sorted(normalized)}"
        )
    return dict(sorted(normalized.items()))


def extract_fdi(value: object) -> int | None:
    """Extract a permanent-tooth FDI number from a JSON key or class name."""

    matches = FDI_PATTERN.findall(str(value))
    return int(matches[-1]) if matches else None


def build_fdi_to_class(
    names: Mapping[int, str] | Sequence[str],
) -> dict[int, int]:
    """Build the FDI-to-class-ID mapping encoded by a class-name sequence."""

    class_names = normalize_class_names(names)
    result: dict[int, int] = {}
    for class_id, class_name in class_names.items():
        fdi = extract_fdi(class_name)
        if fdi is None:
            continue
        if fdi in result:
            raise ValueError(
                f"Multiple classes encode FDI {fdi}: "
                f"{result[fdi]} and {class_id}"
            )
        result[fdi] = class_id
    if not result:
        raise ValueError(
            "Could not extract any FDI labels from checkpoint class names: "
            f"{list(class_names.values())}"
        )
    return result


def validate_left_class_mapping(
    checkpoint_names: Mapping[int, str] | Sequence[str],
) -> dict[int, int]:
    """Validate and return the L-view mapping used by ``segtooth_new.py``."""

    names = normalize_class_names(checkpoint_names)
    if len(names) != len(LEFT_CLASSES):
        raise ValueError(
            "The right checkpoint has "
            f"{len(names)} classes, but segtooth_new.py LEFT_CLASSES has "
            f"{len(LEFT_CLASSES)} entries. Their class IDs cannot be aligned safely."
        )
    right_mapping = build_fdi_to_class(names)
    left_mapping = build_fdi_to_class(LEFT_CLASSES)

    # At a fixed class ID, a flipped L tooth must match the corresponding R tooth.
    mirror_quadrant = {1: 2, 2: 1, 3: 4, 4: 3}
    for left_fdi, class_id in left_mapping.items():
        quadrant, tooth = divmod(left_fdi, 10)
        mirrored_fdi = mirror_quadrant[quadrant] * 10 + tooth
        if right_mapping.get(mirrored_fdi) != class_id:
            raise ValueError(
                "Checkpoint class order is incompatible with segtooth_new.py: "
                f"class {class_id} should pair R FDI {mirrored_fdi} with "
                f"L FDI {left_fdi}."
            )
    return left_mapping


def _find_named_file(
    directory: Path,
    stem: str,
    extensions: Sequence[str],
) -> Path | None:
    """Find one case-insensitive filename match in a directory."""

    if not directory.is_dir():
        return None
    candidates = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.stem.upper() == stem.upper()
        and path.suffix.lower() in extensions
    ]
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple files match stem {stem!r} in {directory}: {candidates}"
        )
    return candidates[0] if candidates else None


def find_process_folders(data_root: str | Path) -> list[Path]:
    """Find every ``*_process`` directory containing a nested ``process`` dir."""

    root = Path(data_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")

    candidates: set[Path] = set()
    for current, directory_names, _ in os.walk(root):
        path = Path(current)
        if path.name.endswith("_process") and (path / "process").is_dir():
            candidates.add(path)
            # A process folder cannot contain another independent process root.
            directory_names.clear()
    return sorted(candidates)


def discover_samples(
    data_root: str | Path,
    model_view: str,
    annotation_dir: str = "tooth_bbox_revise",
) -> list[ToothSample]:
    """Collect one detector's images; right always includes flipped L views."""

    if model_view not in MODEL_VIEWS:
        raise ValueError(f"model_view must be one of {MODEL_VIEWS}, got {model_view}")

    source_views = [model_view]
    if model_view == "right":
        source_views.append("left")

    samples: list[ToothSample] = []
    for process_folder in find_process_folders(data_root):
        image_root = process_folder / "process"
        for subject_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
            for source_view in source_views:
                stem = VIEW_TO_STEM[source_view]
                image_path = _find_named_file(
                    subject_dir,
                    stem,
                    IMAGE_EXTENSIONS,
                )
                annotation_path = _find_named_file(
                    process_folder / annotation_dir / subject_dir.name,
                    stem,
                    (".json",),
                )
                if image_path is None or annotation_path is None:
                    continue
                samples.append(
                    ToothSample(
                        image_path=image_path,
                        annotation_path=annotation_path,
                        subject_id=subject_dir.name,
                        source_view=source_view,
                    )
                )
    return samples


def load_annotation(annotation_path: str | Path) -> list[tuple[int, list[float]]]:
    """Read ``{FDI: [{"box": [x1, y1, x2, y2]}]}`` annotations."""

    path = Path(annotation_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Annotation root must be a JSON object: {path}")

    annotations: list[tuple[int, list[float]]] = []
    for fdi_key, entries in payload.items():
        fdi = extract_fdi(fdi_key)
        if fdi is None:
            raise ValueError(f"Invalid FDI key {fdi_key!r} in {path}")
        if not isinstance(entries, list):
            raise ValueError(f"FDI {fdi} must contain a list of boxes in {path}")
        for entry in entries:
            if not isinstance(entry, dict) or "box" not in entry:
                raise ValueError(
                    f"FDI {fdi} entries must be objects containing 'box' in {path}"
                )
            box = entry["box"]
            if not (
                isinstance(box, list)
                and len(box) == 4
                and all(isinstance(value, (int, float)) for value in box)
            ):
                raise ValueError(
                    f"FDI {fdi} box must contain four numeric values in {path}"
                )
            annotations.append((fdi, [float(value) for value in box]))
    return annotations


def prepare_targets(
    sample: ToothSample,
    fdi_to_class: Mapping[int, int],
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Map JSON FDI labels to class IDs and return clipped XYXY boxes.

    L-view labels use the exact ``LEFT_CLASSES[class_id]`` order from
    ``segtooth_new.py``. Their boxes are flipped with the source image.
    """

    width, height = image_size
    boxes: list[list[float]] = []
    class_ids: list[int] = []
    for fdi, raw_box in load_annotation(sample.annotation_path):
        if fdi not in fdi_to_class:
            raise ValueError(
                f"FDI {fdi} in {sample.annotation_path} is not defined by the "
                f"{sample.source_view} class mapping."
            )
        if len(raw_box) != 4 or not all(math.isfinite(value) for value in raw_box):
            raise ValueError(f"Invalid box for FDI {fdi} in {sample.annotation_path}")

        x1, y1, x2, y2 = raw_box
        x1, x2 = sorted((max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))))
        y1, y2 = sorted((max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))))
        if x2 - x1 < 1.0 or y2 - y1 < 1.0:
            raise ValueError(
                f"Degenerate box for FDI {fdi} in {sample.annotation_path}: {raw_box}"
            )
        if sample.flip_horizontal:
            x1, x2 = float(width) - x2, float(width) - x1

        boxes.append([x1, y1, x2, y2])
        class_ids.append(fdi_to_class[fdi])

    return (
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        np.asarray(class_ids, dtype=np.float32).reshape(-1, 1),
    )


def summarize_samples(
    samples: Sequence[ToothSample],
    class_names: Mapping[int, str] | Sequence[str],
) -> dict[str, object]:
    """Validate all labels and summarize one split without writing files."""

    box_count = 0
    source_views: Counter[str] = Counter()
    class_mappings = {"default": build_fdi_to_class(class_names)}
    if any(sample.source_view == "left" for sample in samples):
        class_mappings["left"] = validate_left_class_mapping(class_names)
    for sample in samples:
        with Image.open(sample.image_path) as image:
            image_size = image.size
        mapping = class_mappings.get(sample.source_view, class_mappings["default"])
        boxes, _ = prepare_targets(sample, mapping, image_size)
        box_count += len(boxes)
        source_views[sample.source_view] += 1
    return {
        "images": len(samples),
        "boxes": box_count,
        "source_views": dict(sorted(source_views.items())),
    }


class InMemoryToothYOLODataset(YOLODataset):
    """Ultralytics dataset backed directly by source images and JSON labels.

    No YOLO label files, image links, image copies, or flipped images are
    written. Left images and their boxes are flipped in memory on every load.
    """

    def __init__(
        self,
        *args,
        samples: Sequence[ToothSample],
        class_names: Mapping[int, str] | Sequence[str],
        **kwargs,
    ) -> None:
        self.samples = list(samples)
        self.class_names = normalize_class_names(class_names)
        self.class_mappings = {
            "default": build_fdi_to_class(self.class_names),
        }
        if any(sample.source_view == "left" for sample in self.samples):
            self.class_mappings["left"] = validate_left_class_mapping(
                self.class_names
            )
        self._sample_by_path = {
            str(sample.image_path.resolve()): sample for sample in self.samples
        }
        super().__init__(*args, cache=False, **kwargs)

    def get_img_files(self, img_path: str | list[str]) -> list[str]:
        del img_path
        image_files = [str(sample.image_path.resolve()) for sample in self.samples]
        if not image_files:
            raise FileNotFoundError("No source images were provided to the dataset")
        return image_files

    def get_labels(self) -> list[dict[str, object]]:
        """Convert revised JSON annotations to Ultralytics label dictionaries."""

        labels: list[dict[str, object]] = []
        valid_files = list(self.im_files)
        for image_path in valid_files:
            sample = self._sample_by_path[str(Path(image_path).resolve())]
            with Image.open(sample.image_path) as image:
                width, height = image.size
            mapping = self.class_mappings.get(
                sample.source_view,
                self.class_mappings["default"],
            )
            boxes, class_ids = prepare_targets(
                sample,
                mapping,
                (width, height),
            )

            xywh = boxes.copy()
            if len(xywh):
                xywh[:, 2:] -= xywh[:, :2]
                xywh[:, :2] += xywh[:, 2:] / 2.0
                xywh[:, [0, 2]] /= float(width)
                xywh[:, [1, 3]] /= float(height)
            else:
                xywh = np.zeros((0, 4), dtype=np.float32)

            labels.append(
                {
                    "im_file": str(sample.image_path.resolve()),
                    "shape": (height, width),
                    "cls": class_ids,
                    "bboxes": xywh,
                    "segments": [],
                    "keypoints": None,
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )
        self.im_files = valid_files
        return labels

    def load_image(
        self,
        index: int,
        *args,
        **kwargs,
    ) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """Load a source image and flip L views in memory before augmentation."""

        image, original_shape, resized_shape = super().load_image(
            index,
            *args,
            **kwargs,
        )
        sample = self._sample_by_path[str(Path(self.im_files[index]).resolve())]
        if sample.flip_horizontal:
            image = cv2.flip(image, 1)
        return image, original_shape, resized_shape


def split_samples(
    samples: Sequence[ToothSample],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[ToothSample], list[ToothSample]]:
    """Split by subject ID so one subject never leaks across splits."""

    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
    grouped: dict[str, list[ToothSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.subject_id, []).append(sample)
    if len(grouped) < 2:
        raise ValueError(
            "At least two distinct subject IDs are required for a train/val split"
        )

    group_ids = sorted(grouped)
    random.Random(seed).shuffle(group_ids)
    val_groups = max(1, round(len(group_ids) * val_ratio))
    val_groups = min(val_groups, len(group_ids) - 1)
    val_ids = set(group_ids[:val_groups])

    train = [sample for sample in samples if sample.subject_id not in val_ids]
    val = [sample for sample in samples if sample.subject_id in val_ids]
    return train, val
