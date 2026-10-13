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

DEFAULT_ANNOTATION_ROOT = "intraoral_adjust/tooth_bbox_adj"

# Fine-tuning keeps every annotated tooth the checkpoint's class map can express,
# including the "cross-view" FDI numbers that ``segtooth.py::TARGET_TEETH_PER_VIEW``
# discards after inference.  Training on the whole visible arch can only help the
# detector learn to see teeth; the per-view whitelist still applies downstream.
# These sets list the extra FDI numbers that are deliberately kept, and exist so
# the label audit can report how much supervision the widened range adds.
# Keep in sync with ``segtooth.py``.
CROSS_VIEW_FDI = {
    "front": {14, 15, 16, 17, 18, 24, 25, 26, 27, 28,
              34, 35, 36, 37, 38, 44, 45, 46, 47, 48},
    # upper/lower already train on their complete class range (16 teeth each).
    "upper": set(),
    "lower": set(),
    "right": {11, 12, 13, 21, 22, 23, 24, 31, 32, 33, 34, 41, 42, 43},
    "left": {11, 12, 13, 14, 21, 22, 23, 31, 32, 33, 41, 42, 43, 44},
}
UNKNOWN_FDI_POLICIES = ("skip", "raise")

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


def _source_views(model_view: str) -> list[str]:
    """Return the image views that feed one detector.

    ``left`` has no checkpoint of its own: left images are flipped in memory and
    trained as additional ``right`` samples, exactly like ``segtooth.py`` infers
    them.
    """

    if model_view not in MODEL_VIEWS:
        raise ValueError(f"model_view must be one of {MODEL_VIEWS}, got {model_view}")
    return ["right", "left"] if model_view == "right" else [model_view]


def index_process_images(data_root: str | Path) -> dict[tuple[str, str], list[Path]]:
    """Index every source image as ``(subject_id, view_stem) -> [paths]``.

    Subject IDs are not guaranteed to be unique across people, so the value is a
    list and the caller narrows it down with the annotation's own size metadata.
    """

    images: dict[tuple[str, str], list[Path]] = {}
    for process_folder in find_process_folders(data_root):
        image_root = process_folder / "process"
        for subject_dir in image_root.iterdir():
            if not subject_dir.is_dir():
                continue
            for path in subject_dir.iterdir():
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    key = (subject_dir.name, path.stem.upper())
                    images.setdefault(key, []).append(path)
    return images


def resolve_source_image(
    candidates: Sequence[Path],
    expected_size: Sequence[int] | None,
    subject_id: str,
    stem: str,
) -> Path:
    """Pick the single source image whose pixel size matches ``orig_size``."""

    if not candidates:
        raise FileNotFoundError(
            f"No source image found for subject {subject_id!r} view {stem!r} under "
            "the given data root."
        )
    wanted = [int(value) for value in expected_size] if expected_size else None
    matching: list[Path] = []
    for candidate in sorted(candidates):
        try:
            with Image.open(candidate) as image:
                size = list(image.size)
        except OSError:
            continue
        if wanted is None or size == wanted:
            matching.append(candidate)
    if len(matching) == 1:
        return matching[0]
    if not matching:
        raise ValueError(
            f"No candidate image for subject {subject_id!r} view {stem!r} matches "
            f"the annotation size {wanted}: {[str(path) for path in candidates]}"
        )
    raise ValueError(
        f"Subject {subject_id!r} view {stem!r} is ambiguous across "
        f"{[str(path) for path in matching]}; pass a narrower --data-root."
    )


def discover_samples_from_annotation_root(
    annotation_root: str | Path,
    data_root: str | Path,
    model_view: str,
) -> list[ToothSample]:
    """Pair a flat ``<annotation_root>/<subject>/<view>.json`` tree with images.

    ``tooth_bbox_adj`` is a parallel, batch-independent tree, so the source image
    is not stored next to the annotation. It is resolved through the image index
    built from ``data_root`` and confirmed against the annotation's own
    ``orig_size`` metadata, which also catches a wrong subject/image pairing.
    """

    root = Path(annotation_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Annotation root does not exist: {root}")

    image_index = index_process_images(data_root)
    samples: list[ToothSample] = []
    for subject_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for source_view in _source_views(model_view):
            stem = VIEW_TO_STEM[source_view]
            annotation_path = subject_dir / f"{stem}.json"
            if not annotation_path.is_file():
                continue
            with annotation_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            subject_id = str(payload.get("case_id") or subject_dir.name)
            if subject_id != subject_dir.name:
                raise ValueError(
                    f"{annotation_path} declares case_id {subject_id!r} but lives in "
                    f"{subject_dir.name!r}; rename the annotation or the folder."
                )
            samples.append(
                ToothSample(
                    image_path=resolve_source_image(
                        image_index.get((subject_id, stem), []),
                        payload.get("orig_size"),
                        subject_id,
                        stem,
                    ),
                    annotation_path=annotation_path,
                    subject_id=subject_id,
                    source_view=source_view,
                )
            )
    return samples


def discover_samples(
    data_root: str | Path,
    model_view: str,
    annotation_dir: str = "tooth_bbox_revise",
    annotation_root: str | Path | None = None,
) -> list[ToothSample]:
    """Collect one detector's images; right always includes flipped L views.

    ``annotation_root`` selects the flat, batch-independent ground-truth tree
    (``intraoral_adjust/tooth_bbox_adj``); leave it ``None`` to use the legacy
    per-batch ``<*_process>/<annotation_dir>`` layout instead.
    """

    source_views = _source_views(model_view)
    if annotation_root is not None:
        return discover_samples_from_annotation_root(
            annotation_root=annotation_root,
            data_root=data_root,
            model_view=model_view,
        )

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


def _validate_box(box: object, path: Path, where: str) -> list[float]:
    """Return ``box`` as four floats, or raise with a precise message."""

    if not (
        isinstance(box, list)
        and len(box) == 4
        and all(isinstance(value, (int, float)) for value in box)
    ):
        raise ValueError(f"{where} box must contain four numeric values in {path}")
    return [float(value) for value in box]


def load_annotation(annotation_path: str | Path) -> list[tuple[int, list[float]]]:
    """Read either supported annotation layout as ``[(fdi, xyxy), ...]``.

    * revision layout -- ``{FDI: [{"box": [x1, y1, x2, y2]}, ...]}``, used by the
      per-batch ``tooth_bbox_revise`` directories;
    * ground-truth layout -- ``{"teeth": [{"fdi": 16, "bbox_yolo": [...]}]}``,
      used by ``intraoral_adjust/tooth_bbox_adj`` (and by ``unfiltered_bbox`` /
      ``tooth_bbox`` exports).

    In the ground-truth layout ``bbox_yolo`` holds the final corrected box;
    ``bbox_padded`` is the same box rounded to integers for cropping, so it is
    only a fallback.
    """

    path = Path(annotation_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Annotation root must be a JSON object: {path}")

    if "teeth" in payload:
        teeth = payload["teeth"]
        if not isinstance(teeth, list):
            raise ValueError(f"'teeth' must be a list in {path}")
        annotations: list[tuple[int, list[float]]] = []
        for index, tooth in enumerate(teeth):
            if not isinstance(tooth, dict):
                raise ValueError(f"teeth[{index}] must be an object in {path}")
            fdi = tooth.get("fdi")
            if not isinstance(fdi, int):
                raise ValueError(f"teeth[{index}] has no integer 'fdi' in {path}")
            box = next(
                (tooth[key] for key in ("bbox_yolo", "bbox_padded") if key in tooth),
                None,
            )
            annotations.append(
                (fdi, _validate_box(box, path, f"teeth[{index}]"))
            )
        return annotations

    annotations = []
    for fdi_key, entries in payload.items():
        fdi = extract_fdi(fdi_key)
        if fdi is None:
            raise ValueError(f"Invalid FDI key {fdi_key!r} in {path}")
        if not isinstance(entries, list):
            raise ValueError(f"FDI {fdi} must contain a list of boxes in {path}")
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict) or "box" not in entry:
                raise ValueError(
                    f"FDI {fdi} entries must be objects containing 'box' in {path}"
                )
            annotations.append(
                (fdi, _validate_box(entry["box"], path, f"FDI {fdi} entries[{index}]"))
            )
    return annotations


@dataclass
class PreparedTargets:
    """Detector-ready XYXY boxes plus an audit of the labels that produced them."""

    boxes: np.ndarray
    class_ids: np.ndarray
    cross_view: Counter[int]
    unknown_fdi: Counter[int]


def prepare_targets(
    sample: ToothSample,
    fdi_to_class: Mapping[int, int],
    image_size: tuple[int, int],
    unknown: str = "skip",
) -> PreparedTargets:
    """Map JSON FDI labels to class IDs and return clipped XYXY boxes.

    Every tooth the class map can express is kept -- deliberately including the
    cross-view FDI numbers (``CROSS_VIEW_FDI``) that
    ``segtooth.py::TARGET_TEETH_PER_VIEW`` filters out at inference time -- so the
    detector learns to see the whole visible arch instead of only the teeth that
    the downstream per-view whitelist happens to want.

    ``unknown`` decides what happens to an FDI the checkpoint has no class for:
    ``"skip"`` (default) drops that box and reports it in the audit, ``"raise"``
    fails instead.  The class map, not a hard-coded range, is the authority, so
    widening a detector is a matter of its checkpoint's class list.

    L-view labels use the exact ``LEFT_CLASSES[class_id]`` order from
    ``segtooth_new.py``. Their boxes are flipped with the source image.
    """

    if unknown not in UNKNOWN_FDI_POLICIES:
        raise ValueError(
            f"unknown must be one of {UNKNOWN_FDI_POLICIES}, got {unknown!r}"
        )

    width, height = image_size
    widened = CROSS_VIEW_FDI.get(sample.source_view, set())
    boxes: list[list[float]] = []
    class_ids: list[int] = []
    cross_view: Counter[int] = Counter()
    unknown_fdi: Counter[int] = Counter()
    for fdi, raw_box in load_annotation(sample.annotation_path):
        if fdi not in fdi_to_class:
            if unknown == "raise":
                raise ValueError(
                    f"FDI {fdi} in {sample.annotation_path} is not defined by the "
                    f"{sample.source_view} class mapping."
                )
            unknown_fdi[fdi] += 1
            continue
        if not all(math.isfinite(value) for value in raw_box):
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
        if fdi in widened:
            cross_view[fdi] += 1

    return PreparedTargets(
        boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        class_ids=np.asarray(class_ids, dtype=np.float32).reshape(-1, 1),
        cross_view=cross_view,
        unknown_fdi=unknown_fdi,
    )


def summarize_samples(
    samples: Sequence[ToothSample],
    class_names: Mapping[int, str] | Sequence[str],
    unknown: str = "skip",
) -> dict[str, object]:
    """Validate all labels and summarize one split without writing files."""

    box_count = 0
    source_views: Counter[str] = Counter()
    cross_view: Counter[int] = Counter()
    unknown_fdi: Counter[int] = Counter()
    class_mappings = {"default": build_fdi_to_class(class_names)}
    if any(sample.source_view == "left" for sample in samples):
        class_mappings["left"] = validate_left_class_mapping(class_names)
    for sample in samples:
        with Image.open(sample.image_path) as image:
            image_size = image.size
        mapping = class_mappings.get(sample.source_view, class_mappings["default"])
        targets = prepare_targets(sample, mapping, image_size, unknown=unknown)
        box_count += len(targets.boxes)
        source_views[sample.source_view] += 1
        cross_view.update(targets.cross_view)
        unknown_fdi.update(targets.unknown_fdi)
    return {
        "images": len(samples),
        "boxes": box_count,
        "source_views": dict(sorted(source_views.items())),
        "cross_view_boxes": sum(cross_view.values()),
        "dropped_unknown_boxes": sum(unknown_fdi.values()),
        "dropped_unknown_fdi": dict(sorted(unknown_fdi.items())),
    }


class InMemoryToothYOLODataset(YOLODataset):
    """Ultralytics dataset backed directly by source images and JSON labels.

    No YOLO label files, image links, image copies, or flipped images are
    written. Left images and their boxes are flipped in memory on every load.

    Labels keep every FDI the checkpoint class map can express, including the
    cross-view teeth that ``segtooth.py::TARGET_TEETH_PER_VIEW`` would drop
    after inference; see ``prepare_targets``.
    """

    def __init__(
        self,
        *args,
        samples: Sequence[ToothSample],
        class_names: Mapping[int, str] | Sequence[str],
        unknown_fdi: str = "skip",
        **kwargs,
    ) -> None:
        if unknown_fdi not in UNKNOWN_FDI_POLICIES:
            raise ValueError(
                f"unknown_fdi must be one of {UNKNOWN_FDI_POLICIES}, got {unknown_fdi!r}"
            )
        self.unknown_fdi = unknown_fdi
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
        # train.py's build_dataset() forwards cache=False, which would collide
        # with the explicit cache=False below ("got multiple values for keyword
        # argument 'cache'"). Caching stays off either way, so accept and drop
        # whatever the caller asked for.
        kwargs.pop("cache", None)
        super().__init__(*args, cache=False, **kwargs)

    def get_img_files(self, img_path: str | list[str]) -> list[str]:
        del img_path
        image_files = [str(sample.image_path.resolve()) for sample in self.samples]
        if not image_files:
            raise FileNotFoundError("No source images were provided to the dataset")
        return image_files

    def get_labels(self) -> list[dict[str, object]]:
        """Convert the JSON annotations to Ultralytics label dictionaries."""

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
            targets = prepare_targets(
                sample,
                mapping,
                (width, height),
                unknown=self.unknown_fdi,
            )
            boxes, class_ids = targets.boxes, targets.class_ids

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
