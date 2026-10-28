#!/usr/bin/env python3
"""Build a front-sextant gingivitis severity dataset from GingiCap YOLO labels.

GingiCap ships one YOLO ``.txt`` per intraoral photo with:

* class ``0``  -- the annotated **upper front sextant** region
* class ``1``  -- the annotated **lower front sextant** region
* classes ``2``--``6`` -- gingivitis severity tooth boxes

``segtooth.py`` maps a ``front`` view to two sextants, ``S2`` (upper anterior,
FDI 13--23) and ``S5`` (lower anterior, FDI 33--43), and crops them as the
union of the member tooth boxes plus directional gum padding.  GingiCap does
not name FDI teeth, so this script uses the class-0/1 region boxes only to
assign each tooth box to ``S2``/``S5`` (tooth-box centre inside the region),
which reproduces the same upper/lower front split.

The sextant label is the **maximum** tooth class inside the region, mapped to
the 3-class severity scheme used by the rest of this repo:

    2, 3 -> mild          (class_id 0)
    4    -> moderate      (class_id 1)
    5, 6 -> severe        (class_id 2)

Output layout::

    <output>/
      training_sextant/
        Images/<sample_id>/S2.jpg
        Images/<sample_id>/S5.jpg
        labels.json
      validation_sextant/...
      test_sextant/...
      distribution.json
      labels.csv

``Images/<sample_id>/{S2,S5}.jpg`` are the two front-sextant crops of the
original ``<split>/Images/<sample_id>.jpg``.  ``labels.json`` records the label
of every crop under the same split.

Usage::

    python process_public/gingicap/build_sextant_dataset.py --overwrite
    python process_public/gingicap/build_sextant_dataset.py --stats_only
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_SPLITS: Dict[str, str] = {
    "training": "Training",
    "validation": "Validation",
    "test": "Test",
}
OUTPUT_SPLITS: Dict[str, str] = {
    "training": "training_sextant",
    "validation": "validation_sextant",
    "test": "test_sextant",
}

# Source classes 0/1 are the annotated upper/lower front sextant regions.
# They are used only to assign tooth boxes to a sextant, never as a label.
REGION_TO_SEXTANT: Dict[int, str] = {0: "S2", 1: "S5"}
SEXTANT_DESCRIPTIONS: Dict[str, str] = {
    "S2": "upper anterior front sextant (source region class 0)",
    "S5": "lower anterior front sextant (source region class 1)",
}

# Severity tooth classes 2-6 -> 3 labels.  Identical to the repo's COCO export
# (gingivitis_{train,test,coco}.json categories mild/moderate/severe).
CLASS_TO_LABEL: Dict[int, str] = {
    2: "mild",
    3: "mild",
    4: "moderate",
    5: "severe",
    6: "severe",
}
LABEL_TO_CLASS_ID: Dict[str, int] = {"mild": 0, "moderate": 1, "severe": 2}
LABELS: Tuple[str, ...] = ("mild", "moderate", "severe")

# Padding constants copied from segtooth.py (front view, no posterior extra).
GUM_PADDING_RATIO = 0.4
SIDE_PADDING_RATIO = 0.1
UPPER_SMALL_OPPOSITE_GUM_RATIO = 0.05
LOWER_SMALL_OPPOSITE_GUM_RATIO = 0.05

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG")

Row = Tuple[int, float, float, float, float]  # cls, cx, cy, w, h (normalised)


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a two-sextant (S2/S5) severity dataset from GingiCap.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(".datasets/intraoral_public/gingicap"),
        help="GingiCap root containing Training/Validation/Test.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".datasets/intraoral_public/gingicap_sextant"),
        help="Output dataset root.",
    )
    parser.add_argument(
        "--crop_mode",
        choices=("segtooth", "region"),
        default="segtooth",
        help=(
            "'segtooth': union of member tooth boxes + segtooth.py directional "
            "padding. 'region': use the annotated class-0/1 sextant box."
        ),
    )
    parser.add_argument(
        "--jpeg_quality", type=int, default=95, help="Saved JPEG quality."
    )
    parser.add_argument(
        "--max_side",
        type=int,
        default=0,
        help="Downscale crops so the longer side is at most this (0 = keep native size).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Remove and recreate the output root."
    )
    parser.add_argument(
        "--stats_only",
        action="store_true",
        help="Only recompute the distribution from existing labels.json files.",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Process at most N samples per split (0 = all)."
    )
    return parser.parse_args()


def read_yolo_label(path: Path) -> List[Row]:
    rows: List[Row] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 fields, got {len(fields)}")
        cls, cx, cy, width, height = fields
        rows.append((int(cls), float(cx), float(cy), float(width), float(height)))
    return rows


def box_edges(row: Row) -> Tuple[float, float, float, float]:
    """Return normalised (x1, y1, x2, y2) for a YOLO row."""
    _, cx, cy, width, height = row
    return cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2


def find_image(images_dir: Path, sample_id: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{sample_id}{extension}"
        if candidate.exists():
            return candidate
    return None


def assign_teeth_to_sextants(
    rows: Sequence[Row],
) -> Tuple[Dict[str, List[Tuple[Row, int]]], Dict[int, Row]]:
    """Assign each severity tooth box to S2/S5 by its centre.

    Returns ``(assignments, region_boxes)`` where ``assignments`` maps a
    sextant to ``(tooth_row, source_class)`` members.
    """
    region_boxes = {row[0]: row for row in rows if row[0] in REGION_TO_SEXTANT}
    assignments: Dict[str, List[Tuple[Row, int]]] = {name: [] for name in REGION_TO_SEXTANT.values()}

    for row in rows:
        cls = row[0]
        if cls not in CLASS_TO_LABEL:
            continue
        cx, cy = row[1], row[2]
        for region_class, sextant in REGION_TO_SEXTANT.items():
            region = region_boxes.get(region_class)
            if region is None:
                continue
            x1, y1, x2, y2 = box_edges(region)
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                assignments[sextant].append((row, cls))
    return assignments, region_boxes


def union_box_pixels(members: Iterable[Row], img_w: int, img_h: int) -> List[float]:
    boxes = [box_edges(row) for row in members]
    x1 = min(box[0] for box in boxes) * img_w
    y1 = min(box[1] for box in boxes) * img_h
    x2 = max(box[2] for box in boxes) * img_w
    y2 = max(box[3] for box in boxes) * img_h
    return [x1, y1, x2, y2]


def clamp_int_box(values: Sequence[float], img_w: int, img_h: int) -> List[int]:
    x1, y1, x2, y2 = (int(value) for value in values)
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(x1 + 1, min(img_w, x2))
    y2 = max(y1 + 1, min(img_h, y2))
    return [x1, y1, x2, y2]


def pad_sextant_box(
    union_box: Sequence[float], sextant: str, img_w: int, img_h: int
) -> List[int]:
    """Replicate segtooth.py's front-view directional sextant padding."""
    x1, y1, x2, y2 = union_box
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)

    is_upper = sextant in {"S1", "S2", "S3"}
    if is_upper:
        pad_top = box_h * GUM_PADDING_RATIO
        pad_bottom = box_h * UPPER_SMALL_OPPOSITE_GUM_RATIO
    else:
        pad_top = box_h * LOWER_SMALL_OPPOSITE_GUM_RATIO
        pad_bottom = box_h * GUM_PADDING_RATIO

    pad_left = box_w * SIDE_PADDING_RATIO
    pad_right = box_w * SIDE_PADDING_RATIO

    return clamp_int_box(
        (x1 - pad_left, y1 - pad_top, x2 + pad_right, y2 + pad_bottom),
        img_w,
        img_h,
    )


def region_box_pixels(region: Row, img_w: int, img_h: int) -> List[int]:
    x1, y1, x2, y2 = box_edges(region)
    return clamp_int_box((x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h), img_w, img_h)


# ---------------------------------------------------------------------------
# Cropping / saving
# ---------------------------------------------------------------------------


def save_jpeg_crop(
    image: Image.Image,
    box: Sequence[int],
    destination: Path,
    quality: int,
    max_side: int,
) -> Tuple[int, int]:
    crop = image.crop(tuple(box))
    if max_side and max(crop.size) > max_side:
        scale = max_side / max(crop.size)
        new_size = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
        crop = crop.resize(new_size, Image.LANCZOS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    crop.save(destination, format="JPEG", quality=quality)
    return crop.size


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------


def process_split(
    source_split_dir: Path,
    output_split_dir: Path,
    crop_mode: str,
    jpeg_quality: int,
    max_side: int,
    limit: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    images_dir = source_split_dir / "Images"
    labels_dir = source_split_dir / "Labels"
    label_paths = sorted(labels_dir.glob("*.txt"))
    if limit:
        label_paths = label_paths[:limit]

    samples: Dict[str, Any] = {}
    skipped: Counter = Counter()
    rows_csv: List[Dict[str, Any]] = []

    for label_path in tqdm(label_paths, desc=f"{source_split_dir.name}", unit="img"):
        sample_id = label_path.stem
        image_path = find_image(images_dir, sample_id)
        if image_path is None:
            skipped["missing_image"] += 1
            continue

        rows = read_yolo_label(label_path)
        assignments, region_boxes = assign_teeth_to_sextants(rows)
        if not any(assignments.values()):
            skipped["no_assigned_teeth"] += 1
            continue

        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        img_w, img_h = image.size

        sample_record: Dict[str, Any] = {}
        for sextant in ("S2", "S5"):
            members = assignments[sextant]
            if not members:
                skipped[f"empty_{sextant}"] += 1
                continue

            source_classes = sorted(cls for _, cls in members)
            max_source_class = max(source_classes)
            label = CLASS_TO_LABEL[max_source_class]

            if crop_mode == "region":
                region = region_boxes.get(0 if sextant == "S2" else 1)
                if region is None:
                    skipped[f"missing_{sextant}_region"] += 1
                    continue
                crop_box = region_box_pixels(region, img_w, img_h)
                crop_source = "region_box"
            else:
                union_box = union_box_pixels((row for row, _ in members), img_w, img_h)
                crop_box = pad_sextant_box(union_box, sextant, img_w, img_h)
                crop_source = "tooth_union+padding"

            image_name = f"{sextant}.jpg"
            destination = output_split_dir / "Images" / sample_id / image_name
            crop_size = save_jpeg_crop(image, crop_box, destination, jpeg_quality, max_side)

            sample_record[sextant] = {
                "image": f"Images/{sample_id}/{image_name}",
                "label": label,
                "class_id": LABEL_TO_CLASS_ID[label],
                "max_source_class": max_source_class,
                "source_classes": source_classes,
                "source_class_counts": dict(sorted(Counter(source_classes).items())),
                "num_teeth": len(source_classes),
                "crop_source": crop_source,
                "crop_box_xyxy": [int(value) for value in crop_box],
                "crop_size_wh": [int(crop_size[0]), int(crop_size[1])],
            }
            rows_csv.append(
                {
                    "split": output_split_dir.name,
                    "sample_id": sample_id,
                    "sextant": sextant,
                    "image": f"{output_split_dir.name}/Images/{sample_id}/{image_name}",
                    "label": label,
                    "class_id": LABEL_TO_CLASS_ID[label],
                    "max_source_class": max_source_class,
                    "source_classes": " ".join(str(c) for c in source_classes),
                    "num_teeth": len(source_classes),
                }
            )

        image.close()
        if sample_record:
            samples[sample_id] = sample_record
        else:
            skipped["no_sextant_record"] += 1

    distribution = Counter(
        record["label"] for record in samples.values() for record in record.values()
    )
    max_class_distribution = Counter(
        record["max_source_class"] for record in samples.values() for record in record.values()
    )
    per_sextant = {
        sextant: Counter(
            record[sextant]["label"]
            for record in samples.values()
            if sextant in record
        )
        for sextant in ("S2", "S5")
    }

    labels = {
        "source_split": source_split_dir.name,
        "crop_mode": crop_mode,
        "label_mapping": {str(cls): label for cls, label in CLASS_TO_LABEL.items()},
        "labels": {label: LABEL_TO_CLASS_ID[label] for label in LABELS},
        "sextants": SEXTANT_DESCRIPTIONS,
        "aggregation": "max severity tooth class inside the sextant",
        "num_samples": len(samples),
        "num_images": sum(len(record) for record in samples.values()),
        "distribution": {label: distribution.get(label, 0) for label in LABELS},
        "samples": samples,
    }
    split_report = {
        "source_split": source_split_dir.name,
        "num_samples": len(samples),
        "num_images": labels["num_images"],
        "label_distribution": {label: distribution.get(label, 0) for label in LABELS},
        "label_percent": {
            label: round(distribution.get(label, 0) / max(1, labels["num_images"]), 4)
            for label in LABELS
        },
        "per_sextant": {
            sextant: {label: counts.get(label, 0) for label in LABELS}
            for sextant, counts in per_sextant.items()
        },
        "max_source_class_distribution": {
            str(cls): max_class_distribution.get(cls, 0) for cls in sorted(CLASS_TO_LABEL)
        },
        "skipped": dict(sorted(skipped.items())),
    }
    return labels, split_report, rows_csv


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Stats-only mode
# ---------------------------------------------------------------------------


def summarize_existing(output: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {"output": str(output), "splits": {}}
    totals: Counter = Counter()
    for split_name, folder in OUTPUT_SPLITS.items():
        labels_path = output / folder / "labels.json"
        if not labels_path.exists():
            report["splits"][split_name] = {"error": "missing labels.json"}
            continue
        data = json.loads(labels_path.read_text(encoding="utf-8"))
        distribution = Counter(data.get("distribution", {}))
        num_images = int(data.get("num_images", 0))
        totals.update(distribution)
        report["splits"][split_name] = {
            "num_samples": data.get("num_samples"),
            "num_images": num_images,
            "label_distribution": {label: distribution.get(label, 0) for label in LABELS},
            "label_percent": {
                label: round(distribution.get(label, 0) / max(1, num_images), 4)
                for label in LABELS
            },
        }
    grand_total = sum(totals.values())
    report["overall"] = {
        "num_images": grand_total,
        "label_distribution": {label: totals.get(label, 0) for label in LABELS},
        "label_percent": {
            label: round(totals.get(label, 0) / max(1, grand_total), 4) for label in LABELS
        },
    }
    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    source_root = args.source
    output_root = args.output

    if args.stats_only:
        print(json.dumps(summarize_existing(output_root), ensure_ascii=False, indent=2))
        return

    if not (source_root / "Training" / "Labels").exists():
        raise FileNotFoundError(f"source not found: {source_root}")
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite to recreate it")
        shutil.rmtree(output_root)

    report: Dict[str, Any] = {
        "source": str(source_root),
        "output": str(output_root),
        "crop_mode": args.crop_mode,
        "label_mapping": {str(cls): label for cls, label in CLASS_TO_LABEL.items()},
        "aggregation": "max severity tooth class inside each front sextant",
        "splits": {},
    }
    all_csv_rows: List[Dict[str, Any]] = []
    totals: Counter = Counter()

    for split_name, source_split_dir in SOURCE_SPLITS.items():
        labels, split_report, csv_rows = process_split(
            source_root / source_split_dir,
            output_root / OUTPUT_SPLITS[split_name],
            crop_mode=args.crop_mode,
            jpeg_quality=args.jpeg_quality,
            max_side=args.max_side,
            limit=args.limit,
        )
        split_output = output_root / OUTPUT_SPLITS[split_name]
        split_output.mkdir(parents=True, exist_ok=True)
        (split_output / "labels.json").write_text(
            json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report["splits"][split_name] = split_report
        all_csv_rows.extend(csv_rows)
        totals.update(Counter(labels["distribution"]))

    grand_total = sum(totals.values())
    report["overall"] = {
        "num_samples": sum(v["num_samples"] for v in report["splits"].values()),
        "num_images": grand_total,
        "label_distribution": {label: totals.get(label, 0) for label in LABELS},
        "label_percent": {
            label: round(totals.get(label, 0) / max(1, grand_total), 4) for label in LABELS
        },
    }

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "distribution.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(output_root / "labels.csv", all_csv_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
