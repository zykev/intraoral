# SPDX-License-Identifier: MIT
"""Build the *prediction* side of the YOLO fine-tuning labels.

Re-runs the SegmentAnyTooth YOLO detector WITHOUT the per-view FDI filter.

`segtooth.py` / `tmp_process.py` apply `TARGET_TEETH_PER_VIEW` inside
`build_tooth_records()`, so `tooth_bbox/<case>/<view>.json` only ever contains
the teeth that belong to that view.  The *unfiltered* detections are what the
`preview/` PNGs were drawn from, and they are needed to judge whether the
annotator's corrections apply to a prediction that actually exists.

This script re-runs the exact same detection path (`predict_boxes`) on
`process/<case>/<view>.png` and writes every detected box to
`intraoral_adjust/unfiltered_bbox_ft/<case>/<view>.json`, keeping the same
schema as `tooth_bbox` so downstream scripts can read either one.
`build_tooth_bbox_adj.py` then applies the manual corrections on top.

Two output trees, on purpose
----------------------------
* ``unfiltered_bbox``    -- predictions of the *original* checkpoints. It is the
  base ``build_tooth_bbox_adj.py`` reads to synthesize the manual corrections,
  and it is the fixed baseline ``estimate_accuracy.py`` compares against, so it
  is treated as frozen and this script refuses to write into it.
* ``unfiltered_bbox_ft`` -- default ``--out``. Use it for re-runs and for
  predictions made with the fine-tuned checkpoints, so the baseline above
  survives to be compared against.

Sanity check performed by the caller: filtering the output with
`TARGET_TEETH_PER_VIEW` should reproduce `tooth_bbox` exactly.

Run with cwd = the datasets root (the repo's ``.datasets/`` symlink directory,
or /disk1/work/zychen/Datasets), using the ``intraoral`` conda env (needs
torch / ultralytics).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# `segtooth.py` lives at the repository root, two levels up from this file.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segtooth import (
    DEFAULT_WEIGHT_DIR,
    VIEW_MAPPING,
    clamp_int_box,
    predict_boxes,
    rel_path,
    valid_box,
)

# Predictions of the original checkpoints: the base the GT is built from and the
# fixed baseline the accuracy report compares against.
BASELINE_OUT_NAME = "unfiltered_bbox"

# Where ``yolo_finetune/train.py`` exports its fine-tuned weights (its default
# ``--project exp``). Derived from REPO_ROOT instead of hard-coded so it still
# resolves when the repo is moved; on this machine it is
# /home/zychen/Documents/intraoral_code/intraoral/exp/segtooth_model.
FINETUNE_WEIGHT_DIR = REPO_ROOT / "exp" / "segtooth_model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run YOLO on process images without the per-view FDI filter."
    )
    parser.add_argument("--base_dir", type=Path, default=Path("intraoral/zeyu"),
                        help="Directory containing <batch>_process folders that hold "
                             "process/ images. Default: intraoral/zeyu")
    parser.add_argument("--out", type=Path, default=Path("intraoral_adjust/unfiltered_bbox_ft"),
                        help="Output directory for <out>/<case>/<view>.json. "
                             "Default: intraoral_adjust/unfiltered_bbox_ft, which "
                             "keeps the frozen baseline in unfiltered_bbox/ intact.")
    parser.add_argument("--force", action="store_true",
                        help="Allow --out to be the frozen baseline tree "
                             "intraoral_adjust/unfiltered_bbox. Only do this when "
                             "regenerating the baseline on purpose, because "
                             "build_tooth_bbox_adj.py and estimate_accuracy.py "
                             "both read that tree.")
    parser.add_argument("--weight_dir", type=Path, default=FINETUNE_WEIGHT_DIR,
                        help="Directory containing SegmentAnyTooth YOLO weights. "
                             "Default: <repo>/exp/segtooth_model, i.e. the export of "
                             "a finished fine-tune run. Pass "
                             "<repo>/.checkpoints/segtooth_model to score the "
                             "original (pre-fine-tune) checkpoints instead.")
    parser.add_argument("--min_conf", type=float, default=0.35)
    parser.add_argument("--min_area", type=int, default=500)
    parser.add_argument("--cases", nargs="*", default=None,
                        help="Optional case-id whitelist (debugging).")
    parser.add_argument("--views", nargs="*", default=None,
                        help="Optional view whitelist (debugging).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.base_dir = Path(args.base_dir).expanduser().resolve()
    args.out = Path(args.out).expanduser().resolve()
    args.weight_dir = Path(args.weight_dir).expanduser().resolve()
    if not args.weight_dir.exists():
        # The default only appears once a fine-tune run has exported into it, so
        # the plain "does not exist" is not enough to act on here.
        hint = (
            "\nThis is the default location, which is created by "
            "`python -m yolo_finetune.train`. Fine-tune a view first, or pass "
            f"--weight_dir {DEFAULT_WEIGHT_DIR} to score the original checkpoints."
            if args.weight_dir == FINETUNE_WEIGHT_DIR.resolve()
            else ""
        )
        raise FileNotFoundError(f"weight_dir does not exist: {args.weight_dir}{hint}")

    # `unfiltered_bbox` is not a scratch directory: build_tooth_bbox_adj.py reads
    # it to synthesize the GT, and estimate_accuracy.py uses it as the frozen
    # comparison baseline. Overwriting it silently would destroy both.
    if args.out.name == BASELINE_OUT_NAME and args.out.exists() and not args.force:
        raise SystemExit(
            f"refusing to write into the frozen baseline tree:\n  {args.out}\n"
            "build_tooth_bbox_adj.py reads it to build the GT and "
            "estimate_accuracy.py uses it as the fixed baseline.\n"
            "Pass a different --out (the default "
            "intraoral_adjust/unfiltered_bbox_ft is safe), or --force if you "
            "really mean to regenerate the baseline."
        )

    batches = sorted(
        d for d in args.base_dir.iterdir()
        if d.is_dir() and d.name.endswith("_process")
    )
    if not batches:
        raise FileNotFoundError(f"no '*_process' folders under {args.base_dir}")

    n_written = 0
    for date_path in batches:
        input_root = date_path / "process"
        if not input_root.exists():
            continue
        for case_dir in sorted(d for d in input_root.iterdir() if d.is_dir()):
            case_id = case_dir.name
            if args.cases and case_id not in args.cases:
                continue
            for stem, view_type, image_path in _iter_views(case_dir, args.views):
                boxes_data = predict_boxes(
                    image_path=image_path,
                    view=view_type,
                    weight_dir=args.weight_dir,
                    min_conf=args.min_conf,
                    min_area=args.min_area,
                )
                records, orig_size = _records(boxes_data, date_path, case_id, stem, image_path)
                out_path = args.out / case_id / f"{stem}.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "case_id": case_id,
                            "view": stem,
                            "view_type": view_type,
                            "image_path": rel_path(image_path, date_path),
                            "orig_size": orig_size,
                            "teeth": records,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                n_written += 1
                print(f"{case_id}/{stem}: {len(records)} boxes")
    print(f"done: {n_written} file(s) -> {args.out}")


def _iter_views(case_dir: Path, want: List[str] | None):
    """Yield (stem, view_type, image_path) for the views present in `case_dir`."""
    for stem, view_type in VIEW_MAPPING.items():
        if want and stem not in want:
            continue
        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
            image_path = case_dir / f"{stem}{ext}"
            if image_path.exists():
                yield stem, view_type, image_path
                break


def _records(
    boxes_data: List[Dict[str, Any]],
    date_path: Path,
    case_id: str,
    view_stem: str,
    image_path: Path,
):
    """Convert raw detection dicts into tooth_bbox-style records (no FDI filter).

    Returns ``(records, [width, height])`` of the source ``process`` image.
    """
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    img_h, img_w = image.shape[:2]

    records: List[Dict[str, Any]] = []
    for item in boxes_data:
        fdi = int(item["fdi_tooth_name"])
        crop_box, clamped = clamp_int_box(item["box"], img_w, img_h)
        if not valid_box(crop_box):
            continue
        crop_path = date_path / "single_tooth" / case_id / view_stem / f"{fdi}.png"
        records.append({
            "fdi": fdi,
            "bbox_yolo": item["box"],
            "confidence": item["confidence"],
            "pad_ratio": 0.0,
            "bbox_padded": crop_box,
            "clamped": clamped,
            "crop_path": rel_path(crop_path, date_path),
        })
    return records, [img_w, img_h]


if __name__ == "__main__":
    main()
