# SPDX-License-Identifier: MIT
"""Honest check: score the fine-tuned vs frozen detectors on the VAL cases only.

`estimate_accuracy.py` reports over all 112 annotated cases, but 90 of them were
used to train the fine-tune, so the headline number is optimistic. `train.py`
splits by subject ID with ``val_ratio=0.2, seed=42`` and does not persist the
split, so it is reproduced here with the same function.

Usage (from anywhere; paths below are relative to the datasets root):
    cd .datasets
    PYTHONPATH=<repo> python <repo>/yolo_finetune/data/eval_holdout.py
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from yolo_finetune.dataset import discover_samples, split_samples  # noqa: E402
from yolo_finetune.data.estimate_accuracy import compare, load  # noqa: E402

VIEWS = ["D", "F", "L", "R", "U"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="intraoral")
    ap.add_argument("--annotation-root", default="intraoral_adjust/tooth_bbox_adj")
    ap.add_argument("--ft", default="intraoral_adjust/unfiltered_bbox_ft")
    ap.add_argument("--base", default="intraoral_adjust/unfiltered_bbox")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def val_subjects(args: argparse.Namespace) -> set[str]:
    """The union of the per-view val splits (each view splits its own subject list)."""
    ids: set[str] = set()
    for model_view in ("front", "upper", "lower", "right"):
        samples = discover_samples(
            data_root=args.data_root,
            model_view=model_view,
            annotation_root=args.annotation_root,
        )
        _, val = split_samples(samples, val_ratio=args.val_ratio, seed=args.seed)
        view_ids = {s.subject_id for s in val}
        print(f"  {model_view:<6} {len(samples):>4} imgs -> val {len(view_ids):>3} subjects")
        ids |= view_ids
    return ids


def score(pred_root: Path, adj_root: Path, keep: set[str], want: set[str]):
    """Aggregate compare() over the adj tree, restricted to `want` cases."""
    total = collections.Counter()
    per_view = collections.defaultdict(collections.Counter)
    n_files = 0
    for case_dir in sorted(p for p in adj_root.iterdir() if p.is_dir()):
        case = case_dir.name
        inside = case in keep
        if (inside and "in" not in want) or (not inside and "out" not in want):
            continue
        for adj_path in sorted(case_dir.glob("*.json")):
            view = adj_path.stem
            op = pred_root / case / f"{view}.json"
            if not pred_root.joinpath(case).is_dir() or not op.exists():
                continue
            n_files += 1
            counts, _ = compare(load(op), load(adj_path))
            for k, v in counts.items():
                total[k] += v
                per_view[view][k] += v
    return total, per_view, n_files


def row(name: str, c: collections.Counter) -> str:
    pred = c["n_pred"] or 1
    gt = c["n_gt"] or 1
    return ("%-5s pred=%-5d gt=%-5d correct=%-5d (%5.1f%% pred, %5.1f%% gt) "
            "loc_err=%-4d relabel=%-3d fp=%-4d miss=%-4d"
            % (name, c["n_pred"], c["n_gt"], c["correct"],
               100 * c["correct"] / pred, 100 * c["correct"] / gt,
               c["loc_err"], c["relabel"], c["fp"], c["miss"]))


def main() -> None:
    args = parse_args()
    print("reproducing the train.py split (val_ratio=%.2f seed=%d):"
          % (args.val_ratio, args.seed))
    val_ids = val_subjects(args)
    print(f"  union -> {len(val_ids)} of 112 subjects held out\n")

    for want, label in (({"in"}, "HELD-OUT VAL"), ({"out"}, "SEEN (TRAIN)")):
        print(f"===== {label} =====")
        for who, root in (("frozen", Path(args.base)), ("ft", Path(args.ft))):
            total, per_view, n = score(root, Path(args.annotation_root), val_ids, want)
            print(f"-- {who} ({n} files)")
            for v in VIEWS:
                if per_view[v]:
                    print("   " + row(v, per_view[v]))
            print("   " + row("ALL", total))
        print()


if __name__ == "__main__":
    os.chdir(os.environ.get("DATASETS_ROOT", "."))
    main()
