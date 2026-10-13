#!/usr/bin/env python3
"""Run the fine-tuned detectors on a batch and visualize where they disagree.

Where the "before" comes from
-----------------------------
``preview/<case>/<view>.png`` is drawn by ``save_preview_original_style`` from the
**raw, unfiltered** detections: ``process_view`` passes ``boxes_data`` straight
from ``predict_boxes`` (before ``build_tooth_records`` applies
``TARGET_TEETH_PER_VIEW``). So the yellow rectangles in ``preview/`` are exactly
what ``build_tooth_bbox_predict.py`` recomputes when it is pointed at the
ORIGINAL checkpoints.

That is why this script does not try to parse the preview PNGs: it re-runs the
same detection path twice on the same images,

    before = <repo>/.checkpoints/segtooth_model   (the original checkpoints)
    after  = <ft-weight-dir>                      (default: exp/segtooth_model)

and diffs the two JSON sets, which is both exact and comparable. Step 1 below
additionally checks that "before" still reproduces the batch's stored
``tooth_bbox/`` (the view-filtered subset of the same detections), so you can
see whether the preview you are comparing against is still current.

What counts as a difference
---------------------------
Per (case, view), boxes are keyed by FDI (``predict_boxes`` keeps one box per
FDI, so there are no duplicates). Same FDI in both -> IoU decides:

    same     IoU >= 0.9                      (nothing changed)
    moved    same FDI, IoU <  0.9            (same tooth, box shifted)
    relabel  leftover pair, IoU >= 0.3, FDI differs
    removed  only in the BEFORE set
    added    only in the AFTER set

``score = 3*(added + removed + relabel) + moved`` ranks the samples;
``churn = score / n_before`` is the same number normalized. Structural changes
(add/remove/relabel) are weighted 3x because a moved box usually still crops the
same tooth, while a missing or mislabelled one does not.

Usage
-----
    cd <repo>/.datasets
    PYTHONPATH=<repo> python <repo>/yolo_finetune/data/compare_predictions.py \\
        --batch intraoral/henry/20251217_process

    # only the views that actually improved (see eval_holdout.py); the rest are
    # taken from the ORIGINAL checkpoints so the diff is a true zero there
    ... --ft-views U D

Outputs ``--out/index.html`` plus one PNG per rendered sample, a per-sample CSV
and the raw diff JSON. Open the HTML to page through the worst offenders.

Only stdlib + opencv + matplotlib. SAM is never loaded: this compares boxes.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segtooth import TARGET_TEETH_PER_VIEW, VIEW_MAPPING  # noqa: E402

DATA_DIR = REPO_ROOT / "yolo_finetune" / "data"
DATASETS_ROOT = Path(os.environ.get("DATASETS_ROOT", REPO_ROOT / ".datasets"))
ORIGINAL_WEIGHT_DIR = REPO_ROOT / ".checkpoints" / "segtooth_model"
FINETUNE_WEIGHT_DIR = REPO_ROOT / "exp" / "segtooth_model"

STEMS = list(VIEW_MAPPING)  # ["F", "U", "D", "L", "R"]

COLORS = {
    "same": "#00c853",
    "moved": "#ff9100",
    "relabel": "#aa00ff",
    "removed": "#ff1744",
    "added": "#2979ff",
}
LEGEND = [
    ("same", "same FDI, IoU>=0.9"),
    ("moved", "same FDI, box moved"),
    ("relabel", "box kept, FDI changed"),
    ("removed", "only in BEFORE"),
    ("added", "only in AFTER"),
]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def resolve_input(path: Path) -> Path:
    """Relative inputs are tried against the cwd first, then the datasets root.

    The sibling scripts (build_tooth_bbox_predict.py, estimate_accuracy.py) only
    work from inside .datasets because they resolve their arguments against the
    cwd. This one accepts either, so it can be launched from the repo root.
    """
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    if path.exists():
        return path.resolve()
    return (DATASETS_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--batch", type=Path, required=True,
                    help="Either a <date>_process folder (the usual case) or a parent "
                         "folder holding exactly one of them.")
    ap.add_argument("--ft-weight-dir", type=Path, default=FINETUNE_WEIGHT_DIR,
                    help="Fine-tuned checkpoints. Default: <repo>/exp/segtooth_model, "
                         "where train.py exports.")
    ap.add_argument("--orig-weight-dir", type=Path, default=ORIGINAL_WEIGHT_DIR,
                    help="Pre-fine-tune checkpoints used as the 'before' side. "
                         "Default: <repo>/.checkpoints/segtooth_model.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output dir (default: <repo>/exp/cmp_<owner>_<date>). "
                         "Contains before/, after/, report.csv, diff.json, index.html.")
    ap.add_argument("--ft-views", nargs="*", default=STEMS, choices=STEMS,
                    help="Which views use the fine-tuned checkpoint. Views left out are "
                         "scored with the original checkpoint, so their diff is exactly "
                         "zero. Use this to deploy only the views that improved.")
    ap.add_argument("--top", type=int, default=25,
                    help="Render this many worst samples. Default 25.")
    ap.add_argument("--min-score", type=int, default=1,
                    help="Skip samples scoring below this. Default 1 (any difference).")
    ap.add_argument("--render-mode", choices=("crop", "full"), default="crop",
                    help="crop: zoom to the union of all boxes (default, readable at "
                         "3380x2628). full: whole image.")
    ap.add_argument("--max-panel-px", type=int, default=1300,
                    help="Cap on each panel's long side in the output PNG. Default 1300.")
    ap.add_argument("--iou-ok", type=float, default=0.9,
                    help="Above this a same-FDI box counts as unchanged. Default 0.9.")
    ap.add_argument("--iou-match", type=float, default=0.3,
                    help="Minimum IoU to pair leftovers as a relabel. Default 0.3.")
    ap.add_argument("--cases", nargs="*", default=None,
                    help="Optional case-id whitelist (debugging).")
    ap.add_argument("--skip-detect", action="store_true",
                    help="Reuse the before/after JSON already under --out.")
    ap.add_argument("--skip-verify", action="store_true",
                    help="Skip step 1, the check of 'before' against stored tooth_bbox.")
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# geometry / io
# --------------------------------------------------------------------------- #
def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def load_tree(root: Path) -> Dict[Tuple[str, str], Dict[int, Dict[str, Any]]]:
    """<root>/<case>/<view>.json -> {(case, view): {fdi: {box, conf}}}."""
    tree: Dict[Tuple[str, str], Dict[int, Dict[str, Any]]] = {}
    if not root.is_dir():
        return tree
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(case_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            tree[(case_dir.name, path.stem)] = {
                int(t["fdi"]): {"box": [float(v) for v in t["bbox_yolo"]],
                                "conf": float(t.get("confidence", 0.0))}
                for t in payload.get("teeth", [])
            }
    return tree


# --------------------------------------------------------------------------- #
# diff
# --------------------------------------------------------------------------- #
def diff_view(
    before: Dict[int, Dict[str, Any]],
    after: Dict[int, Dict[str, Any]],
    iou_ok: float,
    iou_match: float,
) -> List[Dict[str, Any]]:
    """Return one record per paired/unpaired box: kind, fdi, iou."""
    out: List[Dict[str, Any]] = []
    b, a = dict(before), dict(after)

    for fdi in sorted(b):
        if fdi in a:
            v = iou(b[fdi]["box"], a[fdi]["box"])
            out.append({"kind": "same" if v >= iou_ok else "moved",
                        "fdi_before": fdi, "fdi_after": fdi, "iou": round(v, 4),
                        "conf_before": round(b[fdi]["conf"], 4),
                        "conf_after": round(a[fdi]["conf"], 4)})
            b.pop(fdi)
            a.pop(fdi)

    candidates = []
    for bfdi, bbox in b.items():
        for afdi, abox in a.items():
            v = iou(bbox["box"], abox["box"])
            if v >= iou_match:
                candidates.append((v, bfdi, afdi))
    candidates.sort(reverse=True)

    used_b, used_a = set(), set()
    for v, bfdi, afdi in candidates:
        if bfdi in used_b or afdi in used_a:
            continue
        used_b.add(bfdi)
        used_a.add(afdi)
        out.append({"kind": "relabel", "fdi_before": bfdi, "fdi_after": afdi,
                    "iou": round(v, 4),
                    "conf_before": round(b[bfdi]["conf"], 4),
                    "conf_after": round(a[afdi]["conf"], 4)})

    for fdi in sorted(set(b) - used_b):
        out.append({"kind": "removed", "fdi_before": fdi, "fdi_after": None,
                    "iou": 0.0, "conf_before": round(b[fdi]["conf"], 4),
                    "conf_after": None})
    for fdi in sorted(set(a) - used_a):
        out.append({"kind": "added", "fdi_before": None, "fdi_after": fdi,
                    "iou": 0.0, "conf_before": None,
                    "conf_after": round(a[fdi]["conf"], 4)})
    return out


def summarize(records: List[Dict[str, Any]], n_before: int) -> Dict[str, Any]:
    counts = {k: 0 for k in COLORS}
    for rec in records:
        counts[rec["kind"]] += 1
    score = 3 * (counts["added"] + counts["removed"] + counts["relabel"]) + counts["moved"]
    return {
        "n_before": n_before,
        "n_after": n_before + counts["added"] - counts["removed"],
        "score": score,
        "churn": round(score / max(1, n_before), 4),
        "same": counts["same"],
        "moved": counts["moved"],
        "relabel": counts["relabel"],
        "removed": counts["removed"],
        "added": counts["added"],
        "mean_iou_moved": round(
            sum(r["iou"] for r in records if r["kind"] == "moved")
            / max(1, counts["moved"]), 4),
    }


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
def run_detect(batch_parent: Path, out_tree: Path, weight_dir: Path,
               case_ids: List[str], views: Optional[List[str]]) -> None:
    cmd = [
        sys.executable, str(DATA_DIR / "build_tooth_bbox_predict.py"),
        "--base_dir", str(batch_parent),
        "--out", str(out_tree),
        "--weight_dir", str(weight_dir),
        "--cases", *case_ids,
    ]
    if views is not None:
        cmd += ["--views", *views]
    print("  $ " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


# --------------------------------------------------------------------------- #
# step 1: is "before" still the same detector that drew preview/ ?
# --------------------------------------------------------------------------- #
def verify_against_stored(batch: Path, before: Dict, case_ids: Iterable[str],
                          out_dir: Path) -> None:
    rows = []
    per_view: Dict[str, List[int]] = {s: [0, 0, 0, 0, 0] for s in STEMS}
    for case in case_ids:
        for stem, view_type in VIEW_MAPPING.items():
            stored_path = batch / "tooth_bbox" / case / f"{stem}.json"
            if not stored_path.exists():
                continue
            with stored_path.open("r", encoding="utf-8") as fh:
                stored = {int(t["fdi"]): t["bbox_yolo"]
                          for t in json.load(fh).get("teeth", [])}
            # tooth_bbox is the view-filtered subset of the same detections
            keep = set(TARGET_TEETH_PER_VIEW[view_type])
            current = {f: v["box"] for f, v in before.get((case, stem), {}).items()
                       if f in keep}
            common = set(stored) & set(current)
            matched = sum(1 for f in common if iou(stored[f], current[f]) >= 0.99)
            row = {
                "case": case, "view": stem,
                "stored": len(stored), "before_kept": len(current),
                "fdi_common": len(common), "box_match": matched,
                "only_stored": len(set(stored) - set(current)),
                "only_before": len(set(current) - set(stored)),
            }
            rows.append(row)
            acc = per_view[stem]
            acc[0] += len(stored)
            acc[1] += len(current)
            acc[2] += len(common)
            acc[3] += matched
            acc[4] += len(set(stored) - set(current))

    if not rows:
        print("  no tooth_bbox/ found in this batch, skipping")
        return

    with (out_dir / "verify_vs_tooth_bbox.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("  view   stored  before  common  exact-box  only_stored   agreement")
    for stem in STEMS:
        stored, curr, common, matched, only_stored = per_view[stem]
        if not stored:
            continue
        print("  %-5s %7d %7d %7d %10d %12d   %6.1f%%"
              % (stem, stored, curr, common, matched, only_stored,
                 100 * matched / max(1, common)))
    total = [sum(x[i] for x in per_view.values()) for i in range(5)]
    print("  ALL   %7d %7d %7d %10d %12d   %6.1f%%"
          % (*total, 100 * total[3] / max(1, total[2])))
    print("  -> 100% means the original checkpoints still reproduce preview/ "
          "exactly; a low number means the stored preview is stale.")


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def render(row: Dict[str, Any], records: List[Dict[str, Any]], image_path: Path,
           dest: Path, mode: str, max_panel_px: int) -> None:
    import cv2
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w = rgb.shape[:2]

    before = {r["fdi_before"]: r["kind"] for r in records if r["fdi_before"] is not None}
    after = {r["fdi_after"]: r["kind"] for r in records if r["fdi_after"] is not None}

    if mode == "crop":
        xs: List[float] = []
        ys: List[float] = []
        for r in records:
            for side in ("before", "after"):
                payload = row.get(f"_box_{side}", {})
                if r[f"fdi_{side}"] in payload:
                    x1, y1, x2, y2 = payload[r[f"fdi_{side}"]]
                    xs += [x1, x2]
                    ys += [y1, y2]
        if xs:
            pad = 0.10 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
            cx1 = max(0.0, min(xs) - pad)
            cy1 = max(0.0, min(ys) - pad)
            cx2 = min(float(img_w), max(xs) + pad)
            cy2 = min(float(img_h), max(ys) + pad)
        else:
            cx1, cy1, cx2, cy2 = 0.0, 0.0, float(img_w), float(img_h)
    else:
        cx1, cy1, cx2, cy2 = 0.0, 0.0, float(img_w), float(img_h)

    cw = max(1.0, cx2 - cx1)
    ch = max(1.0, cy2 - cy1)
    scale = min(1.0, max_panel_px / max(cw, ch))
    dpi = 100
    fig, axes = plt.subplots(1, 2, figsize=(2 * cw * scale / dpi, ch * scale / dpi), dpi=dpi)
    line = max(1.5, 3.0 * scale * 2)
    font = max(7, 15 * scale * 2)

    for ax, side, label in ((axes[0], "before", "BEFORE  (original checkpoint)"),
                            (axes[1], "after", "AFTER   (fine-tuned)")):
        ax.imshow(rgb, interpolation="nearest")
        ax.set_xlim(cx1, cx2)
        ax.set_ylim(cy2, cy1)
        ax.axis("off")
        boxes = row.get(f"_box_{side}", {})
        status = before if side == "before" else after
        for fdi, kind in sorted(status.items(), key=lambda kv: COLORS[kv[1]]):
            box = boxes.get(fdi)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            color = COLORS[kind]
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=line,
                edgecolor=color, facecolor="none",
                linestyle="--" if kind in ("removed", "added") else "-"))
            ax.text(x1, y1, f" {fdi} ", color="white", fontsize=font, fontweight="bold",
                    va="bottom", ha="left",
                    bbox=dict(facecolor=color, alpha=0.75, edgecolor="none",
                              boxstyle="round,pad=0.15"))
        ax.set_title(f"{label}   n={len(boxes)}", fontsize=font * 1.15)

    fig.suptitle(
        f"{row['case']}  {row['view']}   score={row['score']}  churn={row['churn']:.0%}"
        f"   -{row['removed']} +{row['added']} ~{row['moved']} relabel={row['relabel']}"
        f"   before n={row['n_before']} -> after n={row['n_after']}",
        fontsize=font * 1.25)
    handles = [patches.Patch(facecolor=COLORS[k], edgecolor="none", label=text)
               for k, text in LEGEND]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=font,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def write_index(out_dir: Path, rows: List[Dict[str, Any]], rendered: List[Dict[str, Any]]) -> None:
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>fine-tune A/B: worst disagreements</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}",
        "img{max-width:100%;border:1px solid #444;margin-bottom:6px}",
        "table{border-collapse:collapse;margin-bottom:32px;font-size:13px}",
        "td,th{border:1px solid #444;padding:4px 8px;text-align:right}",
        "td:first-child,th:first-child{text-align:left}",
        "h2{margin:28px 0 8px;font-size:16px}</style>",
        f"<h1>rendered {len(rendered)} of {len(rows)} differing (case, view)</h1>",
    ]
    for row in rendered:
        parts.append(f"<h2>{row['case']} {row['view']} &mdash; score {row['score']} "
                     f"({row['churn']:.0%} churn)</h2>")
        parts.append(f"<img src='{html.escape(row['png'])}' loading='lazy'>")
    parts.append("<h2>all differing samples (ranked)</h2><table>")
    parts.append("<tr><th>case</th><th>view</th><th>score</th><th>churn</th>"
                 "<th>before</th><th>after</th><th>same</th><th>moved</th>"
                 "<th>relabel</th><th>removed</th><th>added</th></tr>")
    for row in rows:
        parts.append("<tr>" + "".join(
            f"<td>{html.escape(str(row[c]))}</td>"
            for c in ("case", "view", "score", "churn", "n_before", "n_after",
                      "same", "moved", "relabel", "removed", "added")) + "</tr>")
    parts.append("</table>")
    (out_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")


# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()

    batch = resolve_input(args.batch)
    if not batch.exists():
        raise FileNotFoundError(f"--batch does not exist: {batch}")
    if batch.name.endswith("_process"):
        batch_parent = batch.parent
    else:
        found = sorted(p for p in batch.iterdir() if p.is_dir() and p.name.endswith("_process"))
        if len(found) != 1:
            raise SystemExit(
                f"--batch must be a *_process folder or a parent with exactly one of "
                f"them; {batch} has {len(found)}")
        batch, batch_parent = found[0], batch

    case_ids = sorted(p.name for p in (batch / "process").iterdir() if p.is_dir())
    if args.cases:
        case_ids = [c for c in case_ids if c in set(args.cases)]
    if not case_ids:
        raise SystemExit(f"no case folders under {batch / 'process'}")
    print(f"batch       : {batch}")
    print(f"cases       : {len(case_ids)}")
    print(f"before      : {args.orig_weight_dir}")
    print(f"after       : {args.ft_weight_dir}  (views: {' '.join(args.ft_views)})")

    out_dir = args.out or (REPO_ROOT / "exp" /
                           f"cmp_{batch_parent.name}_{batch.name.removesuffix('_process')}")
    out_dir = resolve_input(out_dir)
    before_tree = out_dir / "before"
    after_tree = out_dir / "after"
    print(f"out         : {out_dir}\n")

    if args.skip_detect:
        print("[1/4] detection skipped, reusing:")
        print(f"        {before_tree}")
        print(f"        {after_tree}")
    else:
        print("[1/4] detecting with the ORIGINAL checkpoints (the preview boxes)")
        run_detect(batch_parent, before_tree, args.orig_weight_dir.resolve(), case_ids, None)
        print("[2/4] detecting with the FINE-TUNED checkpoints")
        run_detect(batch_parent, after_tree, args.ft_weight_dir.resolve(), case_ids,
                   list(args.ft_views))

    # Views that keep the original checkpoint must be identical on both sides or
    # they would show up as fake differences. Done outside the detect branch so
    # that --skip-detect can re-score an existing after/ tree for a --ft-views
    # subset (e.g. keep only U and D) without re-running any inference.
    pinned = 0
    for case in case_ids:
        for stem in STEMS:
            if stem in args.ft_views:
                continue
            src = before_tree / case / f"{stem}.json"
            if src.exists():
                dst = after_tree / case / f"{stem}.json"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                pinned += 1
    if pinned:
        print(f"      {pinned} view(s) outside --ft-views pinned to the BEFORE result")
    print()

    if args.skip_verify:
        print("[3/4] verification skipped")
    else:
        print("[3/4] checking that BEFORE still reproduces the stored tooth_bbox/")
        verify_against_stored(batch, load_tree(before_tree), case_ids, out_dir)
        print()

    print("[4/4] diffing")
    before_all = load_tree(before_tree)
    after_all = load_tree(after_tree)

    rows: List[Dict[str, Any]] = []
    detail: Dict[str, Any] = {}
    for case in case_ids:
        for stem in STEMS:
            key = (case, stem)
            b = before_all.get(key)
            a = after_all.get(key)
            if b is None and a is None:
                continue
            b = b or {}
            a = a or {}
            records = diff_view(b, a, args.iou_ok, args.iou_match)
            summary = summarize(records, len(b))
            summary["case"] = case
            summary["view"] = stem
            summary["_box_before"] = {f: v["box"] for f, v in b.items()}
            summary["_box_after"] = {f: v["box"] for f, v in a.items()}
            rows.append(summary)
            detail[f"{case}/{stem}"] = {"summary": {k: v for k, v in summary.items()
                                                   if not k.startswith("_")},
                                        "boxes": records}

    rows.sort(key=lambda r: (r["score"], r["churn"]), reverse=True)

    fieldnames = ["case", "view", "score", "churn", "n_before", "n_after", "same",
                  "moved", "relabel", "removed", "added", "mean_iou_moved"]
    with (out_dir / "report.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "diff.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")

    changed = [r for r in rows if r["score"] >= args.min_score]
    rendered: List[Dict[str, Any]] = []
    for row in changed[: args.top]:
        image_path = batch / "process" / row["case"] / f"{row['view']}.png"
        if not image_path.exists():
            continue
        png = out_dir / "render" / f"{row['case']}_{row['view']}.png"
        render(row, detail[f"{row['case']}/{row['view']}"]["boxes"], image_path, png,
               args.render_mode, args.max_panel_px)
        row = dict(row)
        row["png"] = f"render/{png.name}"
        rendered.append(row)
    write_index(out_dir, rows, rendered)

    print(f"\n{'case':<12}{'view':<6}{'score':>6}{'churn':>8}{'before':>8}{'after':>7}"
          f"{'same':>6}{'moved':>7}{'relabel':>9}{'removed':>9}{'added':>7}")
    for row in rows[: min(30, len(rows))]:
        print(f"{row['case']:<12}{row['view']:<6}{row['score']:>6}{row['churn']:>8.0%}"
              f"{row['n_before']:>8}{row['n_after']:>7}{row['same']:>6}{row['moved']:>7}"
              f"{row['relabel']:>9}{row['removed']:>9}{row['added']:>7}")
    print(f"\n{len(changed)} of {len(rows)} (case, view) pairs differ "
          f"(score >= {args.min_score}); rendered the top {len(rendered)}.")
    print(f"open {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
