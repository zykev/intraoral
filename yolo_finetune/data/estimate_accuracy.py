#!/usr/bin/env python3
"""Estimate the pre-trained tooth-number model's accuracy.

Part of the YOLO11 fine-tuning data preparation: this is the QC report that
says how good the checkpoint currently is, so improvements from a fine-tuning
run can be measured against a fixed baseline.

Compares the **unfiltered** model predictions
(``intraoral_adjust/unfiltered_bbox``, the same detections the ``preview/``
PNGs are drawn from -- i.e. *without* the per-view FDI whitelist
``TARGET_TEETH_PER_VIEW``) against the manually corrected ground-truth-proxy
(``intraoral_adjust/tooth_bbox_adj``), purely from the two JSON sets.

Two prediction trees, on purpose
--------------------------------
* ``unfiltered_bbox``    -- predictions of the *original* checkpoints. This is
  the FROZEN baseline: ``build_tooth_bbox_adj.py`` reads it to synthesize the
  GT, and ``--pred`` defaults to it so a fine-tuning run can be compared
  against a number that never moves.
* ``unfiltered_bbox_ft`` -- scratch output of ``build_tooth_bbox_predict.py``
  (its default ``--out``), used for re-runs and for the fine-tuned
  checkpoints. Point ``--pred`` here to score a fine-tuned run, e.g.
  ``--pred intraoral_adjust/unfiltered_bbox_ft``. Never regenerate the frozen
  tree just to score a new model, or the baseline disappears.

Using the unfiltered predictions is deliberate: the question is whether the
detector got **every** tooth it proposed right, not just the teeth that
survive the per-view filter.  ``tooth_bbox_adj`` is built on the same
unfiltered base, so the two sides are comparable.

Matching, per (case, view)
--------------------------
1. Pair every FDI present in both sets (the model likely got the label right;
   the IoU then tells whether the box is also right).
2. Greedily match the leftover boxes by IoU (>0.3): a leftover pair with
   *different* FDI is a **relabel** (the model put the right box on the wrong
   tooth number).
3. Leftover original boxes          -> **false positive** (tooth doesn't exist).
4. Leftover corrected boxes        -> **miss** (tooth the model never found).

Per-tooth outcome
-----------------
correct    : same FDI, IoU >= 0.9   (label + box right)
loc_err    : same FDI, IoU  < 0.9   (right tooth, box imprecise)
relabel    : matched box, FDI wrong
fp         : predicted a non-existent tooth
miss       : failed to predict an existing tooth

Only the standard library is used.
"""

import argparse
import collections
import csv
import glob
import json
import os

VIEWS = ["D", "F", "L", "R", "U"]


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def load(path):
    with open(path, "r", encoding="utf-8") as fh:
        j = json.load(fh)
    return {t["fdi"]: list(t["bbox_yolo"]) for t in j["teeth"]}


def compare(orig, adj, thresh=0.3, iou_ok=0.9):
    """Return (counts, list_of_iou) for one view."""
    counts = collections.Counter()
    ious = []

    O = dict(orig)
    A = dict(adj)

    # 1. same FDI present in both
    for fdi in list(O):
        if fdi in A:
            v = iou(O.pop(fdi), A.pop(fdi))
            ious.append(v)
            counts["teeth"] += 1
            if v >= iou_ok:
                counts["correct"] += 1
            else:
                counts["loc_err"] += 1

    # 2. greedy IoU match on the leftovers -> relabel
    pairs = []
    for of, ob in O.items():
        for af, ab in A.items():
            v = iou(ob, ab)
            if v >= thresh:
                pairs.append((v, of, af))
    pairs.sort(reverse=True)
    used_o, used_a = set(), set()
    for v, of, af in pairs:
        if of in used_o or af in used_a:
            continue
        used_o.add(of)
        used_a.add(af)
        ious.append(v)
        counts["teeth"] += 1
        if of != af:
            counts["relabel"] += 1
        elif v >= iou_ok:
            counts["correct"] += 1
        else:
            counts["loc_err"] += 1

    # 3+4. leftovers
    counts["fp"] += len(set(O) - used_o)          # model-only  (not in GT)
    counts["miss"] += len(set(A) - used_a)        # GT-only     (not predicted)

    counts["n_pred"] += len(orig)
    counts["n_gt"] += len(adj)
    return counts, ious


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", default="intraoral_adjust/unfiltered_bbox",
                    help="unfiltered predictions (<pred>/<case>/<view>.json); "
                         "same detection the preview/ PNGs show. The default "
                         "is the FROZEN baseline; pass "
                         "intraoral_adjust/unfiltered_bbox_ft to score a "
                         "fine-tuned run.")
    ap.add_argument("--adj", default="intraoral_adjust/tooth_bbox_adj")
    ap.add_argument("--out", default="intraoral_adjust/reports")
    args = ap.parse_args()

    # case -> unfiltered prediction file dir
    omap = {}
    for cdir in sorted(glob.glob(os.path.join(args.pred, "*"))):
        if os.path.isdir(cdir):
            omap.setdefault(os.path.basename(cdir), cdir)
    if not omap:
        raise FileNotFoundError(
            "no predictions found under %s (run build_tooth_bbox_predict.py first)" % args.pred
        )

    total = collections.Counter()
    per_view = collections.defaultdict(collections.Counter)
    rows = []

    for ap_path in sorted(glob.glob(os.path.join(args.adj, "*", "*.json"))):
        case = os.path.basename(os.path.dirname(ap_path))
        view = os.path.splitext(os.path.basename(ap_path))[0]
        op = os.path.join(omap.get(case, ""), view + ".json")
        if not os.path.exists(op):
            continue
        A = load(ap_path)
        O = load(op)
        c, ious = compare(O, A)
        for k, v in c.items():
            total[k] += v
            per_view[view][k] += v
        mean_iou = sum(ious) / len(ious) if ious else 0.0
        rows.append([case, view, c["n_pred"], c["n_gt"], c["correct"],
                     c["loc_err"], c["relabel"], c["fp"], c["miss"], round(mean_iou, 4)])

    # ---- report ----
    def line(name, c):
        pred = c["n_pred"] or 1
        gt = c["n_gt"] or 1
        acc = c["correct"] / pred
        return ("%-4s  pred=%-6d gt=%-6d  correct=%-6d (%.1f%%)  loc_err=%-5d "
                "relabel=%-5d  fp=%-5d  miss=%-5d"
                % (name, c["n_pred"], c["n_gt"], c["correct"], 100 * acc,
                   c["loc_err"], c["relabel"], c["fp"], c["miss"]))

    print("== Per view ==")
    for v in VIEWS:
        if per_view[v]:
            print(line(v, per_view[v]))
    print("== Overall ==")
    print(line("ALL", total))

    pred = total["n_pred"] or 1
    gt = total["n_gt"] or 1
    print()
    print("predicted teeth ................ %d" % total["n_pred"])
    print("ground-truth teeth ............. %d" % total["n_gt"])
    print("fully-correct (label+box) ...... %d  (%.2f%% of predictions, %.2f%% of GT)"
          % (total["correct"], 100 * total["correct"] / pred, 100 * total["correct"] / gt))
    print("label correct, box imprecise ... %d  (%.2f%% of predictions)"
          % (total["loc_err"], 100 * total["loc_err"] / pred))
    print("wrong FDI (relabelled) ......... %d  (%.2f%% of predictions)"
          % (total["relabel"], 100 * total["relabel"] / pred))
    print("false positives (spurious) ..... %d  (%.2f%% of predictions)"
          % (total["fp"], 100 * total["fp"] / pred))
    print("missed teeth ................... %d  (%.2f%% of GT)"
          % (total["miss"], 100 * total["miss"] / gt))
    denom = total["correct"] + total["relabel"] + total["loc_err"]
    if denom:
        print()
        print("label accuracy (FDI) ........... %.2f%%"
              % (100 * (total["correct"] + total["loc_err"]) / denom))
        print("box   accuracy (IoU>=0.9) ...... %.2f%%"
              % (100 * total["correct"] / denom))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "accuracy_per_sample.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "view", "n_pred", "n_gt", "correct", "loc_err",
                    "relabel", "false_positive", "miss", "mean_iou"])
        w.writerows(rows)
    print("\nper-sample CSV ->", os.path.join(args.out, "accuracy_per_sample.csv"))


if __name__ == "__main__":
    main()
