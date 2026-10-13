#!/usr/bin/env bash
# Acceptance test for the fine-tuned detectors: score them against the manual GT
# and against the frozen pre-fine-tune baseline, side by side.
#
# NOT the in-training mAP. Labels are 87.3% the model's own output, so validation
# mAP barely moves; what matters is whether the miss count drops. Baseline, from
# estimate_accuracy.py on the frozen checkpoints:
#   ALL   pred=8715 gt=8955  correct=7880 (90.4% of pred, 88.0% of GT)  loc_err=477
#         relabel=128  fp=230  miss=470      label acc 98.49%   box acc(IoU.9) 92.87%
#   miss per view:  D 306   U 81   F 34   L 30   R 19     (D alone is 65% of it)
#   D is the only view under 90%: 880/1422 = 77.7% of its predictions
# So a win looks like: D miss well below 306 and D correct above 880, with the other
# four views' numbers essentially unchanged (they are already >93% and were barely
# touched, so treat any visible drop there as a regression and roll them back).
#
# --- why the `cd` below is load-bearing -----------------------------------------
# build_tooth_bbox_predict.py and estimate_accuracy.py resolve --base_dir / --out /
# --pred / --adj relative to the DATASETS ROOT, not to the repo root:
#     .datasets -> /disk1/work/zychen/Datasets
#         intraoral/zeyu/<batch>_process/process/<case>/<view>.png
#         intraoral_adjust/{tooth_bbox_adj,unfiltered_bbox,unfiltered_bbox_ft,reports}
# Run them from the repo root and `intraoral/zeyu` becomes <repo>/intraoral/zeyu,
# which does not exist:
#     FileNotFoundError: .../intraoral/intraoral/zeyu
# So the script cds into .datasets first. --weight_dir is absolute and unaffected.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASETS_ROOT="${DATASETS_ROOT:-$REPO_ROOT/.datasets}"
PYTHON="${PYTHON:-/home/zychen/miniconda3/envs/intraoral/bin/python}"
DATA="$REPO_ROOT/yolo_finetune/data"

# Overridable so this doubles as the A/B harness for a different experiment.
BASE_DIR="${BASE_DIR:-intraoral/zeyu}"                     # the 112 GT cases
WEIGHTS="${WEIGHTS:-$REPO_ROOT/exp/segtooth_model}"        # build_tooth_bbox_predict's default
PRED_FT="${PRED_FT:-intraoral_adjust/unfiltered_bbox_ft}"  # never the frozen baseline tree
PRED_BASE="${PRED_BASE:-intraoral_adjust/unfiltered_bbox}"
REPORTS="${REPORTS:-intraoral_adjust/reports}"

cd "$DATASETS_ROOT"
export PYTHONPATH="$REPO_ROOT"

echo "### 1/3  re-detect with the fine-tuned checkpoints (560 images, the slow part)"
"$PYTHON" "$DATA/build_tooth_bbox_predict.py" \
    --base_dir "$BASE_DIR" \
    --out "$PRED_FT" \
    --weight_dir "$WEIGHTS"

echo
echo "### 2/3  fine-tuned vs GT   (diff the 'miss' line against step 3)"
"$PYTHON" "$DATA/estimate_accuracy.py" \
    --pred "$PRED_FT" \
    --out "$REPORTS/ft"

echo
echo "### 3/3  frozen baseline vs the same GT   (cheap: predictions already exist)"
"$PYTHON" "$DATA/estimate_accuracy.py" \
    --pred "$PRED_BASE" \
    --out "$REPORTS/baseline"

echo
echo "per-view CSVs:"
echo "  $DATASETS_ROOT/$REPORTS/ft/accuracy_per_sample.csv"
echo "  $DATASETS_ROOT/$REPORTS/baseline/accuracy_per_sample.csv"
