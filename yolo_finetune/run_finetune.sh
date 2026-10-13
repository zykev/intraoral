#!/usr/bin/env bash
# Conservative fine-tune of the four SegmentAnyTooth detectors.
#
# Design: adapt the pretrained model, do not retrain it.
#   * Learning rate is left at the Ultralytics 'optimizer=auto' value, which is
#     what a plain run of train.py already does. Do NOT pass --lr0: 'auto'
#     ignores it and recomputes from the class count alone. That yields
#         front 2.7778e-4   upper 5.0000e-4   lower 5.0000e-4   right 3.5714e-4
#     i.e. NOT 5e-4 everywhere -- 'auto' scales by nc, and front/right have 32
#     and 24 classes against 16 for upper/lower.
#   * freeze=11 freezes model.0..model.10, which is yolo11's entire backbone,
#     BatchNorm running stats included. Only the neck and the Detect head move
#     (47.3% of the parameters), so the pretrained features are bit-for-bit
#     preserved. 'freeze' cannot change the LR -- 'auto' derives it from nc only.
#   * Epoch budget follows how much of each view's ground truth is actually new
#     information (changed + added + deleted boxes as a share of the GT):
#         front 8.1%   upper 22.6%   lower 41.5%   right 6.4%
#     Measured by diffing tooth_bbox_adj against unfiltered_bbox: 87.3% of all
#     8955 GT boxes are the model's own prediction reused verbatim.
#   * Warmup needs care. The trainer floors it at
#         nw = max(round(warmup_epochs * batches_per_epoch), 100)
#     and batches_per_epoch is 12 (front/upper/lower) or 23 (right), so 100
#     iterations is 8.3 / 4.3 epochs. Any value in (0, 8.3] therefore behaves
#     identically -- there is no such thing as a short warmup. Only 0.0 disables
#     it. So: 0.0 on the short runs (100 iters would be most of the run), a real
#     ramp only where the run is long enough to afford it.
#   * left shares the right checkpoint (it is the right model on a flipped
#     image), so there is no separate 'left' command.
#
# Single GPU per process on purpose: Ultralytics DDP re-imports the trainer
# class from __main__ and only ships cfg + overrides to the child, so it cannot
# rebuild this in-memory dataset. Keep each command on one GPU; if you want
# parallelism run them concurrently with different --device values.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-/home/zychen/miniconda3/envs/intraoral/bin/python}"
export PYTHONPATH="$REPO_ROOT"

# --- the four detectors, one command each ------------------------------------
# front: 8.1% new signal -> short and small-LR (auto picks 2.78e-4, the lowest of
#        the four). Warmup off: 100 iters would be 8.3 of 15 epochs.
"$PYTHON" -m yolo_finetune.train --view front --epochs 15 --patience 8  --val-interval 5 --freeze 11 --warmup-epochs 0.0 --device 1 --project exp --swanlab-name yoloft_front_v1

# upper: 22.6% new signal (auto lr0 = 5.00e-4). 100-iter ramp = 8.3 of 40 epochs.
"$PYTHON" -m yolo_finetune.train --view upper --epochs 40 --patience 10 --val-interval 5 --freeze 11 --warmup-epochs 1.0 --device 1 --project exp --swanlab-name yoloft_upper_v1

# lower (occlusal): 41.5% new signal, holds 354 of the 503 added boxes and 306 of
#        the 470 baseline misses -- the view that actually needs adapting. 100-iter
#        ramp = 8.3 of 60 epochs. If misses do not drop, loosen to --freeze 10
#        (57.0% trainable) and only then to --freeze 8 (76.7%): one step at a time.
#        For such an A/B give the second run its own --project: the output dir is
#        derived from the view, not the swanlab name (train.py run_name is always
#        yolo11_<view>_finetune, and Ultralytics then appends a "2"), while the
#        export path exp/segtooth_model/ is fixed and WOULD be overwritten.
"$PYTHON" -m yolo_finetune.train --view lower --epochs 60 --patience 15 --val-interval 5 --freeze 11 --warmup-epochs 1.0 --device 1 --project exp --swanlab-name yoloft_lower_v1

# right: 6.4% new signal, but this single checkpoint also serves the left view
#        (6.4-7.1%), so one command covers two views. 23 batches/epoch, so the
#        100-iter ramp is only 4.3 epochs.
"$PYTHON" -m yolo_finetune.train --view right --epochs 20 --patience 8  --val-interval 5 --freeze 11 --warmup-epochs 1.0 --device 1 --project exp --swanlab-name yoloft_right_v1

# --- acceptance test, not the training mAP -----------------------------------
# In-training mAP is computed against labels 87.3% of which the model produced
# itself, so it barely moves. Score the exported weights instead and compare the
# miss count against the frozen baseline (470 total, 306 of them on lower):
#   cd .datasets && "$PYTHON" ../yolo_finetune/data/build_tooth_bbox_predict.py --out intraoral_adjust/unfiltered_bbox_ft
#   cd .datasets && "$PYTHON" ../yolo_finetune/data/estimate_accuracy.py --pred intraoral_adjust/unfiltered_bbox_ft
# --weight_dir already defaults to exp/segtooth_model, where this script exports.
# Roll the front/left/right views back to the original checkpoints if they lose
# more than ~1 point there; only the lower view is expected to improve.
