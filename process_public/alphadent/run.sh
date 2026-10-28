#!/usr/bin/env bash
# AlphaDent -> single-tooth 数据集转换全流程
# 用法: bash process_public/alphadent/run.sh [stage]
#   stage = views | build | report | all   (默认 all)
set -euo pipefail

PY=${PY:-/home/zychen/miniconda3/envs/intraoral/bin/python}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STAGE=${1:-all}

# 1) 判方向（1455 张，8 并发大约 30 分钟）
if [[ "$STAGE" == "all" || "$STAGE" == "views" ]]; then
    echo "=== [1/2] inferring view + rotation ==="
    "$PY" "$HERE/infer_view.py" --splits train valid test --num_workers 8
    "$PY" "$HERE/infer_view.py" --stats
fi

# 2) 构建单牙数据集（1318 张，约 55 分钟）
#    默认导出所有检测到的单牙（含无标注的负样本）；加 --only_annotated 可丢弃负样本
if [[ "$STAGE" == "all" || "$STAGE" == "build" ]]; then
    echo "=== [2/2] building single-tooth dataset ==="
    "$PY" "$HERE/build_single_tooth.py" --splits train valid
fi

# 3) 汇总报告
if [[ "$STAGE" == "all" || "$STAGE" == "report" ]]; then
    echo "=== report ==="
    "$PY" "$HERE/build_single_tooth.py" --report
fi
