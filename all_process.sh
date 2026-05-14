#!/bin/bash

# --- 1. 配置区域 ---
# 获取命令行参数中的路径，默认为 .datasets/intraoral/zeyu
BASE_DIR=${1:-".datasets/intraoral/garris"}

# 转换为绝对路径，确保在不同目录下 cd 后依然有效
ABS_BASE_DIR=$(readlink -f "$BASE_DIR")

# Conda 初始化路径
CONDA_PATH="/home/peiliu/miniconda3/etc/profile.d/conda.sh"

echo "==========================================="
echo "开始执行全流程预处理"
echo "目标路径: $ABS_BASE_DIR"
echo "==========================================="

# --- 2. 权限初始化 ---
echo "[Step 0] 正在授予目录读写权限..."
if [ -d "$ABS_BASE_DIR" ]; then
    # -R 递归，u+w 赋予当前用户写权限
    chmod -R u+w "$ABS_BASE_DIR"
    echo "权限设置完成。"
else
    echo "警告: 目录 $ABS_BASE_DIR 不存在，跳过权限设置。"
fi

# 初始化 conda 环境
if [ -f "$CONDA_PATH" ]; then
    source "$CONDA_PATH"
else
    echo "错误: 未找到 conda.sh，请检查路径: $CONDA_PATH"
    exit 1
fi

# --- 3. 执行数据格式转换与裁剪 (sam3 环境) ---
echo -e "\n[Step 1/2] 进入数据预处理 (sam3)..."
cd /home/peiliu/Documents/intraoral_process/ || exit 1

conda activate sam3
python preprocess.py --base_dir "$ABS_BASE_DIR"

if [ $? -ne 0 ]; then
    echo "错误: preprocess.py 执行失败"
    exit 1
fi

# --- 4. 执行牙齿分割与保存 (segtooth 环境) ---
echo -e "\n[Step 2/2] 进入牙齿分割 (segtooth)..."
cd /home/peiliu/Documents/SegmentAnyTooth/ || exit 1

conda activate segtooth
python segtooth.py --base_dir "$ABS_BASE_DIR"

if [ $? -ne 0 ]; then
    echo "错误: segtooth.py 执行失败"
    exit 1
fi

echo -e "\n==========================================="
echo "全流程处理已成功完成！"
echo "==========================================="