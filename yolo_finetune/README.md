# YOLO11 tooth detector fine-tuning

本目录用于微调 `segtooth_new.py` 使用的四个 YOLO11 检测模型：

- `front` 使用 `F` 图像。
- `upper` 使用 `U` 图像。
- `lower` 使用 `D` 图像。
- `right` 使用原始 `R` 图像以及在内存中水平翻转的 `L` 图像。
- 不训练单独的 `left` checkpoint，因为现有推理逻辑会翻转左视图并复用
  `segmentanytooth_yolo11_right.pt`。

## 数据发现

`dataset.py` 会递归搜索任意深度下的 `*_process` 文件夹，因此数据根目录下可以有
多个姓名目录。每个样本按如下结构配对：

```text
intraoral/
  person_name/
    20251013_process/
      process/subject_id/F.png
      tooth_bbox_revise/subject_id/F.json
```

默认只读取 `tooth_bbox_revise`，不会自动回退到模型预测生成的 `tooth_bbox`。
需要临时检查原始样例时可显式传入：

```powershell
--annotation-dir tooth_bbox
```

训练集和验证集按 `subject_id` 分组切分，同一个受试者不会同时出现在两个集合中。

## 数据不会被物化

训练直接使用自定义 Ultralytics Dataset 读取：

- 原始 `process/.../*.png` 路径。
- 对应 `tooth_bbox_revise/.../*.json` 标注。
- checkpoint 自带的 class names。

不会创建 `exp/datasets`，也不会生成 YOLO `.txt` 标签、manifest、dataset YAML、
图片副本、hardlink、symlink 或翻转后的 L 图片。标签解析结果只存在于当前训练进程
的内存中。

对于 `right` 模型，Dataset 会读取原始 L 图片，然后在 `load_image()` 中水平翻转
内存数组，同时将 box 从 `[x1, y1, x2, y2]` 转换为
`[W-x2, y1, W-x1, y2]`。磁盘上的 L 图片保持不变。

## 安装

```powershell
pip install -r yolo_finetune/requirements.txt
```

YOLO11 需要较新的 Ultralytics。服务器已有项目环境时，优先沿用该环境中的
PyTorch/CUDA 版本。

W&B 默认启用，训练前需要完成登录：

```powershell
wandb login
```

## 数据检查

先对一个视角执行 dry run。该命令会加载原 checkpoint 的类别顺序、逐个检查 JSON、
FDI 映射和 box，并打印样本/box 统计。不会写入任何文件：

```powershell
python -m yolo_finetune.train `
  --data-root /disk1/work/zychen/intraoral `
  --checkpoint-dir /disk1/work/zychen/Checkpoints/segtooth_model `
  --view front `
  --annotation-dir tooth_bbox_revise `
  --dry-run
```

检查模型结构使用独立入口：

```powershell
python -m yolo_finetune.model `
  /path/to/segmentanytooth_yolo11_front.pt
```

## 训练

训练单个视角：

```powershell
python -m yolo_finetune.train `
  --data-root /path/to/intraoral `
  --checkpoint-dir /path/to/.checkpoints/segtooth_model `
  --view front `
  --annotation-dir tooth_bbox_revise `
  --epochs 100 `
  --batch 8 `
  --imgsz 1024 `
  --device 0 `
  --project exp `
  --wandb-entity your_team `
  --wandb-project intraoral-yolo-finetune `
  --wandb-name front_finetune
```

依次训练全部四个模型：

```powershell
python -m yolo_finetune.train `
  --data-root /path/to/intraoral `
  --checkpoint-dir /path/to/.checkpoints/segtooth_model `
  --view all `
  --annotation-dir tooth_bbox_revise `
  --epochs 100 `
  --device 0 `
  --wandb-entity your_team `
  --wandb-project intraoral-yolo-finetune `
  --wandb-name tooth_detector
```

使用 `--view all` 时，W&B run name 会自动追加 `_front/_upper/_lower/_right`。
训练会固定启用 W&B；`--dry-run` 只检查数据，不会创建 W&B run。

## W&B 指标

每个 epoch 记录：

- `train/box_loss`、`train/cls_loss`、`train/dfl_loss`。
- `val/box_loss`、`val/cls_loss`、`val/dfl_loss`。
- `metrics/precision(B)`、`metrics/recall(B)`。
- `metrics/mAP50(B)`、`metrics/mAP50-95(B)`。
- 各 optimizer parameter group 的 learning rate。

这些验证指标由 Ultralytics detection validator 计算。训练结束后，最后一次验证指标
和 `best_fitness` 也会写入 W&B run summary。checkpoint 不作为 W&B artifact 上传。

每次训练会保留 Ultralytics 原生的：

```text
exp/yolo11_front_finetune/weights/best.pt
exp/yolo11_front_finetune/weights/last.pt
```

并额外汇总出：

```text
exp/segtooth_model/segmentanytooth_yolo11_front.pt
exp/segtooth_model/segmentanytooth_yolo11_upper.pt
exp/segtooth_model/segmentanytooth_yolo11_lower.pt
exp/segtooth_model/segmentanytooth_yolo11_right.pt
```

这些文件可直接放入 `segtooth_new.py --weight_dir` 指向的原权重目录，替换对应
YOLO 文件；SAM 的 `segmentanytooth_vit_tiny.pt` 保持不变。加载方式仍是
`YOLO(model=checkpoint_path)`。

## 重要训练约定

- 类别 ID 不由 JSON 排序重新生成，而是从原 checkpoint 的 `model.names` 读取，
  从而保持预训练 Detect head 与 FDI 类别一一对应。
- L 图的类别映射严格复用 `segtooth_new.py` 的语义：标注 FDI 在
  `LEFT_CLASSES` 中的位置就是训练 class ID。例如 `LEFT_CLASSES[0] == "le28"`，
  则 L 图的 FDI 28 使用 class ID 0。这与推理时
  `fdi = LEFT_CLASSES[predicted_class_id]` 完全一致。
- `right` 固定同时加载 R 和 L，L 图只在内存中翻转。
- 随机水平/垂直翻转被强制关闭。FDI 类别包含左右位置语义，普通 flip augmentation
  不会自动交换类别，会产生错误监督。
- 其余优化器、学习率、AMP 和数据增强参数使用 Ultralytics 默认值。
- Ultralytics 的 RAM/disk image cache 均关闭，尤其不会在源图片旁生成 `.npy`。
- 训练时仅产生正常 run 输出，例如 `weights/best.pt`、`weights/last.pt`、
  `results.csv` 和 `args.yaml`；训练可视化 plots 默认关闭。
