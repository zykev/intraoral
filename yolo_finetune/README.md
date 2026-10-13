# YOLO11 tooth detector fine-tuning

本目录用于微调 `segtooth_new.py` 使用的四个 YOLO11 检测模型：

- `front` 使用 `F` 图像。
- `upper` 使用 `U` 图像。
- `lower` 使用 `D` 图像。
- `right` 使用原始 `R` 图像以及在内存中水平翻转的 `L` 图像。
- 不训练单独的 `left` checkpoint，因为现有推理逻辑会翻转左视图并复用
  `segmentanytooth_yolo11_right.pt`。

## 数据发现

标注默认来自人工修正后的 GT（`tooth_bbox_adj`），它是一个扁平、与 batch 无关的
目录树：

```text
intraoral_adjust/tooth_bbox_adj/<subject>/<view>.json            默认标注（GT）
intraoral/<person>/<batch>_process/process/<subject>/<view>.png  原始图
```

`--annotation-root` 指向这棵树（默认 `intraoral_adjust/tooth_bbox_adj`）；图片不在
标注旁边，而是通过每条标注自带的 `orig_size` 去 `--data-root` 下匹配（像素尺寸
一致才算命中），因此图片和标注错配会直接报错，不会被静默用错。

标注树的两份数据都软链到了仓库内，**不需要 `cd` 到 `/disk1`**：

```text
.datasets/intraoral        -> /disk1/work/zychen/Datasets/intraoral
.datasets/intraoral_adjust -> /disk1/work/zychen/Datasets/intraoral_adjust
```

所以 `--data-root` 和 `--checkpoint-dir` 的默认值就是仓库内相对路径
（`.datasets/intraoral`、`.checkpoints/segtooth_model`）。相对写的
`--annotation-root` 会按下面的顺序查找，第一个存在的胜出：

1. `--data-root` 的同级目录，例如 `<data-root>/../intraoral_adjust/tooth_bbox_adj`
2. `--data-root` 内部
3. 当前工作目录
4. 仓库内的 `.datasets/`

绝对路径直接使用。因此下面的两种写法等价：

```powershell
# 仓库内软链（推荐，无需 cd）
python -m yolo_finetune.train --view all

# 存储卷绝对路径
python -m yolo_finetune.train `
  --data-root /disk1/work/zychen/Datasets/intraoral `
  --annotation-root intraoral_adjust/tooth_bbox_adj `
  --view all
```

`--annotation-root none` 时退回旧的 per-batch 布局，标注放在每个 `*_process`
文件夹里：

```text
intraoral/
  person_name/
    20251013_process/
      process/subject_id/F.png
      tooth_bbox_revise/subject_id/F.json
```

此时用 `--annotation-dir`（默认 `tooth_bbox_revise`）选择子目录。

训练集和验证集按 `subject_id` 分组切分，同一个受试者不会同时出现在两个集合中。

## 跨视角牙齿（widen）

微调**不再**套用 `segtooth.py::TARGET_TEETH_PER_VIEW`，而是保留 checkpoint class
map 能表达的全部 FDI，包括“跨视角”的牙（例如 F 图里的 14-18 / 24-28、R/L 图里越过
中线的 11-13 / 21-24 / 31-34）。理由：让 YOLO 尽量把可见的牙都检出来；per-view
白名单是**推理之后**的策略，不应该反过来限制训练监督。

| 视角 | 原先保留 | 现在保留 | 新增监督 |
|---|---|---|---|
| `front` | 12 颗前牙 | checkpoint 全部 32 类 | +833 box |
| `upper` | 16 颗上牙 | 同上（本来就是全 arch） | 0 |
| `lower` | 16 颗下牙 | 同上（本来就是全 arch） | 0 |
| `right` + 翻转的 `left` | 10 颗右侧后牙 | checkpoint 全部 24 类 | +2461 box |

权威来源是 checkpoint 的 `model.names`，也就是 `dataset.py::CROSS_VIEW_FDI`
记录的差集（改 `segtooth.py` 的表时记得同步它）。

checkpoint 没有对应 class 的 FDI（理论上不该出现）默认**丢弃并在 audit 里报数**
（`dropped_unknown_boxes` / `dropped_unknown_fdi`）；加 `--strict-fdi` 改为直接报错。
当前 `tooth_bbox_adj` 实测 `dropped_unknown_boxes = 0`，即全部标注都能表达。

## 数据不会被物化

训练直接使用自定义 Ultralytics Dataset 读取：

- 原始 `process/.../*.png` 路径。
- 对应的 FDI 标注 JSON（默认 `tooth_bbox_adj/.../*.json`）。
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

SwanLab 默认启用（`workspace=zykev`、`project=dentmap`）。首次使用需要登录一次，
之后 token 会缓存到用户目录：

```powershell
swanlab login
```

也可以完全跳过交互式登录，用环境变量：

```bash
export SWANLAB_API_KEY=<你的 API Key>
```

离线或本地调试用 `SWANLAB_MODE=offline`（不联网，日志落在 `--project` 下）或
`SWANLAB_MODE=local`（本地看板）：

```bash
SWANLAB_MODE=offline python -m yolo_finetune.train --view front --device 0
```

## 数据检查

先对一个视角执行 dry run。该命令会加载原 checkpoint 的类别顺序、逐个检查 JSON、
FDI 映射和 box，并打印样本/box 统计（含跨视角 box 数）。不会写入任何文件：

```powershell
python -m yolo_finetune.train `
  --data-root /disk1/work/zychen/Datasets/intraoral `
  --checkpoint-dir /disk1/work/zychen/Checkpoints/segtooth_model `
  --annotation-root /disk1/work/zychen/Datasets/intraoral_adjust/tooth_bbox_adj `
  --view front `
  --dry-run
```

在仓库根目录直接跑同样一条（全用默认值，走 `.datasets/` 软链）：

```bash
python -m yolo_finetune.train --view front --dry-run
```

检查模型结构使用独立入口：

```powershell
python -m yolo_finetune.model `
  /path/to/segmentanytooth_yolo11_front.pt
```

## 训练

### 默认超参（已经按本项目的失败模式调过）

| 参数 | 默认值 | 为什么 |
|---|---|---|
| `--epochs` | `120` | 见下方 `nbs` 说明：配 `nbs==batch` 时约 1440 个 optimizer step |
| `--batch` | `8` | 1024 分辨率下的显存安全值 |
| `--nbs` | `= --batch`（即 `8`） | **关键**。Ultralytics 用 `nbs` 推导梯度累积：默认 `nbs=64` 时 `accumulate = 64/8 = 8`，每 epoch 只有 `images/64` 个 optimizer step（90 张的视角 = **1.4** 步），基本等于没训。设成 `8` 后每一步都是一个 batch |
| `--patience` | `30` | 早停。Ultralytics 默认 `100`，在 `epochs=120` 时永远不会触发 |
| `--val-interval` | `10` | 每 10 个 epoch 验证一次（最后一个 epoch 一定验证）。验证集只有 22/44 张，但一次验证约等于 1/4 个训练 epoch，而只有 `best.pt` 和早停会读这个结果 |
| `--save-period` | `-1` | 除 `last.pt`/`best.pt` 外每个 epoch 都存一份 `weights/epochN.pt`；`-1` 表示关掉 |
| `--imgsz` | `1024` | 密合面视角（D/U）牙齿很小，再降会伤召回 |
| `--mosaic` | `0.0` | **关闭**。mosaic 把 4/9 张图拼成一张，既破坏了决定 FDI 的全局牙弓布局，又把本来就很密的 D/U 图分辨率再切一半 |
| `--scale` | `0.3` | 相机固定距离、自动翻转已禁用，缩放抖动空间比 Ultralytics 默认 `0.5` 小 |
| `--cos-lr` | 开启 | 余弦学习率；要线性用 `--no-cos-lr` |
| `--freeze` | `None` | 冻结 `model.0 .. model.{N-1}`（含 BatchNorm running stats）。`11` = 整个 backbone，只剩 neck + Detect 头可训 |
| `--warmup-epochs` | `None` | 透传 Ultralytics 的 `warmup_epochs`（默认 `3.0`）。注意有效下限是 100 个 iteration，见 `--freeze` / warmup 一节 |

`--nbs` 不显式给值时等于 `--batch`。想要旧行为（Ultralytics 默认梯度累积）就写
`--nbs 64`，但那样需要把 `--epochs` 提到 ~300 才能凑够相近的步数。

### 学习率不由 `--lr0` 决定

默认 `optimizer=auto` 会打印一行 `ignoring 'lr0=0.01' and 'momentum=0.937'`，然后
按类别数自己算：`lr0 = round(0.002 * 5 / (4 + nc), 6)`。所以四个视角并不相同：

| 视角 | `nc` | 实际 `lr0` |
|---|---|---|
| front | 32 | `2.7778e-4` |
| upper | 16 | `5.0000e-4` |
| lower | 16 | `5.0000e-4` |
| right | 24 | `3.5714e-4` |

想保留“原始设定”就**什么都不要传**；一旦传了 `--lr0`，`auto` 会忽略它，必须同时显式
传 `optimizer`（例如 `AdamW`）才生效。**`--freeze` 不会改变学习率**（`auto` 只看 `nc`），
所以“冻 backbone + 原始 LR”是自洽的。

### `--freeze` 与 warmup：想“轻微适配”时用

`freeze=N` 展开成冻结前缀 `model.0.` … `model.{N-1}.`，而 `yolo11.yaml` 的 backbone
正好是索引 **0–10**（最后一个 `C2PSA` 在 10，neck 从 11 的 `Upsample` 开始）。实测：

| `--freeze` | 冻结的模块 | 可训练参数 |
|---|---|---|
| `None` | 只有 `.dfl` | 2,592,944（100.0%） |
| `5` | 0–4（P1–P3 段） | 2,518,096（97.1%） |
| `8` | 0–7 | 1,987,920（76.7%） |
| `10` | 0–9 | 1,477,200（57.0%） |
| **`11`** | **0–10，整个 backbone** | **1,227,472（47.3%）** |
| `12` | 0–11（与 11 等价） | 1,227,472（47.3%） |

`11` 是能冻住整个 backbone 的**最小整数**；`12` 等价，因为 `model.11` 是无参 `Upsample`。
`13` 就开始切进 neck 了。冻结不只是梯度为 0：`_model_train()` 还会把冻结层里的
`BatchNorm2d` 置成 `eval()`，所以这批权重和 BN running stats **一个 bit 都不变**。
Detect 里的 `.dfl` 永远冻结（16 个参数），所以即使不冻也有 2,592,944 而非 2,592,960 可训。

**warmup 有个容易踩的坑**：`trainer.py:383`

```python
nw = max(round(args.warmup_epochs * nb), 100) if args.warmup_epochs > 0 else -1
```

`nb` = 每 epoch 的 batch 数 = **12**（front/upper/lower，90 张 / batch 8）或 **23**（right，180 张）。
于是 `nw` 被 `100` 这个下限卡住 ⇒ **`(0, 8.3]` 之间任何值效果完全一样，都是 100 个
iteration ≈ 8.3 个 epoch（right 是 4.3 个）**。只有 `0.0` 才是真正关掉 warmup
（此时 `nw = -1`，LR 从第一个 step 就是 `initial_lr`）。所以“把 warmup 调短一点”这件事
不存在，只能选“大概 8 个 epoch”或“没有”。

### 保守微调（`run_finetune.sh`）

`yolo_finetune/run_finetune.sh` 是“只做轻微适配”的推荐配置：保留原始 LR、`--freeze 11`、
按每个视角的**新增信息量**决定 epoch 数。新增信息量 = GT 相对 `unfiltered_bbox`
发生变化（改动 + 新增 + 删除）的框占比：

| 视角 | 新信息占比 | `--epochs` | `--freeze` | `--warmup-epochs` | 实际 warmup |
|---|---|---|---|---|---|
| front | 8.1% | 15 | 11 | `0.0` | 无 |
| upper | 22.6% | 40 | 11 | `1.0` | 100 it（8.3 ep，占 21%） |
| lower (D) | 41.5% | 60 | 11 | `1.0` | 100 it（8.3 ep，占 14%） |
| right | 6.4%（兼 left 7.1%） | 20 | 11 | `1.0` | 100 it（4.3 ep，占 22%） |

front 的 `--warmup-epochs 0.0` 是刻意的：100 个 iteration 在 15 epoch 的跑法里占 8.3 个
epoch（56%），LR 永远到不了峰值，那这个 run 就没意义了。

left 视角**没有单独命令**：它复用 right checkpoint（right 模型 + 图像翻转），所以 4 条命令
就是全部 5 个视角。

lower 的单视角问题 40% 以上是新信息、且 306/470 的漏检都在这里，`--freeze 11` 可能不够；
松绑要一档一档来：`11` → `10`（57.0%）→ `8`（76.7%），跨档可训练参数跳得很大。

为什么这四个 epoch 数加起来才 135：`tooth_bbox_adj` 里 **87.3%（7815/8955）的框就是模型
自己的输出**原样当标签用，真正改变的决策只有 1403/9218（15.2%）。跑的 epoch 越多，模型越是
在记忆那 15% 的改动、同时在 84.7% 已经正确的框上漂移。

训练单个视角：

```powershell
# 全部用默认值，最简写法
python -m yolo_finetune.train --view front --device 0

# 显式写全（等价）
python -m yolo_finetune.train `
  --data-root /path/to/intraoral `
  --checkpoint-dir /path/to/.checkpoints/segtooth_model `
  --annotation-root /path/to/intraoral_adjust/tooth_bbox_adj `
  --view front `
  --epochs 120 `
  --batch 8 `
  --nbs 8 `
  --imgsz 1024 `
  --patience 30 `
  --val-interval 10 `
  --save-period -1 `
  --mosaic 0.0 `
  --scale 0.3 `
  --device 0 `
  --project exp `
  --swanlab-workspace zykev `
  --swanlab-project dentmap `
  --swanlab-name front_finetune
```

依次训练全部四个模型：

```powershell
python -m yolo_finetune.train `
  --view all `
  --device 0 `
  --swanlab-workspace zykev `
  --swanlab-project dentmap `
  --swanlab-name tooth_detector
```

使用 `--view all` 时，SwanLab run name 会自动追加 `_front/_upper/_lower/_right`。
训练会固定启用 SwanLab；`--dry-run` 只检查数据，不会创建 SwanLab run。

SwanLab 参数默认值：

| 参数 | 默认值 |
|---|---|
| `--swanlab-workspace` | `zykev` |
| `--swanlab-project` | `dentmap` |
| `--swanlab-name` | `yolo_finetune_v1`（`--view all` 时自动追加视角后缀） |

## SwanLab 指标

### 文件布局

一个视角一个实验文件夹，所有属于它的东西都在里面：

```
exp/
├── segtooth_model/                      # 导出后的 SegmentAnyTooth 兼容权重
│   ├── segmentanytooth_yolo11_front.pt
│   └── segmentanytooth_yolo11_upper.pt ...
├── yolo11_front_finetune/
│   ├── args.yaml  results.csv         # Ultralytics 原生输出
│   ├── weights/{best,last}.pt
│   └── swanlab/run-20261013_144733-qbpdvh4j/   # SwanLab 本地记录
├── yolo11_upper_finetune/ ...
├── yolo11_lower_finetune/ ...
└── yolo11_right_finetune/ ...
```

SwanLab 默认会把 `run-*` 直接摊在 `log_dir` 下面，所以 `train.py` 显式传了
`log_dir=exp/yolo11_<view>_finetune/swanlab`（不传的话四个视角的记录全混在
`exp/` 根目录里，分不出谁是谁）。

伴随这个改动，`train_args` 里加了 `exist_ok=True`。原因：SwanLab 在
`swanlab.init` 时就建好了 `exp/yolo11_<view>_finetune/swanlab/`，于是
`exp/yolo11_<view>_finetune/` 已经存在；而 Ultralytics 的 `get_save_dir()` 默认走
`increment_path(..., exist_ok=False)`，会把权重写到 `..._finetune2/`，和记录目录
分家。钉住 `exist_ok=True` 之后两边永远是同一个文件夹。副作用：
**同一个视角重复训练会在原地刷新那个文件夹**（`results.csv`、`weights/` 被覆盖），
不再堆积 `..._finetune2`、`..._finetune3`。想保留多次实验就用不同的 `--project`。

`swanlab sync exp/yolo11_<view>_finetune/swanlab/run-<...>` 可以把本地记录补传到云端。

每个 epoch 记录：

- `train/box_loss`、`train/cls_loss`、`train/dfl_loss`。
- `val/box_loss`、`val/cls_loss`、`val/dfl_loss`。
- `metrics/precision(B)`、`metrics/recall(B)`。
- `metrics/mAP50(B)`、`metrics/mAP50-95(B)`。
- `best_fitness`，它等于 `metrics/mAP50-95(B)`，是挑 `best.pt` 和早停的依据。
- 各 optimizer parameter group 的 learning rate。

这些验证指标由 Ultralytics detection validator 计算。训练结束后，最后一次验证指标
和 `best_fitness` 会再以 `final/` 前缀记录一次（SwanLab 没有 W&B 那种 run summary
对象，所以写成同 step 的标量），并且每个 epoch 也会记录 `best_fitness` 曲线。
checkpoint 不作为 SwanLab 的 artifact 上传。

**默认每 `--val-interval`（`10`）个 epoch 验证一次**，最后一个 epoch 必定验证。
验证集是按 `subject_id` 分组切出的那 20%：front/upper/lower 各 22 张、right 44 张。
控制台在验证的 epoch 打印一行汇总（`all  <图数>  <框数>  P  R  mAP50  mAP50-95`），
完整数值写进 `exp/yolo11_<view>_finetune/results.csv`。注意这一行是**按类别平均**的，
训练过程中不打印 per-class AP（Ultralytics 只在非训练模式且 `--verbose` 时打印）；
要看每个 FDI 的 AP 需要单独跑一次 val。

跳过验证的 epoch 仍会写一行 `results.csv` 和一条 SwanLab 曲线，只是数值沿用上一次
验证的结果（曲线呈台阶状），所以横轴 epoch 始终连续。这样做是因为一次验证在
front/upper/lower 上相当于 22 张验证图，而训练集只有 90 张，成本接近 1/4 个训练
epoch，而验证结果在这里只服务两件事：挑 `best.pt` 和早停。真正的逐 FDI 精度看
`estimate_accuracy.py`，它跑在导出后的权重上。`--val-interval 1` 可以恢复
Ultralytics 原始的“每个 epoch 都验证”。

早停仍然按 **epoch** 计：`patience=30` 表示连续 30 个 epoch 没有刷新最好的
mAP50-95 就停，而不是“连续 30 次验证”。`--val-interval` 只是降低测量密度，
所以不要因为验证变稀就把 `--patience` 乘上间隔；但也不要把 `--patience` 设得
比 `--epochs` 还大（那就等于关掉早停）。

训练结束**不会**再额外做一次“最终验证”。Ultralytics 原生的 `final_eval()` 会用
`best.pt` 重新走一遍 `check_det_dataset`，而本项目的 `data` 是内存哨兵值
`in_memory_tooth_dataset`，因此该调用必定抛
`FileNotFoundError: Dataset 'in_memory_tooth_dataset' for task=detect not found`，
位置恰好是全部 epoch 跑完之后，会连带跳过 checkpoint strip 和 `on_train_end`。
`train.py` 的 `ToothDetectionTrainer.final_eval` 已覆盖为只做 strip（保留
`train_results` 汇总），所以不会有这个报错。

项目自己的逐 FDI 精度口径是 `estimate_accuracy.py`（correct / loc_err / relabel /
fp / miss），和这里的 mAP 不是一回事，微调前后对比请看它。

### checkpoint 保存规则

每个 epoch 结束后（`_do_train` 里 validate → 写 CSV → `save_model()` 的顺序）：

| 文件 | 何时写 | 说明 |
|---|---|---|
| `weights/last.pt` | **每个 epoch 都写** | 最新一次的状态，始终是“可续训”的完整快照 |
| `weights/best.pt` | 只在刷新历史最好 `fitness` 时写 | `fitness` 就是 `metrics/mAP50-95(B)`，不是 loss |
| `weights/epochN.pt` | 当 `--save-period N > 0` 且 `N % save_period == 0` | 拉长训练时防丢点，`-1` 关闭 |

写入内容包括：FP16 的 EMA 权重、optimizer 状态、AMP scaler、`best_fitness`、
`train_metrics`、`train_results`（整个 `results.csv`）、`train_args`、git 信息。
EMA 权重里出现 NaN/Inf 时 `save_model()` 会跳过这一 epoch 并打 warning，而不是写坏
文件。训练全部结束后 `final_eval` 会对 `best.pt`/`last.pt` 各做一次
`strip_optimizer`：去掉 optimizer/EMA/scaler，只留 FP16 的 EMA 权重，
`epoch` 置 `-1`，文件缩到 ~5.4 MB，这才是可以拿去推理的权重。

于是 `exp/segtooth_model/segmentanytooth_yolo11_<view>.pt` 就是 `best.pt` 的一份
拷贝（`model.py::export_segtooth_checkpoint` 直接复制 `trainer.best`）。换句话说
导出的一定是**验证过的最好的那次**，而不是最后一个 epoch。

注意 `best.pt` 一开始是空的：只有当某次验证真的刷新了最好成绩才会出现，所以
`best_fitness` 在第一个验证 epoch 之前是 `None`。

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

### 微调后怎么量精度

直接用 `yolo_finetune/eval_fintune.sh`（等价于下面三条命令，并且自己会 `cd`
进 `.datasets/`）：

```bash
bash yolo_finetune/eval_fintune.sh
```

`intraoral_adjust/unfiltered_bbox/` 是**冻结基线**（原始 checkpoint 的预测，
`estimate_accuracy.py` 默认拿它做对照），**不要覆盖**。用微调后的权重重新推理时
写到 `unfiltered_bbox_ft/`（`build_tooth_bbox_predict.py` 的默认 `--out`）：

```bash
cd /home/zychen/Documents/intraoral_code/intraoral/.datasets
DATA=/home/zychen/Documents/intraoral_code/intraoral/yolo_finetune/data

python $DATA/build_tooth_bbox_predict.py \
    --weight_dir /home/zychen/Documents/intraoral_code/intraoral/exp/segtooth_model \
    --out intraoral_adjust/unfiltered_bbox_ft
python $DATA/estimate_accuracy.py --pred intraoral_adjust/unfiltered_bbox_ft
```

这个 `cd` 不能省：两个脚本的 `--base_dir` / `--out` / `--pred` / `--adj` 都是相对
**数据集根目录**（`.datasets/` → `/disk1/work/zychen/Datasets`）解析的。在仓库根目录
跑，`intraoral/zeyu` 会变成 `<repo>/intraoral/zeyu`，直接
`FileNotFoundError: .../intraoral/intraoral/zeyu`。

原始基线（8715 pred / 8955 GT，7880 correct = 88.00% of GT；miss 470，其中 D 视角
单独占 306）见 `intraoral_adjust/reports/accuracy_per_sample.csv`。注意该报告覆盖全部 112 个
case，其中 90 个在训练集里，所以微调后这个数字会偏乐观。

真正没见过的数据要看 `yolo_finetune/data/eval_holdout.py`（train.py 留出的那 22 个
subject）。目前结论：U 视角是唯一干净的提升，D 是「用一个 miss 换一个 loc_err」的
权衡，F/L/R 在留出集上是退步的。

### 在一个新批次上做前后对比（人工看图验证）

`yolo_finetune/data/compare_predictions.py` 把微调后的模型跑在一个新批次上，和**原始
checkpoint 的预测**逐 (case, view) 做差异统计，然后把差得最狠的样本并排画出来，方便
人工判断谁对：

```bash
cd /home/zychen/Documents/intraoral_code/intraoral
PYTHONPATH=$PWD python yolo_finetune/data/compare_predictions.py \
    --batch intraoral/henry/20251217_process \
    --top 25
```

`--batch` 也可以直接给仓库外的绝对路径，脚本会自己找 `.datasets` 根。

「before」不用去解析 `preview/` 的 PNG：`process_view` 传给
`save_preview_original_style` 的 `boxes_data` 是 `predict_boxes` 的**未过滤**原始输出
（`TARGET_TEETH_PER_VIEW` 是在后面的 `build_tooth_records` 里才滤的），所以
`preview/` 里的黄框 == 用原始 checkpoint 重跑一遍 `build_tooth_bbox_predict.py`
的结果。脚本第 3 步会拿重跑结果按视角白名单过滤后和批次里已有的
`tooth_bbox/` 对一遍，打印一致率：**100% 说明你对照的那份 preview 还是当前的**，
低于 100% 说明存的 preview 已经过期了。

差异分类（同一个 FDI 在两边都有就看 IoU）：

| 类别 | 含义 |
| --- | --- |
| `same` | 同一 FDI，IoU ≥ `--iou-ok`（默认 0.9） |
| `moved` | 同一 FDI，框动了 |
| `relabel` | 框还在原地但 FDI 变了（剩下的框按 IoU ≥ `--iou-match`=0.3 贪心配对） |
| `removed` | 只在 before 里 |
| `added` | 只在 after 里 |

排序键 `score = 3*(added + removed + relabel) + moved`，另外给一个归一化的
`churn = score / n_before`。增删改 FDI 权重给 3 倍，因为框挪一下后面裁出来多半还是
同一颗牙，而漏掉或者标错牙就不是了。

输出（默认 `exp/cmp_<owner>_<date>/`，`exp/` 已在 `.gitignore` 里）：

- `index.html` — 直接打开，按 score 从高到低排列，每张图上下两个 panel：左
  BEFORE（原始 checkpoint）、右 AFTER（微调），颜色见下图例
  （绿=没变，橙=动了，紫=换 FDI，红=只在 before，蓝=只在 after，虚线的要么是
  removed 要么是 added）。绿/橙/紫在两边都会出现，所以左右对着看就能发现
  「这个框原来看得到现在没了」。
- `report.csv` — 每个 (case, view) 一行，含 `score` / `churn` / 各类计数。
- `diff.json` — 逐颗牙的明细（含 confidence 变化）。
- `verify_vs_tooth_bbox.csv` — 上面第 3 步的一致率明细。

常用开关：`--ft-views U D` 只让这两个视角用微调权重（其余视角两边都用原始权重，
差异恒为 0），用来模拟「只上线 U/D」的效果；`--render-mode full` 画整图而不是裁到
框的区域；`--max-panel-px` 控制输出 PNG 大小；`--skip-detect` 复用已有的
`before/` `after/`；`--cases` 只跑指定 case（调试用）。

注意：`build_tooth_bbox_predict.py` 每次都重新 `YOLO(...)` 载入权重，所以这一步是
纯 IO/加载瓶颈（270 张图大约要十几分钟）。

## 重要训练约定

- 类别 ID 不由 JSON 排序重新生成，而是从原 checkpoint 的 `model.names` 读取，
  从而保持预训练 Detect head 与 FDI 类别一一对应。
- 标注里的每个 FDI 只要 checkpoint 能表达就都会参与训练，**不再**按
  `TARGET_TEETH_PER_VIEW` 缩减；因此 F / R / (翻转的) L 会带上跨视角的牙，
  训练监督比推理产物更宽。
- L 图的类别映射严格复用 `segtooth_new.py` 的语义：标注 FDI 在
  `LEFT_CLASSES` 中的位置就是训练 class ID。例如 `LEFT_CLASSES[0] == "le28"`，
  则 L 图的 FDI 28 使用 class ID 0。这与推理时
  `fdi = LEFT_CLASSES[predicted_class_id]` 完全一致。
- `right` 固定同时加载 R 和 L，L 图只在内存中翻转。
- 随机水平/垂直翻转被强制关闭。FDI 类别包含左右位置语义，普通 flip augmentation
  不会自动交换类别，会产生错误监督。
- `mosaic` 默认关闭（`--mosaic 0.0`）。拼图会打乱决定 FDI 的全局牙弓布局，并
  把本就稀疏的 D/U 图分辨率减半；要开启必须给出替代的全局上下文方案。
- `nbs` 默认等于 `batch`，否则 Ultralytics 的梯度累积会把每 epoch 的 optimizer
  step 压到 `images/64`（90 张的视角只有 1.4 步）。
- 优化器仍是 Ultralytics 的自动选择（本项目数据量下为 AdamW，`lr0` 由 `nc` 推出）；
  数据增强里除了上面显式列出的参数，其余沿用 Ultralytics 默认值。
- Ultralytics 的 RAM/disk image cache 均关闭，尤其不会在源图片旁生成 `.npy`。
- 训练时仅产生正常 run 输出，例如 `weights/best.pt`、`weights/last.pt`、
  `results.csv` 和 `args.yaml`；训练可视化 plots 默认关闭。
