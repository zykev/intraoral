# 微调数据准备 / Fine-tuning data preparation

把人工修正过的标注转成 YOLO11 微调用的 GT（`tooth_bbox_adj`），并量化 checkpoint
当前的精度，作为微调后的对照基线。

## 脚本

| 脚本 | 作用 
|---|---|
| `build_tooth_bbox_predict.py` | 重跑 YOLO 检测，**不加** per-view FDI 过滤；默认写到 `unfiltered_bbox_ft/`。这是标注在 `preview/` 里实际看到的预测
| `validate_adjust.py` | 体检人工标注文件本身（文件名、txt 格式、labelme 标签合法性等） 
| `build_tooth_bbox_adj.py` | `unfiltered_bbox` + 人工修正 → **GT `tooth_bbox_adj/`** 
| `visualize_adjust.py` | 红框=原始预测 / 蓝框=修正后，只画有改动的视角 |
| `estimate_accuracy.py` | 预测 vs GT → correct / loc_err / relabel / fp / miss 

## 目录约定

```
intraoral/zeyu/<batch>_process/process/<case>/<view>.png   原始图（输入）
intraoral/zeyu/<batch>_process/tooth_bbox/<case>/<view>.json  过滤版预测（仅用于元数据）

intraoral_adjust/sample/annotation.txt                     修正规则说明
intraoral_adjust/sample/<case>/{D,F,L,R,U}.{txt,json}      人工修正（输入）
intraoral_adjust/unfiltered_bbox/<case>/<view>.json        原始 checkpoint 预测（冻结基线）
intraoral_adjust/unfiltered_bbox_ft/<case>/<view>.json     重跑/微调模型预测（`--out` 默认值）
intraoral_adjust/tooth_bbox_adj/<case>/<view>.json         GT（最终产物）
intraoral_adjust/vis_adj/<case>/<view>.png                 可视化
intraoral_adjust/reports/                                  所有报告
```

### 两棵预测树（不要弄混）

- `unfiltered_bbox/` = **原始 checkpoint 的预测，已冻结**。`build_tooth_bbox_adj.py`
  读它来合成人工修正，`estimate_accuracy.py` 把它当固定基线，**所以任何时候都不该
  重写它**。`build_tooth_bbox_predict.py` 已经默认拒绝写入，除非显式加 `--force`。
- `unfiltered_bbox_ft/` = `build_tooth_bbox_predict.py` 的**默认 `--out`**。重跑、以及
  用微调后的 checkpoint 推理都写这里，这样上面的基线还留着能对比。

用微调后的模型打分（**这就是默认行为**，`--weight_dir` 默认已经指向
`<repo>/exp/segtooth_model`，`--out` 默认为 `unfiltered_bbox_ft`）：

```bash
cd /home/zychen/Documents/intraoral_code/intraoral/.datasets
DATA=/home/zychen/Documents/intraoral_code/intraoral/yolo_finetune/data

python $DATA/build_tooth_bbox_predict.py \
    --weight_dir /home/zychen/Documents/intraoral_code/intraoral/exp/segtooth_model \
    --out intraoral_adjust/unfiltered_bbox_ft
python $DATA/estimate_accuracy.py --pred intraoral_adjust/unfiltered_bbox_ft
```

要重现原始基线的预测（`unfiltered_bbox/`），把权重目录换回原始 checkpoint：

```bash
python $DATA/build_tooth_bbox_predict.py \
    --weight_dir /home/zychen/Documents/intraoral_code/intraoral/.checkpoints/segtooth_model \
    --out intraoral_adjust/unfiltered_bbox_ft
```

注意 `exp/segtooth_model` 只有在真的跑完一次微调（`python -m yolo_finetune.train`）
之后才会存在，否则脚本会直接报 `FileNotFoundError: weight_dir does not exist`。

## 有新数据进来时怎么跑

### 1. 取新标注

新的标注会传到 Google Drive，下载到 `intraoral_adjust/sample/`：

```bash
python download_from_gdrive.py     # 见仓库根目录，会跳过本地已存在的文件
```

### 2. 全流程（按顺序）

```bash
cd /home/zychen/Documents/intraoral_code/intraoral/.datasets
DATA=/home/zychen/Documents/intraoral_code/intraoral/yolo_finetune/data

# (a) 体检新标注 —— 先跑，把格式问题挡在前面
python $DATA/validate_adjust.py

# (b) 重跑检测，拿到标注实际看到的预测
#     默认写到 unfiltered_bbox_ft/，不会动冻结的 unfiltered_bbox/
python $DATA/build_tooth_bbox_predict.py          # 需 intraoral env

# (c) 合成 GT
python $DATA/build_tooth_bbox_adj.py

# (d) 人工复查改了什么
python $DATA/visualize_adjust.py --max-side 1800  # 需 intraoral env

# (e) 精度报告
python $DATA/estimate_accuracy.py
```

用 `intraoral` env 时把 `python` 换成
`/home/zychen/miniconda3/envs/intraoral/bin/python`。

### 3. 增量 vs 重建

- `(b)` 会重跑全部 view。只想补新增的 case：`--cases <id> [<id> ...]`。
  它默认写 `--out intraoral_adjust/unfiltered_bbox_ft`；`unfiltered_bbox/` 是冻结基线，
  重写会被拒绝，除非真的想重建基线时显式 `--force`。
- `(c)` 每次**全量重建** `tooth_bbox_adj`，就地覆盖，不需要手动清目录。
- `(d)` 只渲染有改动的视角，所以 `vis_adj/` 的图片集合 = 待复查清单。
- `(a)` `(c)` `(e)` 都是纯 stdlib，几秒到几十秒。

## 更新后的检查项

`(a)` `(c)` `(e)` 都会往 `intraoral_adjust/reports/` 写文件：

| 文件 | 看什么 |
|---|---|
| `validation_report.csv` | `stray_file` / `labelme_bad_label` / `txt_bad_line` 等，逐 (case,view) |
| `build_warnings.txt` | `unmatched rename source` ＝改名的源 FDI 在该视角压根没被预测出来，只能手改；`... overwrites predicted ... (IoU x.xx)` ＝改名撞车，IoU 高是重复检测，IoU 低是相邻牙需人眼确认 |
| `provenance.json` | 每个牙的来源：`kept` / `renamed` / `relocated` / `added` |
| `accuracy_per_sample.csv` | 逐样本精度 |

`build_warnings.txt` 最后几行是**唯一需要人工决定的**：源 FDI 没被预测出来的
rename/delete（无法自动合并），以及非法 FDI 目标。

## 接入微调

`train.py` / `dataset.py` 已经支持这棵平行树，**默认就读它**，而且默认路径全部走
仓库内的软链，所以在仓库根目录直接跑即可：

```bash
cd /home/zychen/Documents/intraoral_code/intraoral
python -m yolo_finetune.train --view all --dry-run   # 先 dry run，通过后去掉 --dry-run
```

等价于显式写全（也可以用存储卷绝对路径）：

```bash
python -m yolo_finetune.train \
  --data-root .datasets/intraoral \
  --checkpoint-dir .checkpoints/segtooth_model \
  --annotation-root .datasets/intraoral_adjust/tooth_bbox_adj \
  --view all --dry-run
```

相对写的 `--annotation-root` 会依次在 `--data-root` 同级目录、`--data-root` 内部、
当前目录、`.datasets/` 下查找，第一个存在的胜出；绝对路径直接用。

`--annotation-root` 的图片不放在标注旁边，所以 `dataset.py` 用每条标注自带的
`orig_size` 去 `--data-root` 下匹配像素尺寸一致的 `<subject>/<view>.png`；匹配不上或
有多个候选都会直接报错。退回旧的 per-batch 布局用 `--annotation-root none
--annotation-dir tooth_bbox_revise`。

切分按 `subject_id` 分组（同一人不会同时出现在 train/val）。

### 训练时不再过滤视角

`tooth_bbox_adj` 是完整牙弓（未经 view 过滤），训练**也刻意不过滤**：
`prepare_targets()` 保留 checkpoint class map 能表达的**全部** FDI，包括
`segtooth.py::TARGET_TEETH_PER_VIEW` 在推理后会丢掉的“跨视角”牙。目的是让 YOLO
尽量把可见的牙都检出来；per-view 白名单是推理侧的策略，不该反过来限制训练监督。

实测多出来的监督量：`front` +833 box，`right`+`left` +2461 box，`upper`/`lower`
本来就是全 arch，增加 0。细节见 `../README.md` 的“跨视角牙齿（widen）”一节。

checkpoint 没有对应 class 的 FDI（理论上不该出现）默认丢弃并在 audit 里报数
（`dropped_unknown_boxes` / `dropped_unknown_fdi`）；加 `--strict-fdi` 改为报错。
