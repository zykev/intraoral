# AlphaDent → single-tooth 数据集转换

把 AlphaDent 原始分割数据集改造成「单牙」格式，供下游病害检测/分类训练使用。

两个脚本，按顺序执行：

| 脚本 | 作用 |
|------|------|
| `infer_view.py` | 用 Qwen-VL 判断每张全口照的**拍摄方向**（F/R/L/U/D）和**需要顺时针旋转的角度**，写入 `view_labels.json` |
| `build_single_tooth.py` | 按方向调用 `segtooth.py` 的 YOLO-seg 模型检测牙齿 → 裁单牙图 → 把原 txt 多边形换算到单牙图坐标系 → 写 `images_tooth/` + `labels_tooth/`（每个样本目录里还保存一份正方向整图 `<stem>.jpg`） |

## 输入数据结构

```
<base_dir>/                      # 默认 .datasets/intraoral_public/alphadent
  images/{train,valid,test}/*.jpg
  labels/{train,valid}/*.txt      # YOLO-seg: cls x1 y1 x2 y2 ... (归一化)
```

## 输出数据结构

```
<base_dir>/
  view_labels.json                # 方向推断结果
  images_tooth/<split>/<stem>/<stem>.jpg       # 正方向后的整图（与原图同名）
  images_tooth/<split>/<stem>/<fdi>.png        # 单牙裁剪图（train/valid）
  labels_tooth/<split>/<stem>.json             # 单牙标注（train/valid）
  single_tooth_report.json                     # 汇总报告
  single_tooth_vis/<split>__<stem>.jpg         # 调试可视化（可选）
```

每个样本目录是自包含的：`<stem>.jpg` 是**旋转矫正后的整张全口照**（无论 `rotation` 是 0
还是 90/180/270 都会写），`<fdi>.png` 都是它的裁剪块。因此 `crop_bbox`、`bbox_xyxy`、
`rot_size` 这些整图坐标可以直接画在 `<stem>.jpg` 上验证。JPEG 质量由 `--upright_quality`
控制（默认 95）。

`labels_tooth/<split>/<stem>.json` 关键字段：

```jsonc
{
  "image": "images/train/p002_F_52_001.jpg",
  "upright_image": "images_tooth/train/p002_F_52_001/p002_F_52_001.jpg",
  "stem": "p002_F_52_001",
  "split": "train",
  "view": "F",                       // F/R/L/U/D
  "view_confidence": 1.0,
  "rotation": 0,                     // 原图顺时针旋转这么多度后为正方向
  "rotation_trials": {"0": 24, "90": 1, "180": 20, "270": 0},
  "orig_size": [5472, 3648],         // 原图 [w, h]
  "rot_size": [5472, 3648],          // 旋转后 [w, h]，若 rotation 是 90/270 则宽高互换
  "num_labels": 24,                  // 原 txt 多边形数
  "num_detections": 20,              // 通过 FDI 过滤后的检测数
  "num_orphan_polygons": 3,          // 未分配到任何牙齿的多边形
  "teeth": [
    {
      "crop_path": "images_tooth/train/p002_F_52_001/11.png",
      "fdi": 11,
      "label_ids": [2],
      "label_names": ["Crown"],
      "crop_size": [883, 1230],      // [w, h]，等于 PNG 实际尺寸、等于 crop_bbox 宽高
      "crop_bbox": [2167, 655, 3050, 1885],     // 正方向整图坐标系 xyxy
      "bbox_xyxy": [2238.1, 740.6, 2990.4, 1800.6],   // 检测框，正方向整图像素
      "confidence": 0.9689,
      "clipped": false,
      "polygons": [
        {
          "label_id": 2,
          "label_name": "Crown",
          "points": [[384.0, 66.0], ...],     // 单牙图坐标系（像素），0 <= x <= crop_size[0]-1
          "area_ratio_inside": 1.0
        }
      ]
    }
  ]
}
```

### 坐标系约定（重要）

所有几何量都位于**旋转后的正方向坐标系**，原图像素坐标不会出现在任何输出里：

```
原图 --rotate(rotation)--> 正方向整图 --crop(crop_bbox)--> 单牙图
                │                                    │
                └── 多边形同步 rotate_points ─────────┴─ 减 crop 原点 --> polygons[].points
```

| 字段 | 坐标系 |
| --- | --- |
| `images_tooth/.../<stem>.jpg` | 正方向整图（所有整图坐标都指向它） |
| `images_tooth/.../<fdi>.png` | 正方向整图的裁剪块（内容已直立） |
| `polygons[].points` | **单牙图自身**坐标系，可直接画在对应 PNG 上，`0 <= x <= crop_size[0]-1`、`0 <= y <= crop_size[1]-1` |
| `crop_size` | 单牙图 `[w, h]`，与 PNG 尺寸、`crop_bbox` 宽高一致 |
| `crop_bbox` / `bbox_xyxy` | 正方向整图，像素 xyxy |
| `rot_size` | 正方向整图 `[w, h]`（90°/270° 时宽高互换） |
| `orig_size` + `rotation` | 唯一可用于逆变换回原图的字段 |

`points` 是「正方向整图坐标 − `crop_bbox` 左上角」得到的，**没有任何缩放**；
反变换就是 `points + crop_bbox[:2]`。

下游训练**不需要再做任何旋转**。若要映射回原图 txt 标注：先 `+ crop_bbox 原点`，再按 `rotation` 逆变换（`0: (x, y)`；`90: (y, H-1-x)`；`180: (W-1-x, H-1-y)`；`270: (W-1-y, x)`，其中 `W/H` 为 `orig_size`，`(x, y)` 为旋转后坐标）。这与 `rotate_points` 严格互为逆运算（`cv2.rotate` 的像素映射是 `X = W-1-x`）。

- 一张单牙图可能对应多条标注（多颗病灶），都放在同一个 `teeth[i].polygons` 里。
- 单牙图命名固定为 `<fdi>.png`，同一张全口照内 FDI 唯一，不会重名。
- **默认导出所有检测到的单牙**，包括没有病灶标注的：这些牙齿的 `polygons` 和 `label_ids` 为空列表（负样本），图片照常写在 `images_tooth` 里。如果不想要负样本，加 `--only_annotated`。

`single_tooth_report.json` 里的关键汇总字段：

| 字段 | 说明 |
|---|---|
| `images` | 已处理的整图数 |
| `teeth` | 导出的单牙总数 |
| `teeth_with_labels` | 其中有病灶标注的牙数 |
| `teeth_without_labels` | 负样本牙数 |
| `polygons` | 写入 `labels_tooth` 的多边形总数 |
| `views` / `rotations` / `labels` | 方向、旋转角、类别分布 |
| `missing_view_labels` | 还没跑 `infer_view.py` 的图 |
| `failures` | 处理出错的图 |

## 运行

先确认 Python 环境有 `opencv-python`, `numpy`, `ultralytics`, `torch`, `requests`（本仓库用 conda 环境 `intraoral`）。

### 1. 推断方向

```bash
# 小规模试跑
python process_public/alphadent/infer_view.py --limit 20 --splits train valid

# 全量（train + valid + test）
python process_public/alphadent/infer_view.py --splits train valid test --num_workers 8

# 查看统计 / 低置信度样本
python process_public/alphadent/infer_view.py --stats
```

- 每次请求会把原图 4 个旋转角度拼成 2×2 网格发给 VLM，让模型先判方向、再选哪一格是正方向，比直接问「转多少度」稳得多。
- 断点续跑：默认跳过 `view_labels.json` 里已有的条目，加 `--overwrite` 重跑。
- 调 prompt 可以先用 `--dry_run --preview_dir /tmp/grid` 把网格图存下来看。

### 2. 构建单牙数据集

```bash
# 小规模试跑（加可视化）
python process_public/alphadent/build_single_tooth.py --limit 20 --save_vis

# 全量
python process_public/alphadent/build_single_tooth.py --splits train valid

# 汇总报告
python process_public/alphadent/build_single_tooth.py --report
```

- 默认权重目录：`exp/segtooth_model/`（四级视图的 finetune 权重），不存在时回落到 `.checkpoints/segtooth_model/`。可用 `--weight_dir` 指定。
- `--rotation_mode search`（默认，**推荐**）：在 `0/90/180/270` 四个角度里各跑一遍检测，用「标注多边形被检测框覆盖的程度」打分，取最优。
  - 打分键（越大越好）：`(覆盖率≥min_assign_ratio 的多边形个数, 平均覆盖率, 左右手性, 检测牙数, 置信度和)`。
  - 用多边形覆盖而不是「检测牙数」当主键，是因为 YOLO 在错误角度上照样能检出一堆牙，只有角度正确时病灶多边形才会落进对应的牙框里（实测 10/10 正确，而按牙数打分会选错，见下文）。
  - 平局时优先靠前的角度（即优先 `0`），保证本来就正着的图不会被无谓旋转。
- `--rotation_mode vlm`：直接用 `infer_view.py` 里 VLM 给出的旋转角（**不推荐**，VLM 判方向很准但判「哪一格是正的」会出错）；`none`：完全不旋转。
- 裁剪框会先把检测框按 `--pad_ratio` 外扩，再继续外扩直到包住所有分配到的多边形（外加 `--poly_margin_ratio` 边距），保证标注不会被打断。

### 常用参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--min_conf` | 0.35 | 检测置信度阈值 |
| `--min_area` | 500 | 检测框最小面积（原图像素） |
| `--pad_ratio` | 0.08 | 检测框四周额外留白比例 |
| `--poly_margin_ratio` | 0.02 | 多边形外扩边距（相对裁剪框短边） |
| `--min_assign_ratio` | 0.5 | 多边形落在检测框内的最小面积比，低于则视为 orphan |
| `--only_annotated` | off | 只导出有标注多边形的牙（丢弃负样本）|
| `--no_fdi_filter` | off | 不做 FDI 白名单过滤，保留所有检测 |
| `--overwrite` | off | 重新处理已存在的 `labels_tooth` JSON |

## 读数注意事项

- 文件名第 2 个 token 是**性别**（`F`/`M`），不是方向；方向完全靠 `infer_view.py` 推断。
- `num_orphan_polygons` 不为 0 通常是正常的：例如下颌咬合面（D）照片背景里能看到上颌牙齿，这些多边形属于被 FDI 白名单排除的牙位，会被记为 orphan。orphan 过多（>20%）说明方向判断或视图白名单有问题，先跑 `--stats` 和 `--save_vis` 检查。
- 原数据集没有 EXIF 方向信息，**很多图是倒着的**，所以旋转搜索是必须的。实测例子（与 `amy_new/20260703_process/process/<case>/{U,D}.png` 的正方向参考对比）：

  | 图片 | view | 存储状态 | 需要旋转 | 证据（polygon 覆盖率 0° vs 180°） |
  | --- | --- | --- | --- | --- |
  | `train/p002_F_52_002` | U | 倒置（前牙在下） | 180° | 0.75 → **0.95** |
  | `train/p002_F_52_003` | D | 倒置（前牙在上） | 180° | 0.72 → **0.90** |
  | `train/p002_F_52_004` | U | 倒置 | 180° | 0.45 → **0.80** |
  | `train/p002_F_52_005` | D | 倒置 | 180° | 0.77 → **0.87** |
  | `valid/p001_F_51_002` | U | 倒置 | 180° | 0.50 → **0.93** |
  | `valid/p001_F_51_003` | D | 倒置 | 180° | 0.82 → **0.95** |
  | `valid/p015_M_53_001` | D | 倒置 | 180° | 0.62 → **0.83** |
  | `valid/p001_F_51_001` | D | 侧躺 | 0°（已经最好） | 0°=**0.60** > 90°=0.37 |
  | `train/p002_F_52_001` | F | 正 | 0° | **0.78** vs 0.05/0.67/0.00 |
  | `valid/p001_F_51_004` | U | 正 | 0° | **0.92** vs 0.57/0.42/0.78 |

- 方向判断（`infer_view.py`）始终只看**解剖结构**（硬腭 vs 舌/口底、双牙弓 vs 单侧牙弓），不看图像边框，所以图片是倒着的也不影响 view 分类。
- 单张图直接问 VLM 判 view 会退化（实测 12 张里 2 张把 D 判成 U，因为舌背和硬腭在倒置时容易混），2×2 旋转网格反而更准，所以 `infer_view.py` 保留了网格 prompt。
- 负样本（无标注牙）里可能包含少量 YOLO 误检测（例如边缘的牙龈/牙冠碎片）。如果下游训练对负样本噪声敏感，建议用 `--only_annotated` 或者按 `confidence` 再过滤一道。
