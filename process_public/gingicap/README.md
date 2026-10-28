# GingiCap → 前牙区 sextant 严重度数据集

把 GingiCap 的原始 YOLO 标注重新组织成「每个样本两张前牙区切图 + 一个三分类标签」的数据集，用于 gingivitis 严重度分类（mild / moderate / severe）。

## 1. 原始数据

```
.datasets/intraoral_public/gingicap/
  Training/Images/<id>.jpg      Validation/Images/...      Test/Images/...
  Training/Labels/<id>.txt      Validation/Labels/...      Test/Labels/...
```

每个 `.txt` 是标准 YOLO 检测标注 `cls cx cy w h`（归一化），类别含义：

| 类别 | 含义 | 用途 |
|---|---|---|
| 0 | 上颌前牙区（sextant）框 | 只用于把牙齿分配到 S2 |
| 1 | 下颌前牙区（sextant）框 | 只用于把牙齿分配到 S5 |
| 2, 3 | gingivitis 轻度 | 标签来源 |
| 4 | gingivitis 中度 | 标签来源 |
| 5, 6 | gingivitis 重度 | 标签来源 |

统计确认：Training 732 / Validation 182 / Test 182 张图，**每张都恰好有 1 个 class-0 和 1 个 class-1 框**，且所有 class 2–6 牙齿框的中心都落在且仅落在其中一个区域框内（无遗漏、无重复）。所以「按中心点分上下前牙区」在本数据集上是无歧义、100% 覆盖的。

## 2. 切分逻辑（对齐 `segtooth.py` 的 front 视图）

`segtooth.py` 中 front 视图对应两个 sextant：

- `S2` = 上颌前牙（FDI 13–23）
- `S5` = 下颌前牙（FDI 33–43）

GingiCap 的牙齿框没有 FDI 信息，因此这里用 class-0/1 区域框做「上/下前牙区」判定，再完全复用 `segtooth.py` 的裁剪几何：

1. 取中心点落在该区域的 class 2–6 牙齿框；
2. 对这些框求 union；
3. 套用 `segtooth.py` 的 `pad_sextant_box`（front）方向性留白：
   - 上颌 S2：上方留 `0.4 × h`（牙龈），下方留 `0.05 × h`，左右各 `0.1 × w`；
   - 下颌 S5：上方留 `0.05 × h`，下方留 `0.4 × h`，左右各 `0.1 × w`。

> `--crop_mode region` 可改为直接用人工标注的 class-0/1 区域框裁剪，不额外 padding。
> 两种方式都只影响图片内容，不改变标签。

## 3. 标签规则

每个 sextant 切图取该区域内**所有 class 2–6 牙齿框的最大类别**，再映射成三分类：

```
2, 3        -> mild      (class_id 0)
4           -> moderate  (class_id 1)
5, 6        -> severe    (class_id 2)
```

这个 2/3→mild、4→moderate、5/6→severe 的映射与仓库里已有的
`gingivitis_{train,test,coco}.json`（categories: `gingivitis_mild / moderate / severe`）完全一致：
`1462 = 244(class2)+1218(class3)`、`2178(class4)`、`5150 = 3399(class5)+1751(class6)`。

## 4. 输出结构

```
.datasets/intraoral_public/gingicap_sextant/
  training_sextant/
    Images/<sample_id>/S2.jpg     # 上颌前牙区
    Images/<sample_id>/S5.jpg     # 下颌前牙区
    labels.json
  validation_sextant/
    Images/...  labels.json
  test_sextant/
    Images/...  labels.json
  distribution.json               # 各 split 分布汇总
  labels.csv                      # 扁平表，方便 pandas 读取
```

`labels.json`（每个 split 一个）：

```jsonc
{
  "source_split": "Training",
  "crop_mode": "segtooth",
  "label_mapping": {"2": "mild", "3": "mild", "4": "moderate", "5": "severe", "6": "severe"},
  "labels": {"mild": 0, "moderate": 1, "severe": 2},
  "sextants": {"S2": "...", "S5": "..."},
  "aggregation": "max severity tooth class inside the sextant",
  "num_samples": 732,
  "num_images": 1464,
  "distribution": {"mild": 82, "moderate": 236, "severe": 1146},
  "samples": {
    "00001": {
      "S2": {
        "image": "Images/00001/S2.jpg",
        "label": "moderate",
        "class_id": 1,
        "max_source_class": 4,
        "source_classes": [3, 3, 3, 3, 4, 4],
        "source_class_counts": {"3": 4, "4": 2},
        "num_teeth": 6,
        "crop_source": "tooth_union+padding",
        "crop_box_xyxy": [1260, 983, 5986, 2235],   // 原图像素坐标
        "crop_size_wh": [4726, 1252]
      },
      "S5": { ... }
    }
  }
}
```

- `source_classes` 保留了原图该 sextant 里所有牙齿的类别，方便做标签审计或换聚合规则。
- `crop_box_xyxy` 是原图坐标系，可直接画回原图核对。

## 5. 运行

```bash
# 全量构建（约 1096 张原图 -> 2192 张切图）
python process_public/gingicap/build_sextant_dataset.py --overwrite

# 只跑前 20 个样本检查
python process_public/gingicap/build_sextant_dataset.py --limit 20 --overwrite --output /tmp/gsc

# 不下图、只重算分布
python process_public/gingicap/build_sextant_dataset.py --stats_only

# 可选参数
#   --crop_mode segtooth|region   切图方式（默认 segtooth）
#   --max_side 1600               长边下采样，控制磁盘占用（默认 0 = 原分辨率）
#   --jpeg_quality 95
```

依赖：`Pillow`、`tqdm`（不需要 ImageMagick）。原图 7286×4863，默认原分辨率输出约 **2.8 GB**。
如需减小体积可用 `--max_side 1600` 之类做长边下采样。

## 6. 标签分布（max 规则）

单位是 sextant 切图张数（每张原图贡献 2 张）。

| Split | 切图数 | mild | moderate | severe |
|---|---:|---:|---:|---:|
| training | 1464 | 82 (5.6%) | 236 (16.1%) | 1146 (78.3%) |
| validation | 364 | 17 (4.7%) | 71 (19.5%) | 276 (75.8%) |
| test | 364 | 20 (5.5%) | 84 (23.1%) | 260 (71.4%) |
| **合计** | **2192** | **119 (5.4%)** | **391 (17.8%)** | **1682 (76.7%)** |

按 sextant 拆开看，S2/S5 的分布也基本一致（severity 偏向一致），说明不是某个区域单独把 severe 拉高的。

### 换一种聚合方式会怎样？

同样按区域分配牙齿，只改「区域 -> 单标签」的聚合规则（ordinal: mild=0/moderate=1/severe=2）：

| 聚合规则 | training mild/mod/sev | validation mild/mod/sev | test mild/mod/sev |
|---|---|---|---|
| **max（当前）** | 5.6 / 16.1 / **78.3** | 4.7 / 19.5 / **75.8** | 5.5 / 23.1 / **71.4** |
| mean（四舍五入） | 12.6 / 26.2 / 61.2 | 10.7 / 39.0 / 50.3 | 11.3 / 42.9 / 45.9 |
| median | 13.4 / 23.7 / 62.9 | 11.5 / 36.0 / 52.5 | 11.8 / 41.2 / 47.0 |
| majority vote | 18.0 / 21.1 / 60.9 | 14.8 / 33.0 / 52.2 | 14.6 / 39.3 / 46.2 |

## 7. 这种规则合理吗？

**结论：映射本身合理，但「max」聚合会让 severe 占比高达 ~77%，训练时需要显式处理类别不平衡。**

合理的部分：

1. **映射有外部一致性**。2/3→mild、4→moderate、5/6→severe 与数据集自带的 COCO 三分类导出完全吻合，不是拍脑袋的切分。
2. **max 符合临床语义**。sextant 级牙周/牙龈严重度评估通常看「最重位点」，只要区域内有一颗牙是 severe，就应把该 sextant 判为 severe，漏报 severe 比误报代价更高。
3. **区域划分无歧义**。class 0/1 边界在数据上 100% 覆盖所有牙齿框，没有「既不在上也不在下」或「同时落在两边」的样本，因此标签可复现。
4. **不会产生空标签**。所有样本的两个 sextant 都有至少一颗牙。

需要注意的问题：

1. **严重不平衡**：mild 只有 5.4%，severe 76.7%。mild 类在 validation/test 各只有 17/20 张，指标方差会很大。建议：
   - 训练用 class-weighted / focal loss / balanced sampler；
   - 报告 macro-F1、每类 recall，而不是只看 accuracy（全判 severe 就有 ~77% accuracy）；
   - 或者考虑合并 mild+moderate 成「non-severe」做二分类对照实验。
2. **max 会放大局部噪声**：只要标注里有一颗牙被标成 severe，整个 sextant 就是 severe。数据集中 class 6 本身很多（1751 个框），这是 severe 占主导的直接原因。如果需要「区域整体严重度」，mean/median/majority 更平滑（severe 会降到 ~46–63%），但会稀释真正的高危区域。
3. **标签是 ordinal 的**：mild < moderate < severe 有顺序关系。如果模型允许，用 ordinal regression 或 CORAL 损失通常比纯 softmax 更贴合任务。
4. **同一患者多张图可能跨 split**：GingiCap 的 split 是官方给的，但若同一患者的多张照片分布在不同 split，会有轻微泄漏。做严格评估时建议按患者/编号前缀再核查一遍。

如果要更稳妥，建议保留当前 `max` 版本作为主数据集（与临床「最重位点」一致），同时用脚本里的 `source_classes` 字段离线生成 mean/median 版本做消融，不需要重新裁图。

## 8. 相关文件

| 文件 | 作用 |
|---|---|
| `build_sextant_dataset.py` | 构建脚本（切图 + 标签 + 分布统计） |
| `../../segtooth.py` | sextant 定义与 padding 参数来源 |
