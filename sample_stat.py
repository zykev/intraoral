# %%
import os
import json
from collections import defaultdict

def process_dental_dataset(root_path, output_txt="unhealthy_details.txt"):
    # --- 初始化统计容器 ---
    
    # 1. 图片级统计
    img_stats = {
        'total': 0,
        'healthy': 0,
        'unhealthy': 0
    }
    
    # 2. 样本级统计
    sample_tracker = defaultdict(set)
    
    # 3. 不健康类别详细记录 (修改处：使用字典列表存储路径)
    # 结构: {'Label名称': ['图片路径1', '图片路径2', ...]}
    label_details = defaultdict(list)

    print(f"开始处理目录: {root_path} ...")

    # --- 遍历文件 ---
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith(".json"):
                json_path = os.path.join(dirpath, filename)
                
                # 假设对应的图片是同名的 png 文件
                # 如果你的图片格式是 jpg，请修改这里为 .jpg
                img_filename = filename.replace(".json", ".png")
                img_path = os.path.join(dirpath, img_filename)

                # 路径解析 Sample ID
                try:
                    parent_dir = os.path.dirname(dirpath)
                    sample_id = os.path.basename(parent_dir)
                except Exception:
                    sample_id = "Unknown"

                img_stats['total'] += 1
                
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    flags = data.get('flags', {})
                    is_unhealthy = flags.get('1', False)

                    if is_unhealthy:
                        img_stats['unhealthy'] += 1
                        sample_tracker[sample_id].add('unhealthy')
                        
                        # 统计具体的病灶类别并记录路径
                        shapes = data.get('shapes', [])
                        for shape in shapes:
                            label = shape.get('label')
                            if label:
                                # 将该图片的路径添加到对应 Label 的列表中
                                label_details[label].append(img_path)
                    else:
                        img_stats['healthy'] += 1
                        sample_tracker[sample_id].add('healthy')

                except Exception as e:
                    print(f"读取或解析文件失败 {json_path}: {e}")

    # --- 汇总样本级统计 ---
    sample_summary = {'total': len(sample_tracker), 'healthy': 0, 'unhealthy': 0}
    for s_id, statuses in sample_tracker.items():
        if 'unhealthy' in statuses:
            sample_summary['unhealthy'] += 1
        else:
            sample_summary['healthy'] += 1

    # --- 控制台输出基础统计 ---
    print("\n" + "="*40)
    print("           数据统计概览           ")
    print("="*40)
    print(f"总 Sample 数: {sample_summary['total']} (不健康: {sample_summary['unhealthy']})")
    print(f"总 图片 数  : {img_stats['total']} (不健康: {img_stats['unhealthy']})")
    print("-" * 40)

    # --- 保存详细报告到 TXT 文件 ---
    try:
        # 按数量从多到少排序
        sorted_labels = sorted(label_details.items(), key=lambda item: len(item[1]), reverse=True)

        with open(output_txt, "w", encoding="utf-8") as f:
            f.write("="*50 + "\n")
            f.write("        口腔数据不健康类别详细统计报告\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"统计时间: {os.path.abspath(output_txt)}\n")
            f.write(f"数据源路径: {root_path}\n\n")

            if not sorted_labels:
                f.write("未检测到不健康样本数据。\n")
            
            for label, paths in sorted_labels:
                count = len(paths)
                f.write(f"【 类别 Label: {label} 】\n")
                f.write(f"  - 统计数量: {count}\n")
                f.write(f"  - 图片路径列表:\n")
                
                for idx, p in enumerate(paths, 1):
                    f.write(f"    {idx}. {p}\n")
                
                f.write("\n" + "-"*50 + "\n\n")
        
        print(f"详细统计报告已生成: {output_txt}")
        print(f"包含类别、数量及每张图片的具体路径。")

    except Exception as e:
        print(f"写入TXT文件失败: {e}")

if __name__ == "__main__":
    # 请修改为实际路径
    DATASET_DIR = r".datasets/intraoral/annosample/single_tooth"
    
    if os.path.exists(DATASET_DIR):
        process_dental_dataset(DATASET_DIR)
    else:
        print("路径不存在，请检查配置。")


# %%
import os
import json
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
random.seed(42)


label_dict = {"1": "Caries", "11": "White spot leison", "3": "Filling without caries", 
              "6": "Fissure sealant", "10": "Non-caries disease (hard tissue)", "12": "Staining", 
              "loss of fissure sealant": "Loss of fissure sealant", "abnormal central cusp": "Abnormal central cusp"}

def visualize_unhealthy_samples(root_path):
    # --- 1. 数据收集 ---
    # 结构: {'Label名称': [{'json': json_path, 'img': img_path}, ...]}
    label_data = defaultdict(list)
    allowed_labels = set(label_dict.keys())

    print(f"正在扫描数据: {root_path} ...")

    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith(".json"):
                json_path = os.path.join(dirpath, filename)
                # 假设图片是 png，如果是 jpg 请修改
                img_path = os.path.join(dirpath, filename.replace(".json", ".png"))
                
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 只有不健康的才处理
                    if data.get('flags', {}).get('1', False):
                        shapes = data.get('shapes', [])
                        for shape in shapes:
                            label = str(shape.get('label'))
                            if label in allowed_labels:
                                label_data[label].append({
                                    'json': json_path,
                                    'img': img_path
                                })
                except Exception as e:
                    continue

    if not label_data:
        print("未找到不健康样本，无法可视化。")
        return

    # --- 2. 随机采样与绘图准备 ---
    active_labels = sorted(label_data.keys())
    num_labels = len(active_labels)
    print(f"共发现 {num_labels} 个不健康类别，准备生成可视化组图...")

    # 计算网格行列 (例如 10个类别 -> 3行4列)
    cols = 4
    rows = math.ceil(num_labels / cols)

    # 创建画布
    plt.figure(figsize=(5 * cols, 5 * rows))
    # plt.suptitle(f"不健康牙齿类别随机采样可视化 (总类别数: {num_labels})", fontsize=16, y=0.98)

    # --- 3. 循环处理每个类别 ---
    for idx, label in enumerate(active_labels):
        # 随机选一个样本
        if label in label_dict.keys():
            sample_pair = random.choice(label_data[label])
            json_file = sample_pair['json']
            img_file = sample_pair['img']

            # 读取图片
            if not os.path.exists(img_file):
                print(f"警告: 图片不存在 {img_file}")
                continue
                
            # OpenCV 读取的是 BGR，需转为 RGB 以便 Matplotlib 显示
            img = cv2.imread(img_file)
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 读取 JSON 获取坐标
            with open(json_file, 'r', encoding='utf-8') as f:
                js_content = json.load(f)

            # --- 4. 绘制多边形 ---
            shapes = js_content.get('shapes', [])
            found_shape = False
            
            for shape in shapes:
                # 只绘制当前关注的这个 Label，避免图片太乱
                # 如果想绘制该图上所有的病灶，去掉 `if shape['label'] == label:` 判断即可
                if shape.get('label') == label:
                    points = shape.get('points', [])
                    if points:
                        # 坐标点转为 numpy int32 数组，OpenCV 绘图需要整数
                        pts = np.array(points, np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        
                        # 绘制轮廓 (图像, 坐标点, 是否闭合, 颜色(RGB), 线宽)
                        # 颜色: 红色 (255, 0, 0)
                        cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=3)
                        
                        found_shape = True

            # --- 5. 添加到子图 ---
            ax = plt.subplot(rows, cols, idx + 1)
            ax.imshow(img)
            p = Path(img_file)
            sample_title = f"{p.parent.parent.name}_{p.parent.name}{p.stem}"
            label_title = label_dict[label]
            ax.set_title(label_title, fontsize=12)
            # ax.set_title(f"Label: {label_title}\nSample: {sample_title}", fontsize=10)
            ax.axis('off')  # 关闭坐标轴显示

    # --- 6. 输出与保存 ---
    plt.tight_layout()
    output_vis_path = "caries_visualization.png"
    plt.savefig(output_vis_path, dpi=150)
    print(f"可视化组图已保存至: {os.path.abspath(output_vis_path)}")
    plt.show()

if __name__ == "__main__":
    # 请替换为你的数据集路径
    DATASET_DIR = r".datasets/intraoral/annosample/single_tooth"
    
    if os.path.exists(DATASET_DIR):
        visualize_unhealthy_samples(DATASET_DIR)
    else:
        print("路径不存在")



# %%
import os
import json
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
random.seed(42)

# --- 配置 ---
ROOT_DIR = ".datasets/intraoral/annosample/sextant"
PERIO_GRADES = ["G0", "G1", "G3", "G4"]
# 假设牙菌斑等级在 label 中体现，这里根据实际 label 自动提取
# 默认颜色：红色用于标注
ANNOTATION_COLOR = (255, 0, 0) 

def process_dental_data(root_path):
    # 数据存储：{grade: [path_dict, ...]}
    perio_samples = defaultdict(list)
    plaque_samples = defaultdict(list)
    
    # 统计计数
    perio_counts = defaultdict(int)
    plaque_counts = defaultdict(int)

    print("正在扫描数据集并分析标注信息...")

    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith(".json"):
                json_path = os.path.join(dirpath, filename)
                img_path = os.path.join(dirpath, filename.replace(".json", ".png"))
                
                if not os.path.exists(img_path):
                    continue

                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 1. 牙周炎分级统计 (Flags)
                flags = data.get('flags', {})
                for g in PERIO_GRADES:
                    if flags.get(g):
                        perio_samples[g].append(img_path)
                        perio_counts[g] += 1
                        break # 假设一张图只有一个等级

                # 2. 牙菌斑分级统计 (Shapes)
                shapes = data.get('shapes', [])
                if not shapes:
                    plaque_samples["None"].append(img_path)
                    plaque_counts["None"] += 1
                else:
                    # 获取该图片中包含的所有牙菌斑等级
                    current_img_labels = set()
                    for s in shapes:
                        label = s.get('label')
                        if label:
                            current_img_labels.add(label)
                    
                    for lbl in current_img_labels:
                        plaque_samples[lbl].append({'img': img_path, 'json': json_path})
                        plaque_counts[lbl] += 1

    return perio_samples, perio_counts, plaque_samples, plaque_counts

def draw_polygons(img, json_path, target_label=None):
    """在图片上绘制多边形标注"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for shape in data.get('shapes', []):
        label = shape.get('label')
        # 如果指定了 label，则只画该 label 的框；否则全画
        if target_label and label != target_label:
            continue
            
        pts = np.array(shape.get('points', []), np.int32)
        if pts.size > 0:
            cv2.polylines(img, [pts.reshape((-1, 1, 2))], True, ANNOTATION_COLOR, 3)
    return img

def visualize_grid(data_dict, type, num_samples=3):
    """生成组图：行为样本，列为等级"""
    grades = sorted(data_dict.keys())
    if not grades: return
    
    fig, axes = plt.subplots(num_samples, len(grades), figsize=(4 * len(grades), 4 * num_samples))
    # fig.suptitle(title, fontsize=20, y=1.02)

    for col, grade in enumerate(grades):
        available = data_dict[grade]
        # 随机抽取样本
        selected = random.sample(available, min(len(available), num_samples))
        
        for row in range(num_samples):
            ax = axes[row, col] if len(grades) > 1 else axes[row]
            
            if row < len(selected):
                sample = selected[row]
                # 处理数据格式差异
                img_p = sample['img'] if isinstance(sample, dict) else sample
                
                img = cv2.imread(img_p)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 如果是牙菌斑，且不是 "None" 等级，绘制标注
                if type == "Plaque" and grade != "None" and isinstance(sample, dict):
                    img = draw_polygons(img, sample['json'], target_label=grade)
                
                ax.imshow(img)
                # 提取展示用的 ID: SampleID_Dir_File
                p = Path(img_p)
                display_id = f"{p.parts[-3]}_{p.parts[-2]}_{p.stem}"
                ax.set_title(grade, fontsize=12)
                # ax.set_title(f"{grade}\n{display_id}", fontsize=10)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center')
            
            ax.axis('off')

    plt.tight_layout()
    output_vis_path = f"{type}_visualization.png"
    plt.savefig(output_vis_path, dpi=150)
    plt.show()

# --- 主程序 ---
if __name__ == "__main__":
    p_samples, p_counts, plq_samples, plq_counts = process_dental_data(ROOT_DIR)

    print("\n" + "="*30)
    print("牙周炎 (Periodontitis) 统计:")
    for g in PERIO_GRADES:
        print(f"  {g}: {p_counts[g]} 张")

    print("\n牙菌斑 (Plaque) 统计:")
    for lbl in sorted(plq_counts.keys()):
        print(f"  {lbl}: {plq_counts[lbl]} 张")
    print("="*30 + "\n")

    # 可视化
    visualize_grid(p_samples, "Gingivitis")
    visualize_grid(plq_samples, "Plaque")


# %%
import os
from collections import defaultdict

def analyze_dental_folders(base_paths):
    print(f"{'根目录':<20} | {'总图片数':<10} | {'样本(Sample)数':<12} | {'平均每样本张数'}")
    print("-" * 75)
    
    for root_path in base_paths:
        if not os.path.exists(root_path):
            print(f"{root_path:<20} | 路径不存在")
            continue
            
        png_count = 0
        samples = set()
        
        # 遍历根目录
        # 假设结构是 root/sample_id/view_id/xxx.png
        for entry in os.scandir(root_path):
            if entry.is_dir():
                sample_id = entry.name
                has_png_in_sample = False
                
                # 递归查找该样本下的所有 png
                for sub_root, _, files in os.walk(entry.path):
                    for file in files:
                        if file.lower().endswith('.png'):
                            png_count += 1
                            has_png_in_sample = True
                
                if has_png_in_sample:
                    samples.add(sample_id)
        
        sample_num = len(samples)
        avg = png_count / sample_num if sample_num > 0 else 0
        
        print(f"{root_path:<20} | {png_count:<12} | {sample_num:<14} | {avg:.2f}")

# 你的文件夹路径
target_dirs = [
    ".datasets/intraoral/annosample/single_tooth",
    ".datasets/intraoral/annosample/sextant"
]

if __name__ == "__main__":
    analyze_dental_folders(target_dirs)
# %%
