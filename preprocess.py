import os
import argparse
from pathlib import Path
from itertools import islice
import torch
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from tqdm import tqdm

# 注册 HEIF/HEIC 文件格式的打开器
register_heif_opener()

from transformers import Sam3Processor, Sam3Model

# ------------------------------------------------------------------
# 0. 参数解析器
# ------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 牙科图像预处理与裁剪脚本")
    parser.add_argument(
        "--base_dir", 
        type=str, 
        default=".datasets/intraoral/zeyu",
        help="数据集的根目录 (包含日期子文件夹)"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=".checkpoints/sam3",
        help="SAM3 模型存储路径"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="模型推理的批次大小"
    )
    parser.add_argument(
        "--padding", 
        type=int, 
        default=100,
        help="裁剪时向外扩展的像素值"
    )
    return parser.parse_args()

args = parse_args()

# 将路径转换为绝对路径
BASE_DIR = Path(args.base_dir).resolve()
LOCAL_MODEL_PATH = Path(args.model_path).resolve()
ERROR_LOG_PATH = BASE_DIR / "preprocess_errors.txt"

# 其他配置
BATCH_SIZE = args.batch_size
PADDING_PIXELS = args.padding
TEXT_PROMPT = "mouth" 

# --- 1. 环境初始化 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Sam3Model.from_pretrained(LOCAL_MODEL_PATH).to(device)
processor = Sam3Processor.from_pretrained(LOCAL_MODEL_PATH)
model.eval()

def log_error(msg):
    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# ------------------------------------------------------------------
# 2. 预处理阶段：文件格式转换 (原位覆盖)
# ------------------------------------------------------------------

def convert_raw_images(root_path, target_extensions=('.jpg', '.jpeg', '.heic', '.webp', '.png')):
    # 统一转为小写后缀
    target_extensions = [ext.lower() for ext in target_extensions]
    
    # 获取所有需要检查的图片文件（包含已经为 .png 的文件）
    all_files = [
        p for p in root_path.rglob("*") 
        if p.is_file() and p.suffix.lower() in target_extensions
    ]
    
    if not all_files:
        print("未发现任何图片文件。")
        return

    print(f"正在处理图片（格式转换 + 尺寸修正），共 {len(all_files)} 个文件...")
    
    for old_path in tqdm(all_files, desc="处理进度", unit="file"):
        try:
            is_png = old_path.suffix.lower() == '.png'
            new_path = old_path.with_suffix('.png')
            
            with Image.open(old_path) as img:
                # 自动处理 EXIF 旋转信息（防止手机拍照导致的预览与像素坐标不符）
                img = ImageOps.exif_transpose(img)
                
                w, h = img.size
                needs_rotate = (h >= w)
                
                # 如果是 PNG 且不需要旋转，直接跳过
                if is_png and not needs_rotate:
                    continue
                
                # 执行旋转
                if needs_rotate:
                    img = img.rotate(90, expand=True)
                
                # 保存结果
                img.save(new_path, format='PNG')
            
            # 如果原文件不是 PNG，或者原文件是需要旋转的 PNG，删除旧文件
            if not is_png:
                old_path.unlink()
                
        except Exception as e:
            log_error(f"[PROCESS_ERROR] 路径: {old_path} | 原因: {e}")

# ------------------------------------------------------------------
# 3. SAM 核心处理函数
# ------------------------------------------------------------------
def process_batch(batch_files_info):
    images = []
    valid_batch = []
    
    for full_path in batch_files_info:
        try:
            img = Image.open(full_path).convert("RGB")
            images.append(img)
            valid_batch.append(full_path)
        except Exception as e:
            log_error(f"[OPEN_ERROR] 路径: {full_path} | 原因: {e}")

    if not images: return

    try:
        inputs = processor(images=images, text=[TEXT_PROMPT]*len(images), return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs, threshold=0.5, mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )

        for i, (image, result, full_path) in enumerate(zip(images, results, valid_batch)):
            if 'boxes' in result and len(result['boxes']) > 0:
                box = result['boxes'][0].cpu().numpy()
                xmin, ymin, xmax, ymax = box
                w, h = image.size

                crop_box = (
                    max(0, int(xmin - PADDING_PIXELS)),
                    max(0, int(ymin - PADDING_PIXELS)),
                    min(w, int(xmax + PADDING_PIXELS)),
                    min(h, int(ymax + PADDING_PIXELS))
                )
                
                cropped_image = image.crop(crop_box)
                
                # 提取三层结构路径: .../日期/样本ID/图片
                sample_id_folder = full_path.parent.name
                date_folder = full_path.parent.parent.name
                
                # 构造目标路径: .../日期_process/process/样本ID/图片
                save_dir = BASE_DIR / f"{date_folder}_process" / "process" / sample_id_folder
                
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / full_path.name
                cropped_image.save(save_path)
    except Exception as e:
        log_error(f"[SAM_ERROR] 批处理异常 | 原因: {e}")

# ------------------------------------------------------------------
# 4. 主程序执行
# ------------------------------------------------------------------
if __name__ == "__main__":
    if ERROR_LOG_PATH.exists():
        ERROR_LOG_PATH.unlink()

    print(f"当前处理根目录: {BASE_DIR}")
    
    # 第一步：原位转换
    convert_raw_images(BASE_DIR)

    # 第二步：获取所有转换后的图片
    all_pngs = [p for p in BASE_DIR.rglob("*.png") if "_process" not in str(p)]
    
    num_images = len(all_pngs)
    if num_images == 0:
        print("没有找到待处理的 PNG 图片。")
    else:
        print(f"找到 {num_images} 张图片，开始分批处理...")
        it = iter(all_pngs)
        num_batches = (num_images + BATCH_SIZE - 1) // BATCH_SIZE
        
        pbar = tqdm(total=num_batches, desc="SAM 裁剪进度", unit="batch")
        while True:
            batch = list(islice(it, BATCH_SIZE))
            if not batch: break
            process_batch(batch)
            pbar.update(1)
        pbar.close()

    if ERROR_LOG_PATH.exists():
        print(f"\n任务完成，但存在异常，请查看: {ERROR_LOG_PATH}")
    else:
        print("\n--- 所有任务顺利完成 ---")