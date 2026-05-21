# %%
# https://github.com/thangngoc89/SegmentAnyTooth
import cv2
import os

def extract_frames(video_path, output_dir, frame_rate=1):
    """
    将视频文件切分成图片帧。
    
    Args:
        video_path (str): 输入 MOV 视频文件的完整路径。
        output_dir (str): 保存输出图片帧的目录。
        frame_rate (int/float): 帧提取频率。
                                 如果 frame_rate >= 1，则表示每秒提取多少帧。
                                 如果 frame_rate < 1，则表示每隔多少秒提取一帧（例如 0.5 表示每 2 秒提取 1 帧）。
    """
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"错误: 视频文件未找到在 {video_path}")
        return

    # 1. 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")
    
    # 2. 初始化视频捕获对象
    vidcap = cv2.VideoCapture(video_path)
    
    if not vidcap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return

    # 3. 获取视频基本信息
    fps = vidcap.get(cv2.CAP_PROP_FPS)  # 视频的原始帧率
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频原始帧率 (FPS): {fps}")
    print(f"视频总帧数: {frame_count}")
    
    # 计算实际提取帧的间隔
    if frame_rate <= 0:
        print("帧率参数无效，使用默认值 1 帧/秒。")
        frame_rate = 1
        
    # 计算我们需要跳过多少帧来达到目标提取频率
    if frame_rate >= 1:
        # 目标是每秒提取 frame_rate 帧
        frame_skip = int(fps / frame_rate)
    else:
        # 目标是每 1/frame_rate 秒提取 1 帧
        frame_skip = int(fps * (1 / frame_rate))
        
    frame_skip = max(1, frame_skip) # 确保至少跳过 1 帧
    
    print(f"将每隔 {frame_skip} 帧提取一帧。")
    
    # 4. 循环提取帧
    success, image = vidcap.read()
    frame_num = 0
    extracted_count = 0

    while success:
        # 检查是否应该提取当前帧
        if frame_num % frame_skip == 0:
            # 构造输出文件名 (使用五位数序列号)
            frame_filename = os.path.join(output_dir, f"frame_{extracted_count:05d}.jpg")
            
            # 保存帧为图片文件
            cv2.imwrite(frame_filename, image)
            extracted_count += 1
        
        # 读取下一帧
        success, image = vidcap.read()
        frame_num += 1

    vidcap.release()
    print("-" * 30)
    print(f"完成！总共从 {video_path} 提取了 {extracted_count} 帧图片到 {output_dir}")

# --- 配置您的视频和输出路径 ---
# 替换为您的 MOV 文件名或完整路径
VIDEO_INPUT_PATH = "C:\\Users\\Chen Zeyu\\Desktop\\L.MOV"
# 输出文件夹名称
OUTPUT_DIRECTORY = "frames_output"
# 目标提取频率：设置为 1，表示每秒提取 1 帧。
# 设置为 30，表示提取视频中的所有帧 (假设视频是 30 FPS)。
EXTRACTION_RATE = 1 
# ---------------------------------

if __name__ == "__main__":
    # 确保脚本在正确的目录下运行，可以简化路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_full_path = os.path.join(base_dir, OUTPUT_DIRECTORY)
    
    # 如果您的视频文件在脚本所在目录，可以简化 VIDEO_INPUT_PATH
    # 假设您的 MOV 文件名是 test.mov
    # VIDEO_INPUT_PATH = os.path.join(base_dir, "test.mov")
    
    extract_frames(VIDEO_INPUT_PATH, output_full_path, EXTRACTION_RATE)


# %%

import os
from PIL import Image
from pillow_heif import register_heif_opener

# 注册 HEIF/HEIC 文件格式的打开器
register_heif_opener()

# 配置要处理的文件夹和文件类型
ROOT_DIR = '/home/peiliu/Documents/intraoral_process/dataset'          # 原始图片所在的根文件夹
OUTPUT_DIR = '/home/peiliu/Documents/intraoral_process/process'   # 转换后的图片将保存到这个新文件夹
TARGET_EXTENSIONS = ('.jpg', '.jpeg', '.heic') # 需要转换的后缀
# -----------------

def convert_and_save_to_new_folder(root_path, output_path):
    """遍历指定路径，将 JPG, HEIC 文件转换为 PNG，并保存到新的目录结构中"""
    
    if not os.path.exists(root_path):
        print(f"错误: 原始根目录 '{root_path}' 不存在。")
        return
        
    print(f"--- 开始处理目录: {root_path} ---")
    
    # 遍历根目录下的所有文件和文件夹
    for subdir, dirs, files in os.walk(root_path):
        
        # 构造相对于 ROOT_DIR 的子目录路径 (例如: R 或 R/subfolder)
        # os.path.relpath('id/R', 'id') -> 'R'
        relative_path = os.path.relpath(subdir, root_path)
        
        # 构造新的输出子目录的完整路径 (例如: process/R)
        new_subdir = os.path.join(output_path, relative_path)
        
        # 确保新的输出子目录存在
        if not os.path.exists(new_subdir):
            os.makedirs(new_subdir)
        
        # 遍历当前目录下的所有文件
        for file in files:
            file_lower = file.lower()
            
            # 检查文件后缀是否需要转换
            if file_lower.endswith(TARGET_EXTENSIONS):
                
                old_path = os.path.join(subdir, file)
                
                # 构造新的文件路径（后缀改为 .png，路径指向 OUTPUT_DIR）
                base_name = os.path.splitext(file)[0]
                new_file = f"{base_name}.png"
                new_path = os.path.join(new_subdir, new_file)
                
                try:
                    # 尝试打开并转换图像
                    img = Image.open(old_path)
                    img.save(new_path)
                    
                    print(f"[成功] 转换: {old_path} -> {new_path}")
                    
                except Exception as e:
                    print(f"[失败] 转换 {old_path}: {e}")
            
    print("--- 所有文件转换完成 ---")

if __name__ == "__main__":
    convert_and_save_to_new_folder(ROOT_DIR, OUTPUT_DIR)


# %%
# segment by sam3
import torch
#################################### For Image ####################################
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
bpe_path = '.checkpoints/bpe_simple_vocab_16e6.txt.gz'
checkpoint_path = '.checkpoints/sam3/sam3.pt'

# Load the model
model = build_sam3_image_model(checkpoint_path=checkpoint_path, bpe_path=bpe_path)
model.to(device)
model.eval()

processor = Sam3Processor(model, confidence_threshold=0.5)
# Load an image
image = Image.open(".datasets/intraoral/test/s1_process/process/testid1/U.png")
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    inference_state = processor.set_image(image)
    # Prompt the model with text
    output = processor.set_text_prompt(state=inference_state, prompt="teeth")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

# %%
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results

plot_results(image, output)


# %%

from transformers import Sam3Processor, Sam3Model
import torch
from PIL import Image
import requests

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = '.checkpoints/sam3'

model = Sam3Model.from_pretrained(checkpoint_path).to(device)
processor = Sam3Processor.from_pretrained(checkpoint_path)

# Load image
image = Image.open("process/s20251003/D.png").convert("RGB")

# Segment using text prompt
inputs = processor(images=image, text="mouth", return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

# Post-process results
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)[0]

print(f"Found {len(results['masks'])} objects")
# Results contain:
# - masks: Binary masks resized to original image size
# - boxes: Bounding boxes in absolute pixel coordinates (xyxy format)
# - scores: Confidence scores


# %%
from transformers import Sam3Processor, Sam3Model
import torch
from PIL import Image
import os

# 1. 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. 加载模型 (假设你的路径是正确的)
checkpoint_path = '.checkpoints/sam3'
try:
    model = Sam3Model.from_pretrained(checkpoint_path).to(device)
    processor = Sam3Processor.from_pretrained(checkpoint_path)
except OSError:
    print(f"Error: checkpoint path '{checkpoint_path}' not found. Please check the path.")
    # 为了演示代码逻辑，这里如果加载失败可能会报错停止

# 3. 加载图片
# 假设图片路径是存在的，如果不存在请修改路径
image_paths = [
    "process/s20251003/D.png",
    "process/s20251003/L.png",
    "process/s20251003/F.png",
    "process/s20251003/R.png",
    "process/s20251003/U.png",
]

images = []
valid_indices = [] # 记录成功加载的图片索引
for idx, path in enumerate(image_paths):
    if os.path.exists(path):
        images.append(Image.open(path).convert("RGB"))
        valid_indices.append(idx)
    else:
        print(f"Warning: Image path not found: {path}")

if not images:
    raise FileNotFoundError("No images were loaded. Please check the paths.")

# 4. 准备提示词 (确保提示词数量与图片一致)
text_prompts = ["tooth"] * len(images) # 自动匹配图片数量

# 5. 数据预处理
inputs = processor(images=images, text=text_prompts, return_tensors="pt").to(device)

# 6. 模型推理
print("Running inference...")
with torch.no_grad():
    outputs = model(**inputs)

# 7. 后处理
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)

for i, res in enumerate(results):
    print(f"Image {i+1} ({image_paths[valid_indices[i]]}): Found {len(res['boxes'])} objects")

# %%
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import math

# 配置
padding_pixels = 100
num_images = len(images)

# 动态计算子图布局 (例如：如果有5张图，排成1行)
cols = num_images
rows = 1
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 6))

# 如果只有一张图，axes不是列表，需要转换以便循环
if num_images == 1:
    axes = [axes]
elif num_images > 1:
    axes = axes.flatten()

# 遍历所有图片结果
for i, (image, result, prompt) in enumerate(zip(images, results, text_prompts)):
    ax = axes[i]
    
    # 显示原图
    ax.imshow(image)
    img_width, img_height = image.size
    
    # 获取 Boxes
    if 'boxes' in result:
        boxes = result['boxes'].cpu().numpy()
    else:
        boxes = []

    # 绘制 Boxes
    for box in boxes:
        # box 格式: [xmin, ymin, xmax, ymax]
        xmin_orig, ymin_orig, xmax_orig, ymax_orig = box
        
        # 应用 Padding (并防止越界)
        xmin_padded = max(0, xmin_orig - padding_pixels)
        ymin_padded = max(0, ymin_orig - padding_pixels)
        xmax_padded = min(img_width, xmax_orig + padding_pixels)
        ymax_padded = min(img_height, ymax_orig + padding_pixels)
        
        width_padded = xmax_padded - xmin_padded
        height_padded = ymax_padded - ymin_padded
        
        # 创建矩形
        rect = patches.Rectangle(
            (xmin_padded, ymin_padded),
            width_padded,
            height_padded,
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)
        
    ax.set_title(f"Image {i+1}: {len(boxes)} Objects\nPrompt: '{prompt}'")
    ax.axis('off')

plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

num_images = len(images)
cols = num_images
rows = 1
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 6))

if num_images == 1:
    axes = [axes]
elif num_images > 1:
    axes = axes.flatten()

# 遍历所有图片结果
for i, (image, result, prompt) in enumerate(zip(images, results, text_prompts)):
    ax = axes[i]
    
    # 显示原图
    ax.imshow(image)
    
    # 获取 Masks
    if 'masks' in result:
        # masks shape: [num_masks, height, width]
        masks = result['masks'].cpu().numpy()
    else:
        masks = []

    # 如果有检测到对象
    if len(masks) > 0:
        # 生成随机颜色
        colors = np.random.random((len(masks), 3))
        
        # 创建一个空白的 RGBA 图层用于叠加
        # 获取图像尺寸 (H, W) - 注意 PIL size 是 (W, H)
        H, W = masks[0].shape 
        overlay = np.zeros((H, W, 4), dtype=np.float32)

        for j, mask in enumerate(masks):
            # 将 mask 转为 float (0.0 或 1.0)
            mask_bool = mask.astype(bool)
            
            # 叠加颜色
            # 这里使用简单的方法：只要 mask 为 true，就叠加颜色
            # 注意：如果 mask 有重叠，后绘制的会覆盖前面的。
            # 更好的可视化是将所有 mask 合并，这里为了代码简洁逐个叠加
            
            color = colors[j]
            # 设置颜色通道
            overlay[mask_bool, 0] = color[0] # R
            overlay[mask_bool, 1] = color[1] # G
            overlay[mask_bool, 2] = color[2] # B
            # 设置透明度通道 (Alpha)
            overlay[mask_bool, 3] = 0.5      # 透明度 0.5
            
        ax.imshow(overlay)
        
    ax.set_title(f"Image {i+1}: {len(masks)} Masks\nPrompt: '{prompt}'")
    ax.axis('off')

plt.tight_layout()
plt.show()

# %%
import tarfile
from pathlib import Path

PERSON_DIRS = ["amy_new", "chenghao_new", "haozhou", "yinghao"]
ROOT_DIR = Path(".datasets/intraoral")

def archive_process_folders(root_dir: Path):
    root_dir = root_dir.resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"{root_dir} 不存在")

    for person in PERSON_DIRS:
        person_dir = root_dir / person
        if not person_dir.exists():
            print(f"[跳过] 目录不存在: {person_dir}")
            continue

        process_dirs = sorted(
            [p for p in person_dir.iterdir() if p.is_dir() and p.name.endswith("_process")]
        )
        if not process_dirs:
            print(f"[跳过] 未找到 _process 目录: {person_dir}")
            continue

        archive_path = root_dir / f"{person}_processes.tar.gz"
        print(f"打包 {person_dir} -> {archive_path}")

        with tarfile.open(archive_path, "w:gz") as tar:
            for d in process_dirs:
                tar.add(d, arcname=d.name)

        print(f"[完成] {archive_path} 大小: {archive_path.stat().st_size} bytes")

if __name__ == "__main__":
    archive_process_folders(ROOT_DIR)



# %%
from PIL import Image, ImageOps
import os

old_path = ".datasets/intraoral_anno/orth_test/orth_test/23069/R.JPG"
new_path = "tmp.png"

# 1. 补齐逻辑：通过后缀安全判断原图是否已经是 PNG 格式
is_png = old_path.lower().endswith('.png')

with Image.open(old_path) as img:
    # 自动处理 EXIF 旋转信息（防止手机拍照导致的预览与像素坐标不符）
    img = ImageOps.exif_transpose(img)
    
    w, h = img.size
    needs_rotate = (h >= w)
    
    # 如果是 PNG 且不需要旋转，直接跳过
    if is_png and not needs_rotate:
        # 注意：continue 只能在 for 或 while 循环内部使用。
        # 如果你这段代码本身就在循环里，请保留 continue；如果是单张图片测试，请改成 pass。
        pass 
    
    # 执行旋转（逆时针旋转 90 度并扩展画布）
    if needs_rotate:
        img = img.rotate(90, expand=True)
    
    # 统一保存为无损的 PNG 格式
    img.save(new_path, format='PNG')
# %%
