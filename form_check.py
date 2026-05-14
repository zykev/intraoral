# %%
# 文件格式检查与自动旋转
import os
from pathlib import Path
from PIL import Image  # 需要安装: pip install Pillow

def verify_and_fix_dataset(root_dir):
    root = Path(root_dir)
    required_names = {'F', 'U', 'D', 'L', 'R'}
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.PNG'}
    
    errors = []
    valid_count = 0
    rotate_count = 0

    # 1. 遍历时间文件夹
    time_dirs = [d for d in root.iterdir() if d.is_dir()]
    
    for time_dir in time_dirs:
        # 2. 遍历样本 ID 文件夹
        sample_dirs = [d for d in time_dir.iterdir() if d.is_dir()]
        
        for sample_path in sample_dirs:
            # 过滤掉隐藏文件（如 .DS_Store），只保留指定后缀的图片文件
            files_in_sample = [f for f in sample_path.iterdir() if f.is_file() and f.suffix in valid_extensions]
            filenames_without_ext = {f.stem for f in files_in_sample}
            
            # 校验基本结构：文件名必须是 F, U, D, L, R 且共 5 张
            is_structure_valid = (filenames_without_ext == required_names) and (len(files_in_sample) == 5)
            
            if is_structure_valid:
                pass
            else:
                actual_files = [f.name for f in files_in_sample]
                errors.append({
                    "path": str(sample_path),
                    "found_files": actual_files,
                    "reason": "文件名不符或数量不是5 (注意需为大写且无空格)"
                })

    # 输出结果报告
    print(f"--- 处理报告 ---")
    print(f"结构合格样本数: {valid_count}")
    print(f"结构不合格/处理失败样本数: {len(errors)}")
    print("-" * 20)

    if errors:
        print("以下文件夹存在问题:")
        for err in errors:
            print(f"\n[路径]: {err['path']}")
            if 'found_files' in err: print(f" [实际文件]: {err['found_files']}")
            print(f" [原因]: {err['reason']}")

# 执行
# 注意：你的路径中有个点 '.datasets'，请确认路径是否正确
target_path = '.datasets/intraoral/garris_20251216' 
verify_and_fix_dataset(target_path)
# %%
