# %%
# 文件格式检查与自动旋转
import os
from pathlib import Path
from PIL import Image  # 需要安装: pip install Pillow

def consolidate_duplicate_dirs(root_dir):
    """如果发现重复目录层级（如 20251216/20251216），将内层文件移到外层"""
    root = Path(root_dir)
    time_dirs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    for time_dir in time_dirs:
        # 检查是否只有一个同名的子目录
        subdirs = [d for d in time_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        if len(subdirs) == 1 and subdirs[0].name == time_dir.name:
            inner_dir = subdirs[0]
            print(f"检测到重复目录: {time_dir.name}/{inner_dir.name}")
            
            # 将内层的所有文件和目录移到外层
            for item in inner_dir.iterdir():
                target = time_dir / item.name
                if target.exists():
                    print(f"  警告: 目标已存在 {target.name}，跳过")
                else:
                    item.rename(target)
                    print(f"  ✓ 移动: {item.name}")
            
            # 删除空的内层目录
            inner_dir.rmdir()
            print(f"  ✓ 删除空目录: {inner_dir.name}\n")


def verify_and_fix_dataset(root_dir):
    root = Path(root_dir)
    required_names = {'F', 'U', 'D', 'L', 'R'}
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.PNG', '.heic', '.HEIC'}
    
    errors = []
    valid_count = 0
    rotate_count = 0
    renamed_count = 0

    # 1. 遍历时间文件夹
    time_dirs = [d for d in root.iterdir() if d.is_dir()]
    
    for time_dir in time_dirs:
        # 2. 遍历样本 ID 文件夹
        sample_dirs = [d for d in time_dir.iterdir() if d.is_dir()]
        
        for sample_path in sample_dirs:
            # 先清理文件名中的空格
            files_in_sample = [f for f in sample_path.iterdir() if f.is_file() and f.suffix in valid_extensions]
            
            for file in files_in_sample:
                if ' ' in file.name:
                    new_name = file.name.replace(' ', '')
                    new_path = file.parent / new_name
                    file.rename(new_path)
                    print(f"✓ 重命名: {file.name} → {new_name}")
                    renamed_count += 1
            
            # 重新获取清理后的文件列表
            files_in_sample = [f for f in sample_path.iterdir() if f.is_file() and f.suffix in valid_extensions]
            filenames_without_ext = {f.stem for f in files_in_sample}
            
            # 校验基本结构
            is_structure_valid = (filenames_without_ext == required_names) and (len(files_in_sample) == 5)
            
            if is_structure_valid:
                valid_count += 1
            else:
                actual_files = [f.name for f in files_in_sample]
                errors.append({
                    "path": str(sample_path),
                    "found_files": actual_files,
                    "reason": "文件名不符或数量不是5 (注意需为大写且无空格)"
                })

    # 输出结果报告
    print(f"--- 处理报告 ---")
    print(f"重命名文件数: {renamed_count}")
    print(f"结构合格样本数: {valid_count}")
    print(f"结构不合格样本数: {len(errors)}")
    print("-" * 20)

    if errors:
        print("以下文件夹存在问题:")
        for err in errors:
            print(f"\n[路径]: {err['path']}")
            if 'found_files' in err: print(f" [实际文件]: {err['found_files']}")
            print(f" [原因]: {err['reason']}")

# 执行
target_path = '.datasets/intraoral/amy_new'  # 替换为你的数据集路径
# consolidate_duplicate_dirs(target_path)
verify_and_fix_dataset(target_path)
# %%
