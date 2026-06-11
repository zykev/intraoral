import os
import shutil
from pathlib import Path

def zip_process_folders(base_path="."):
    """
    将指定人名文件夹下以 'process' 结尾的子文件夹分别进行压缩。
    
    :param base_path: 包含 zeyu, amy 等文件夹的根目录，默认为当前目录
    """
    # 指定的目标人名文件夹
    target_users = ["zeyu", "amy", "chenghao", "garris"]
    base_dir = Path(base_path)

    for user in target_users:
        user_dir = base_dir / user
        
        # 检查人名文件夹是否存在
        if not user_dir.exists() or not user_dir.is_dir():
            print(f"提示: 找不到文件夹 {user_dir}，已跳过。")
            continue
            
        print(f"正在处理 {user} 的文件夹...")
        
        # 计数器，记录处理了多少个 process 文件夹
        zip_count = 0
        
        # 遍历人名文件夹下的所有子项目
        for item in user_dir.iterdir():
            # 判断是否为文件夹，且名字以 'process' 结尾（忽略大小写，如需严格匹配可去掉 .lower()）
            if item.is_dir() and item.name.lower().endswith("process"):
                # 定义压缩包的输出路径（和原文件夹在同一目录下，名字相同）
                # shutil.make_archive 会自动加上 .zip 后缀
                output_zip_path = user_dir / item.name
                
                print(f"  正在压缩: {item.name} -> {item.name}.zip")
                
                # 执行压缩，format='zip' 表示压缩为 zip 格式
                shutil.make_archive(
                    base_name=str(output_zip_path),
                    format='zip',
                    root_dir=str(item)
                )
                zip_count += 1
                
        if zip_count == 0:
            print(f"  {user} 文件夹下没有找到以 'process' 结尾的子文件夹。")
        else:
            print(f"  {user} 处理完毕，共压缩了 {zip_count} 个文件夹。\n")

if __name__ == "__main__":
    # 如果你的脚本和 zeyu, amy 等文件夹在同一个目录下，直接运行即可
    # 如果在其他地方，传入对应的路径，例如：zip_process_folders("/path/to/folders")
    zip_process_folders('/home/zychen/Documents/intraoral_code/intraoral/.datasets/intraoral')