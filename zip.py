import os
import zipfile

# 1. 定义 chenghao 文件夹的路径（如果脚本就在 chenghao 目录下运行，可以直接用 '.'）
target_dir = "/disk1/work/zychen/Datasets/intraoral/chenghao"

# 检查路径是否存在
if not os.path.exists(target_dir):
    print(f"错误：找不到路径 {target_dir}，请检查路径是否正确。")
    exit()

# 2. 遍历文件夹中的所有文件
for file_name in os.listdir(target_dir):
    # 只处理以 .zip 结尾的文件
    if file_name.lower().endswith('.zip'):
        zip_path = os.path.join(target_dir, file_name)
        
        # 获取不带后缀的文件夹名称（例如 "20251013"）
        folder_name = file_name[:-4] 
        output_dir = os.path.join(target_dir, folder_name)
        
        # 3. 创建同名文件夹（如果不存在的话）
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"正在解压: {file_name} -> {folder_name}/")
        
        # 4. 执行解压操作
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
        except Exception as e:
            print(f"解压 {file_name} 失败: {e}")

print("所有 ZIP 文件解压完成！")