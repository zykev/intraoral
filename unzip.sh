# Step 1: Batch unzip
echo "[Step 1] 开始解压 zip 文件..."
ZIP_DIR=".datasets/intraoral/amy_new"
if [ -d "$ZIP_DIR" ]; then
    cd "$ZIP_DIR"
    for zip_file in *.zip; do
        if [ -f "$zip_file" ]; then
            echo "正在解压: $zip_file"
            unzip -q "$zip_file"
            if [ $? -eq 0 ]; then
                echo "✓ 解压成功: $zip_file"
            else
                echo "✗ 解压失败: $zip_file"
                exit 1
            fi
        fi
    done
    cd - > /dev/null
    echo "所有 zip 文件解压完成。"
else
    echo "警告: 目录 $ZIP_DIR 不存在，跳过解压。"
fi