#!/bin/bash

# 检查是否提供了命令行参数
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_dir> <target_dir>"
    exit 1
fi

# 获取命令行参数
source_dir="$1"
target_dir="$2"

# 创建目标文件夹(如果不存在)
mkdir -p "$target_dir"

# 遍历源文件夹及其子文件夹
for dir in "$source_dir"/*; do
    if [ -d "$dir" ]; then # 如果是子文件夹
        dir_name=$(basename "$dir")
        for file in "$dir"/*.csv; do
            if [ -f "$file" ]; then # 如果是CSV文件
                new_file_name="${dir_name}_$(basename "$file")"
                cp "$file" "$target_dir/$new_file_name"
                echo "Copied $file to $target_dir/$new_file_name"
            fi
        done
    fi
done