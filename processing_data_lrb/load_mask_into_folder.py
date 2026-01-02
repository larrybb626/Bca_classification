import os
import shutil

# ================= 配置路径 =================
# 1. 存放 Mask 的源文件夹 (你刚才整理好的那个文件夹)
# 注意：请确保这个路径是你存放带编号文件名(如 002 ZHANG..._merge.nii.gz)的真实路径
src_dir = '/nas_3/LaiRuiBin/Bca_MRI/T2WI_label_merge'

# 2. 目标根目录 (下面有 002 ZHANG DE FEN 等子文件夹)
dst_root = '/nas_3/LaiRuiBin/Bca_MRI/images_tr'

# ================= 主程序 =================

if not os.path.exists(src_dir):
    print(f"错误: 源文件夹不存在 -> {src_dir}")
    exit()

if not os.path.exists(dst_root):
    print(f"错误: 目标根目录不存在 -> {dst_root}")
    exit()

print("开始分发 Mask 文件...")
success_count = 0
skip_count = 0
error_count = 0

# 遍历源文件夹中的所有文件
for fname in os.listdir(src_dir):
    if not (fname.endswith('.nii') or fname.endswith('.nii.gz')):
        continue

    # --- 核心逻辑：提取文件夹名称 ---
    # 文件名示例: "002 ZHANG DE FEN_merge.nii.gz"
    # 目标: 提取 "002 ZHANG DE FEN"

    # 使用 _merge 进行分割，取第一部分
    if '_merge' in fname:
        folder_name = fname.split('_merge')[0]
    else:
        # 如果文件名里没有 _merge，可能是不符合规范的文件，尝试去掉扩展名
        # 但根据你的描述，应该都是带 _merge 的
        print(f"警告: 文件名不包含 '_merge'，跳过 -> {fname}")
        skip_count += 1
        continue

    # --- 构造路径并复制 ---
    target_folder_path = os.path.join(dst_root, folder_name)

    # 检查目标子文件夹是否存在
    if os.path.isdir(target_folder_path):
        src_file = os.path.join(src_dir, fname)
        dst_file = os.path.join(target_folder_path, fname)

        try:
            shutil.copy2(src_file, dst_file)
            print(f"成功: {fname} -> {folder_name}/")
            success_count += 1
        except Exception as e:
            print(f"复制失败: {fname} -> {e}")
            error_count += 1
    else:
        # 如果目标文件夹不存在 (可能是名字不匹配，或者images_tr里没这个人)
        print(f"未找到目标文件夹: {folder_name} (对应文件: {fname})")
        skip_count += 1

print("\n" + "=" * 30)
print("处理完成！")
print(f"成功分发: {success_count} 个文件")
print(f"跳过/未找到目录: {skip_count} 个文件")
if error_count > 0:
    print(f"复制错误: {error_count} 个文件")
print("=" * 30)