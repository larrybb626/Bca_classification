import pandas as pd
import os
import shutil
import re

# ================= 配置路径 =================
# 1. 刚刚生成的匹配好的 Excel 文件路径
excel_path = '/nas_3/LaiRuiBin/Dongguan_project/bca_classification/莞医训练集_Label已匹配.xlsx'

# 2. 源文件夹路径
src_dir_1 = '/nas_3/LaiRuiBin/Bca_MRI/T2WI_label'  # 主文件夹
src_dir_2 = '/nas_3/LaiRuiBin/Bca_MRI/T2WI_label(xkx)'  # 副文件夹 (xkx)

# 3. 输出的新文件夹路径
output_dir = '/nas_3/LaiRuiBin/Bca_MRI/T2WI_label_merge'


# ================= 核心逻辑函数 =================

def normalize_key(text):
    """
    生成纯字母 Key 用于匹配：
    '009 CHEN GUO CHENG' -> 'CHENGUOCHENG'
    'DINGRUNNAN_merge1.nii.gz' -> 'DINGRUNNAN'
    """
    if pd.isna(text):
        return ""
    text = str(text).upper()
    # 去除文件后缀
    text = text.replace('.NII.GZ', '').replace('.NII', '')
    # 去除 _merge 及其后续数字
    text = re.sub(r'_MERGE\d*', '', text)
    # 只保留字母 (去掉数字、空格、标点)
    clean_text = ''.join(filter(str.isalpha, text))
    return clean_text


def extract_variant_and_ext(filename):
    """
    从原文件名中提取：
    1. 变体后缀 (如 '_merge', '_merge1', 或空)
    2. 文件扩展名 (如 '.nii.gz')
    """
    # 1. 确定扩展名
    if filename.lower().endswith('.nii.gz'):
        ext = '.nii.gz'
        base = filename[:-7]
    elif filename.lower().endswith('.nii'):
        ext = '.nii'
        base = filename[:-4]
    else:
        ext = os.path.splitext(filename)[1]
        base = os.path.splitext(filename)[0]

    # 2. 提取 merge 部分
    # 查找是否有 _merge 或 _merge1, _merge2...
    match = re.search(r'(_merge\d*)', base, re.IGNORECASE)

    if match:
        variant = match.group(1).lower()  # 例如 '_merge1'
    else:
        variant = None  # 表示原名里没有 merge，需要补

    return variant, ext


# ================= 主程序 =================

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"创建输出目录: {output_dir}")

# 1. 读取 Excel 建立【Key -> 完整标准名】的映射
print("读取 Excel 建立标准名映射...")
df = pd.read_excel(excel_path)

# 自动找最长的那一列通常是 '编号 姓名'，例如 '病人名称'
col_name = '病人名称' if '病人名称' in df.columns else df.columns[0]
print(f"使用列 '{col_name}' 作为命名标准。")

name_map = {}  # {'CHENGUOCHENG': '009 CHEN GUO CHENG'}
for idx, row in df.iterrows():
    full_name = str(row[col_name]).strip()  # Excel里的原样：009 CHEN GUO CHENG
    key = normalize_key(full_name)  # 清洗后：CHENGUOCHENG
    if key:
        name_map[key] = full_name

print(f"映射建立完成，共有 {len(name_map)} 个目标人物。")

# 2. 扫描并处理文件
files_processed = 0
files_skipped = 0

# 定义源文件夹列表，顺序很重要：先主后副
source_folders = [src_dir_1, src_dir_2]
# 用于记录已经生成的文件名，防止副文件夹覆盖主文件夹的同名文件
generated_filenames = set()

print("开始处理文件...")

for src_dir in source_folders:
    if not os.path.exists(src_dir):
        continue

    print(f"正在扫描: {src_dir}")

    for fname in os.listdir(src_dir):
        if not (fname.endswith('.nii') or fname.endswith('.nii.gz')):
            continue

        # 1. 提取文件的 Key
        file_key = normalize_key(fname)

        # 2. 匹配 Excel 中的名单
        if file_key in name_map:
            # 获取 Excel 中的标准全名 (如 "009 CHEN GUO CHENG")
            standard_name_prefix = name_map[file_key]

            # 3. 分析原文件名的变体情况
            variant, ext = extract_variant_and_ext(fname)

            # 4. 构建新文件名
            # 逻辑：标准名 + (原有的merge后缀 或 补上_merge) + 扩展名
            if variant:
                # 如果原文件本来就有 _merge 或 _merge1，直接接上去
                # 例如: "009 CHEN GUO CHENG" + "_merge1" + ".nii.gz"
                new_filename = f"{standard_name_prefix}{variant}{ext}"
            else:
                # 如果原文件没有 merge (通常是 xkx 里的)，补上 _merge
                # 例如: "210 WANG YU YOU" + "_merge" + ".nii.gz"
                new_filename = f"{standard_name_prefix}_merge{ext}"

            # 5. 复制文件 (防重名覆盖)
            if new_filename not in generated_filenames:
                src_path = os.path.join(src_dir, fname)
                dst_path = os.path.join(output_dir, new_filename)

                try:
                    shutil.copy2(src_path, dst_path)
                    generated_filenames.add(new_filename)
                    files_processed += 1
                    if files_processed % 20 == 0:
                        print(f"已处理 {files_processed} 个文件...", end='\r')
                except Exception as e:
                    print(f"\n复制出错: {fname} -> {e}")
            else:
                # 如果名字冲突（通常是因为主文件夹已经有了同名文件，副文件夹的就跳过）
                pass
        else:
            files_skipped += 1

print("\n" + "=" * 30)
print("处理完成！")
print(f"总计生成文件: {files_processed}")
print(f"输出目录: {output_dir}")
print("=" * 30)