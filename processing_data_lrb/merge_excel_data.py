import pandas as pd
import re


def normalize_name(text):
    """
    标准化名字函数：
    1. 转大写
    2. 去除 .nii.gz 等后缀
    3. 去除 _merge 等标记
    4. 去除所有非字母字符（空格、数字、下划线）
    例如: "002 ZHANG DE FEN" -> "ZHANGDEFEN"
    例如: "CHENGUOCHENG_merge.nii.gz" -> "CHENGUOCHENG"
    """
    if pd.isna(text):
        return ""
    text = str(text).upper()
    # 去除常见的文件后缀
    text = text.replace('.NII.GZ', '').replace('.XLSX', '')
    # 去除 _merge 及其后面的数字
    text = re.sub(r'_MERGE\d*', '', text)
    # 只保留英文字母
    clean_text = ''.join(filter(str.isalpha, text))
    return clean_text


# ================= 1. 读取并构建 Label 字典 =================

label_map = {}  # 结构: {'ZHANGDEFEN': 0, 'LIUWEITANG': 1, ...}

# --- 读取源文件 1: all_train_data_1222_h5.xlsx ---
print("正在读取源文件 1: all_train_data_1222_h5.xlsx ...")
try:
    # 假设第一列是 Label，第二列是 文件名
    # 根据你的Snippet，这个文件似乎没有标准表头，或者表头是 '1','2'
    df_src1 = pd.read_excel('all_train_data_1222_h5.xlsx', header=None)

    # 遍历查找包含 Label (0/1) 和 文件名 (.nii.gz) 的列
    # 简单策略：遍历行，找到像是数据的内容
    for idx, row in df_src1.iterrows():
        # 尝试将第一列转为 Label，第二列转为名字
        try:
            val_label = row[0]
            val_name = row[1]

            # 检查 label 是否有效 (0 或 1)
            if str(val_label).strip() in ['0', '1', '0.0', '1.0']:
                clean_name = normalize_name(val_name)
                if clean_name:
                    label_map[clean_name] = int(float(val_label))
        except:
            continue
    print(f"源文件 1 处理完毕，当前收集到 {len(label_map)} 个匹配信息。")

except Exception as e:
    print(f"源文件 1 读取失败: {e}")

# --- 读取源文件 2: 12.07完成莞医训练集独有人名（外送）.xlsx ---
print("正在读取源文件 2: 12.07完成莞医训练集独有人名（外送）.xlsx ...")
try:
    df_src2 = pd.read_excel('12.07完成莞医训练集独有人名（外送）.xlsx', sheet_name='独有人名')
    # 根据你的描述和截图
    # 拼音列通常是第二个 '姓名' (Pandas会自动重命名为 姓名.1)
    # Label列是 '有无浸入肌层无0 有1'

    pinyin_col = '姓名.1' if '姓名.1' in df_src2.columns else '姓名'  # 自动适配
    label_col = '有无浸入肌层无0 有1'

    if label_col not in df_src2.columns:
        print(f"警告：在文件2中未找到 '{label_col}' 列，尝试模糊匹配...")
        # 尝试找包含 '肌层' 的列
        for col in df_src2.columns:
            if '肌层' in str(col):
                label_col = col
                break

    for idx, row in df_src2.iterrows():
        p_name = row.get(pinyin_col)
        p_label = row.get(label_col)

        if pd.notna(p_name) and pd.notna(p_label):
            # 清洗名字
            clean_name = normalize_name(p_name)
            # 提取 Label (处理可能存在的文本，如 "0 (无)")
            try:
                # 简单提取数字
                label_str = str(p_label).strip()
                if label_str in ['0', '1', '0.0', '1.0']:
                    label_map[clean_name] = int(float(label_str))
            except:
                pass

    print(f"源文件 2 处理完毕，当前总计收集到 {len(label_map)} 个匹配信息。")

except Exception as e:
    print(f"源文件 2 读取失败: {e}")

# ================= 2. 处理目标文件并匹配 =================

print("-" * 30)
print("正在处理目标文件: 莞医训练集.xlsx ...")

try:
    target_path = '莞医训练集.xlsx'
    df_target = pd.read_excel(target_path)

    # 假设目标名字在 '病人名称' 列 (根据你的Snippet)
    name_col = '病人名称'
    if name_col not in df_target.columns:
        # 如果找不到，取第一列
        name_col = df_target.columns[0]
        print(f"提示：未找到 '病人名称' 列，默认使用第一列 '{name_col}' 进行匹配")

    # 新增 Label 列
    matched_labels = []

    unmatched_list = []  # 用于记录未匹配的名单

    for idx, row in df_target.iterrows():
        raw_name = row[name_col]
        clean_name = normalize_name(raw_name)

        if clean_name in label_map:
            matched_labels.append(label_map[clean_name])
        else:
            matched_labels.append(None)  # 未找到填空
            if pd.notna(raw_name):  # 记录未匹配的原始名字
                unmatched_list.append(raw_name)

    # 赋值回 DataFrame
    df_target['Label'] = matched_labels

    # 保存结果
    output_file = '../莞医训练集_Label已匹配.xlsx'
    df_target.to_excel(output_file, index=False)

    print(f"处理完成！文件已保存为: {output_file}")

    # ================= 3. 输出匹配情况 =================
    total_rows = len(df_target)
    matched_count = total_rows - len(unmatched_list)

    print("-" * 30)
    print(f"【匹配统计报告】")
    print(f"总人数: {total_rows}")
    print(f"成功匹配: {matched_count}")
    print(f"未匹配: {len(unmatched_list)}")
    print("-" * 30)

    if unmatched_list:
        print("以下人员未在源文件中找到 Label (请检查拼写差异):")
        for name in unmatched_list:
            print(f" - {name}")
    else:
        print("完美！所有人均已匹配。")

except Exception as e:
    print(f"处理目标文件时出错: {e}")