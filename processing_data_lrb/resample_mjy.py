"""antspy needs python version 3.8+, so I chose to use local interpreter"""
"""
医学影像配准变换模型（Transform Models）配置说明：
1. 线性配准阶段：
   - Translation：仅处理X/Y/Z轴平移，用于初步对齐。
   - Rigid（刚性）：包含平移与旋转（6自由度），适用于同一病例不同模态（如T1/T2）或不同时间点的配准。
   - Affine（仿射）：包含平移、旋转、错切与非等向缩放（12自由度），用于修正受检者体位差异或作为非线性配准的初始化。
2. 非线性配准阶段：
   - SyN（对称归一化）：ANTs核心算法，提供微分同胚的可变形配准，能够处理不同个体间器官形状的解剖学精细差异。
   - BSplineSyN：引入B样条平滑约束的SyN，适用于噪声较大或需要更平滑位移场的场景。
通常采用“Rigid -> Affine -> SyN”的级联策略，实现从全局粗对齐到局部精细形变的逐级收敛。
"""
import os
import glob
import SimpleITK as sitk
from antspy_registration import get_series

def itk_resample(moving, target, t_type="image", resamplemethod=sitk.sitkLinear):
    """
    重采样函数
    :param moving: 需要变换的图像 (SimpleITK Image)
    :param target: 目标模板图像 (SimpleITK Image, 提供Spacing, Origin, Direction, Size)
    :param t_type: 类型 "image" 或 "mask"
    :param resamplemethod: 插值方法
    :return: 重采样后的图像
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(target)  # 直接使用目标图像作为参考，自动获取Size, Origin, Spacing, Direction

    # 设置变换 (Identity 表示不进行空间变形，只进行重采样对齐网格)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))

    if t_type == "mask":
        # Mask 必须用最近邻插值，保证结果只有 0 和 1
        resampler.SetOutputPixelType(sitk.sitkUInt8)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    elif t_type == "image":
        resampler.SetOutputPixelType(sitk.sitkFloat32)
        resampler.SetInterpolator(resamplemethod)

    itk_img_resampled = resampler.Execute(moving)
    return itk_img_resampled


if __name__ == '__main__':
    # 路径配置
    dcm_data_dir = r'/nas_3/LaiRuiBin/Bca_MRI/images_tr'
    save_folder = r'/nas_3/LaiRuiBin/Bca_MRI/resample_1'
    param_file = r'/nas_3/LaiRuiBin/Dongguan_project/bca_classification/processing_data_lrb/series_param.json'

    dcms = [item for item in glob.glob(dcm_data_dir + "/*") if os.path.isdir(item)]

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i, dcm in enumerate(dcms, 1):
        patient_name = os.path.basename(dcm)
        print(f"Processing ({i}/{len(dcms)}): {patient_name}")

        # 创建保存目录
        patient_save_dir = os.path.join(save_folder, patient_name)
        if not os.path.exists(patient_save_dir):
            os.makedirs(patient_save_dir)

        # 1. 获取序列信息 (主要为了找到 T2 作为 Target)
        try:
            ser_dict = get_series(dcm, param_file)
        except Exception as e:
            print(f"  Error reading series: {e}")
            continue

        # 2. 锁定 Target (T2)
        if "T2" not in ser_dict or ser_dict["T2"]["filename"] == "None":
            print(f"  Warning: No T2 found for {patient_name}, skipping.")
            continue

        target_file_path = ser_dict["T2"]["filename"]
        target_image = sitk.ReadImage(target_file_path)

        # 记录 T2 的文件名，防止重复处理或者用于排除
        target_filename = os.path.basename(target_file_path)

        # 3. 【核心修改】直接遍历文件夹内的所有文件，而不是依赖 ser_dict
        # 这样可以确保 merge1, merge2...merge6 都会被扫到
        all_files = os.listdir(dcm)

        for fname in all_files:
            # 只处理 .nii 或 .nii.gz
            if not (fname.endswith('.nii') or fname.endswith('.nii.gz')):
                continue

            full_path = os.path.join(dcm, fname)

            # 跳过 T2 本身 (通常不需要重采样自己，或者如果需要也可以放开)
            if fname == target_filename:
                # 如果你想把 T2 也复制/重采样过去，可以去掉 continue，并将 is_mask 设为 False
                # 这里为了节省时间，假设 T2 本身就是基准，不需要处理
                # 但为了保存目录的完整性，通常也需要保存一份 T2
                pass

            # 判断文件类型
            is_mask = False
            # 只要文件名包含这些关键字，就认定为 Mask
            if "label" in fname.lower() or "mask" in fname.lower() or "merge" in fname.lower():
                is_mask = True

            # 如果不是 Mask，也不是 T2，那可能是 DWI, ADC 等
            # 我们可以检查它是否在 ser_dict 的其他键中，或者直接默认处理所有 nii
            # 这里采取策略：处理所有 Mask + 处理 ser_dict 中记录的其他图像

            # 读取图像
            try:
                moving_image = sitk.ReadImage(full_path)
            except:
                print(f"  Failed to read: {fname}")
                continue

            # 设置重采样类型
            t_type = "mask" if is_mask else "image"

            # 如果是 image 类型，建议只处理那些确实是 MRI 序列的文件 (避免处理临时文件)
            # 简单的过滤：如果是 Mask，一定处理；如果是 Image，检查是否在 ser_dict 的 filenames 列表里
            # (这样可以避免处理一些无关的 nii 文件)
            all_ser_filenames = [os.path.basename(v['filename']) for k,v in ser_dict.items() if v['filename'] != 'None']
            if not is_mask and fname not in all_ser_filenames:
                 # 如果这个文件既不是 Mask，也不是 ser_dict 识别出的序列，暂时跳过
                 # (当然，如果你确定文件夹里全是必须要的文件，可以去掉这个判断)
                 continue

            # 执行重采样
            resampled_img = itk_resample(moving_image, target_image, t_type=t_type)

            # 保存
            save_nii = os.path.join(patient_save_dir, fname)
            sitk.WriteImage(resampled_img, save_nii)

            if is_mask:
                print(f"  Saved Mask: {fname}")
            else:
                print(f"  Saved Img : {fname}")

    print("All processing done.")
    """修改后的代码"""
    # for i, dcm in enumerate(dcms, 1):
    #     ser_dict = get_series(dcm, param_file)
    #     dir_temp = dcm.split('/')
    #     resample_dir = '/'.join(dir_temp)
    #     target = os.path.join(resample_dir, os.path.basename(ser_dict["T2"]["filename"]))
    #     target_image = sitk.ReadImage(target)
    #     if not os.path.exists(resample_dir):
    #         os.makedirs(resample_dir)
    #     for ser in ser_dict.keys():
    #         if ser_dict[ser]["filename"] != "None":
    #             moving_file = ser_dict[ser]["filename"]
    #             moving_image = sitk.ReadImage(moving_file)
    #             if "mask" in ser_dict[ser]['filename'] and "_mask" not in ser_dict[ser]['filename']:
    #                 resampled_img = itk_resample(moving_image, target_image, t_type="mask")
    #             else:
    #                 resampled_img = itk_resample(moving_image, target_image)
    #             # 使用病人文件夹名称构建保存路径
    #             save_folder_patient = os.path.join(save_folder, os.path.basename(dcm))
    #             if not os.path.exists(save_folder_patient):    #检查这个路径是否存在
    #                 os.makedirs(save_folder_patient)           #如果不存在则递归地创建目录，会创建名为sace_folder_patient的文件夹
    #             save_nii = os.path.join(save_folder_patient, os.path.basename(moving_file))
    #             sitk.WriteImage(resampled_img, save_nii)
    #     print('{} is done'.format(ser_dict['info']['id']), '{}/{}'.format(i, len(dcms)))
