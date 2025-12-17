import SimpleITK
import ants
import os
import json
import logging
import multiprocessing
import nibabel as nib
import glob


# 根据给定的工作目录和参数字典，获取图像文件的名称并更新参数字典
def get_nii_file_name(sdict, workdir):
    # series_file_list = os.listdir(workdir)
    series_file_list = []
    for root, dirs, files in os.walk(workdir):
        for file in files:
            if file.endswith('.nii.gz'):
                series_file_list.append(file)

    tmp_list = series_file_list.copy()
    for series in sorted(sdict.keys()):
        if "filename" not in sdict[series].keys():
            # 遍历json的可能的名字序列
            for series_dict_name in sdict[series]["name"]:
                if "filename" in sdict[series].keys():
                    continue
                # 找对应的实际文件名
                for series_file_name in series_file_list:
                    # json名字对应上实际文件名
                    if series_dict_name in series_file_name:
                        sdict[series].update({"filename": os.path.join(workdir, series_file_name)})
                        break
        # 若没读到filename 报错
        assert "filename" in sdict[series].keys(), "warning! cannot find {} filename in {}".format(series, workdir)
    return sdict


# 从参数文件中获取系列信息并返回一个包含系列信息的字典
def get_series(workdir, params):
    assert os.path.isfile(params), "Param file does not exist at " + params
    with open(params, 'r') as f:
        json_str = f.read()
    sdict = json.loads(json_str)
    """2.22修改"""

    sdict.update({"info": {
        "filename": "None",
        "dcmdir": workdir,
        "id": workdir.split(os.sep)[-1],
    }})
    sdict = get_nii_file_name(sdict, workdir)

    # t1 redirect to resampled_t1
    # t1 dict的文件名改成重采样的t1，使用重采样后的作为模板
    # dir_temp = sdict["info"]["dcmdir"].split('/')
    # dir_temp[-2] = "CE-MRI-resample"
    # resample_name = '/'.join(dir_temp)
    # ### 对上resample里的正确的名字
    # for name in os.listdir(resample_name):
    #     if "t1ts" in name or "t1fs" in name:
    #         dir_temp.append(name)
    #         break
    # resample_name = '/'.join(dir_temp)
    # ###
    # sdict["T1"].update({"filename": resample_name})

    # assert os.path.isfile(resample_name), "Resampled T1 does not exist"
    return sdict


# 创建日志文件和日志记录器
def make_log(work_dir, repeat=False):
    if not os.path.isdir(work_dir):
        os.makedirs(work_dir)
    # make log file, append to existing
    idno = os.path.basename(work_dir)
    log_file = os.path.join(work_dir, idno + "_log.txt")
    if repeat:
        open(log_file, 'w').close()
    else:
        open(log_file, 'a').close()
    # make logger
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)  # should this be DEBUG?
    # set all existing handlers to null to prevent duplication
    logger.handlers = []
    # create file handler that logs debug and higher level messages
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatterch = logging.Formatter('%(message)s')
    formatterfh = logging.Formatter("[%(asctime)s]  [%(levelname)s]:     %(message)s", "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatterch)
    fh.setFormatter(formatterfh)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False
    logger.info("####################### STARTING NEW LOG #######################")


# 对图像进行配准，并更新系列字典中的相关信息
def reg_series(ser_dict, repeat=False):
    # logging
    logger = logging.getLogger("my_logger")
    logger.info("REGISTERING IMAGES:")
    # dcm_dir prep
    dcm_dir = ser_dict["info"]["dcmdir"]
    # sort serdict keys so that the atlas reg comes up first - this makes sure atlas registration is first
    sorted_keys = []
    for key in sorted(ser_dict.keys()):
        if key == "ASL":
            continue
        sorted_keys.append(key)

    if "ASL" in ser_dict.keys():
        sorted_keys.append("ASL")

    # if reg is false, or if there is no input file found, then just make the reg filename same as unreg filename
    for ser in sorted_keys:
        # first, if there is no filename, set to None
        if "filename" not in ser_dict[ser].keys():
            ser_dict[ser].update({"filename": "None"})
        if ser_dict[ser]["filename"] == "None" or "reg" not in ser_dict[ser].keys() or not ser_dict[ser]["reg"]:
            ser_dict[ser].update({"filename_reg": ser_dict[ser]["filename"]})
            ser_dict[ser].update({"transform": "None"})
            ser_dict[ser].update({"reg": False})
        # if reg True, then do the registration using translation, affine, nonlin, or just applying existing transform
    # handle registration
    for ser in sorted_keys:
        if ser_dict[ser]["reg"] and ser_dict[ser]["reg"] not in sorted_keys:
            # todo
            trans_method = ser_dict[ser]["reg"]
            if os.path.isfile(ser_dict[ser]["reg_target"]):
                template = ser_dict[ser]["reg_target"]
            else:
                # todo  ASL这里没有filename_reg这个键值，所以会报错
                template = ser_dict[ser_dict[ser]["reg_target"]]["filename_reg"]

            # handle surrogate moving image
            if "reg_moving" in ser_dict[ser]:
                movingr = ser_dict[ser]["reg_moving"]
                movinga = ser_dict[ser]["filename"]
            else:
                movingr = ser_dict[ser]["filename"]
                movinga = ser_dict[ser]["filename"]
            # handle registration options here
            if "reg_option" in ser_dict[ser].keys():
                option = ser_dict[ser]["reg_option"]
            else:
                option = None

            transforms = get_reg_transform(moving_nii=movingr,
                                           template_nii=template,
                                           work_dir=dcm_dir,
                                           type_of_transform=trans_method,
                                           option=option, )
            # todo
            transforms_file = transforms['fwdtransforms']
            # handle interp option
            if "interp" in ser_dict[ser].keys():
                interp = ser_dict[ser]["interp"]
            else:
                interp = 'linear'
            niiout = ants_apply(movinga, template, interp, transforms_file, dcm_dir)
            ser_dict[ser].update({"filename_reg": niiout})
            ser_dict[ser].update({"transform": transforms_file})
        # 如果是用别的序列的配准方法
        elif ser_dict[ser]["reg"] and ser_dict[ser]["reg"] in sorted_keys:
            transforms_file = ser_dict[ser_dict[ser]["reg"]]["transform"]
            template = ser_dict[ser_dict[ser]["reg"]]["filename_reg"]
            moving = ser_dict[ser]["filename"]
            # 插值
            if "interp" in ser_dict[ser].keys():
                interp = ser_dict[ser]["interp"]
            else:
                interp = 'linear'
            niiout = ants_apply(moving, template, interp, transforms_file, dcm_dir)
            ser_dict[ser].update({"filename_reg": niiout})
            ser_dict[ser].update({"transform": transforms_file})
    return ser_dict


# 获取图像之间的变换矩阵
def get_reg_transform(moving_nii, template_nii, work_dir, type_of_transform, option=None):
    logger = logging.getLogger("my_logger")
    # 给transform文件建文件夹
    transform_file_folder = os.path.join(work_dir, 'trans_file')
    if not os.path.exists(transform_file_folder):
        os.makedirs(transform_file_folder)
    # moving和模板
    moving_name = os.path.basename(moving_nii).split(".")[0]
    template_name = os.path.basename(template_nii).split(".")[0]
    # transform文件名
    outprefix = os.path.join(transform_file_folder, moving_name + "_2_" + template_name + "_")
    # initial_transform_option = option["reg_com"] if isinstance(option, dict) and "reg_com" in option.keys() else 1 #不知道什么参数
    moving_img = ants.image_read(moving_nii)
    template_img = ants.image_read(template_nii)

    # 第一阶段：BOLDRigid 配准
    trans_affine = ants.registration(
        fixed=template_img,
        moving=moving_img,
        type_of_transform="BOLDRigid",
        outprefix="step1_",
        write_composite_transform=False  # 仿射变换生成.mat文件
    )
    # 获取生成的仿射变换文件路径
    initial_transform_for_syn = trans_affine["fwdtransforms"][0]

    if type_of_transform == "Affine":
        # todo
        transformation = ants.registration(fixed=template_img,
                                           moving=moving_img,
                                           outprefix=outprefix,
                                           random_seed=2023,
                                           # 多尺度参数：3个层级，长度必须一致
                                           # aff_smoothing_sigmas=[6, 4, 1],
                                           type_of_transform=type_of_transform,
                                           write_composite_transform=False,
                                           # initial_transform=initial_transform_option,
                                           # todo
                                           # winsorize_lower_quantile=0.005,
                                           # winsorize_upper_quantile=0.995,
                                           # aff_iterations=[1000, 1000, 1000],
                                           # aff_shrink_factors=[[6, 4, 2], [6, 4, 2]],
                                           # grad_step=0.1,
                                           )
    elif type_of_transform == "SyN" or type_of_transform == "SyNBold":
        transformation = ants.registration(fixed=template_img,
                                           moving=moving_img,
                                           outprefix=outprefix,
                                           random_seed=2023,
                                           # aff_smoothing_sigmas=[6, 4, 1],
                                           type_of_transform=type_of_transform,
                                           # initial_transform="BOLDRigid",
                                           write_composite_transform=True,
                                           initial_transform=initial_transform_for_syn,  # 关键：传递文件路径
                                           # todo
                                           # winsorize_lower_quantile=0.005,
                                           # winsorize_upper_quantile=0.995,
                                           # aff_iterations=[1000, 1000, 1000],
                                           # reg_iterations=[50, 50],
                                           # aff_shrink_factors=[[6, 4, 2], [6, 4, 2]],
                                           # grad_step=0.1,
                                           )

    logger.info("- Registering image " + moving_nii + " to " + template_nii)
    return transformation


# 将变换应用于图像
def ants_apply(moving, fixed, interp, transform_list, work_dir):
    # logging
    logger = logging.getLogger("my_logger")
    # 获取registration保存的位置
    dir_temp = work_dir.split('/')
    # dir_temp[-2] = "CE-MRI-reg"
    reg_dir = '/'.join(dir_temp)
    # 不存在就建造
    if not os.path.exists(reg_dir):
        os.makedirs(reg_dir)
    # enforce list
    if not isinstance(moving, list):
        moving = [moving]
    if not isinstance(transform_list, list):
        transform_list = [transform_list]
    # create output list of same shape
    output_nii = moving
    # define extension
    ext = ".nii"
    # for loop for applying reg
    for ind, mvng in enumerate(moving, 0):
        # get ants img
        moving_img = ants.image_read(mvng)
        fixed_img = ants.image_read(fixed)
        # define output path
        output_nii[ind] = os.path.join(reg_dir, os.path.basename(mvng).split(ext)[0] + '_w.nii.gz')
        # do registration if not already done
        # todo
        warped_image = ants.apply_transforms(fixed=fixed_img, moving=moving_img, transformlist=transform_list,
                                             interpolator=interp)
        logger.info("- Creating warped image " + output_nii[ind])
        # 获取nii图像并保存
        warped_nii = ants.to_nibabel(warped_image)
        nib.save(warped_nii, output_nii[ind])
    # if only 1 label, don't return array
    if len(output_nii) == 1:
        output_nii = output_nii[0]
    return output_nii


# 将重采样的T1图像移动到配准到文件夹
def move_resampled_t1_to_regfolder(series_dict):
    t1_image_name = series_dict["T1"]["filename"]
    t1_image = SimpleITK.ReadImage(t1_image_name)
    reg_new_dir = os.path.dirname(series_dict["T2"]["filename_reg"])
    reg_filename = os.path.join(reg_new_dir, os.path.basename(t1_image_name).split(".nii")[0] + '_w.nii.gz')
    SimpleITK.WriteImage(t1_image, reg_filename)
    series_dict["T1"].update({"filename_reg": reg_filename})

    tmp_dir = series_dict["info"]["dcmdir"]
    tmp_list = tmp_dir.split('/')
    # tmp_list[-2] = "CE-MRI-reg"
    tmp_dir = "/".join(tmp_list)
    series_dict["info"].update({"dcmdir": tmp_dir})
    return series_dict


# 设置窗宽和窗位
def window_width_and_level(series_dict):
    print("set window and level")
    for ser in series_dict.keys():
        if ser == "B0":
            img = SimpleITK.ReadImage(series_dict[ser]["filename"])
            img_array = SimpleITK.GetArrayFromImage(img)
            img_array[img_array > 500] = 500
            img_new = SimpleITK.GetImageFromArray(img_array)
            img_new.CopyInformation(img)
            SimpleITK.WriteImage(img_new, series_dict[ser]["filename"])
        if ser == "DWI":
            img = SimpleITK.ReadImage(series_dict[ser]["filename"])
            img_array = SimpleITK.GetArrayFromImage(img)
            img_array[img_array > 40] = 40
            img_new = SimpleITK.GetImageFromArray(img_array)
            img_new.CopyInformation(img)
            SimpleITK.WriteImage(img_new, series_dict[ser]["filename"])
    return series_dict



