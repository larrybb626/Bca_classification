import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import h5py
import os
global nii_name
import pandas as pd
import math
import pywt
def nii_loader(nii_path):
    print('#Loading ', nii_path, '...')
    data = nib.load(nii_path)

    print("--Loading size:", data.shape)

    return data

def my_resize(o_data, transform_size = None, transform_rate = None):
    print('#Resizing...')
    data = o_data
    print("--Original size:", data.shape)
    if transform_size:
        o_width, o_height, o_queue = data.shape
        width, height, queue = transform_size
        data = zoom(data, (width/o_width, height/o_height, queue/o_queue))
    elif transform_rate:
        data = zoom(data, transform_rate)
        # data = zoom(data, (transform_rate, transform_rate, transform_rate))

    print("--Transofmed size:", data.shape)

    return data

def centre_window_cropping(o_data, reshapesize = None):
    print('#Centre window cropping...')
    data = o_data
    or_size = data.shape
    target_size = (reshapesize[0], reshapesize[1], or_size[2])

    # pad if or_size is smaller than target_size
    if (target_size[0] > or_size[0]) | (target_size[1] > or_size[1]) :
        if target_size[0] > or_size[0]:
            pad_size = int((target_size[0] - or_size[0]) / 2)
            data = np.pad(data, ((pad_size, pad_size),(0, 0), (0, 0)))
        if target_size[1] > or_size[1]:
            pad_size = int((target_size[1] - or_size[1]) / 2)
            data = np.pad(data, ((0, 0), (pad_size, pad_size), (0, 0)))

    #  centre_window_cropping
    cur_size = data.shape
    centre_x = float(cur_size[0] / 2)
    centre_y = float(cur_size[1] / 2)
    dx = float(target_size[0] / 2)
    dy = float(target_size[1] / 2)
    data = data[int(centre_x - dx + 1):int(centre_x + dx), int(centre_y - dy + 1): int(centre_y + dy), :]

    data = my_resize(data, transform_size=target_size)

    return data

def wavelet_preprocessing(o_data, wavelet='db1', level=1):
    print('#Wavelet preprocessing...')
    data = o_data
    processed_data = []
    for i in range(data.shape[2]):
        # 对每一层进行二维小波变换
        coeffs = pywt.wavedec2(data[:, :, i], wavelet, level=level)
        # 重构图像
        reconstructed = pywt.waverec2(coeffs, wavelet)
        processed_data.append(reconstructed)
    processed_data = np.array(processed_data).transpose(1, 2, 0)
    print("--Processed size:", processed_data.shape)
    return processed_data

def getListIndex(arr, value) :
    dim1_list = dim2_list = dim3_list = []
    if (arr.ndim == 3):
        index = np.argwhere(arr == value)
        dim1_list = index[:, 0].tolist()
        dim2_list = index[:, 1].tolist()
        dim3_list = index[:, 2].tolist()

    else :
        raise ValueError('The ndim of array must be 3!!')

    return dim1_list, dim2_list, dim3_list

def ROI_cutting(o_data, o_roi, expend_voxel = 500):
    print('#ROI cutting...')
    data = o_data
    roi = o_roi

    [I1, I2, I3] = getListIndex(roi, 1)
    d1_min = min(I1)
    d1_max = max(I1)
    d2_min = min(I2)
    d2_max = max(I2)
    d3_min = min(I3)
    d3_max = max(I3)
    print(d3_min, d3_max)

    if expend_voxel > 0:
        d1_min -= expend_voxel
        d1_max += expend_voxel
        d2_min -= expend_voxel
        d2_max += expend_voxel
        # d3_min -= expend_voxel
        # d3_max += expend_voxel

        d1_min = d1_min if d1_min>0 else 0
        d1_max = d1_max if d1_max<data.shape[0]-1 else data.shape[0]-1
        d2_min = d2_min if d2_min>0 else 0
        d2_max = d2_max if d2_max<data.shape[1]-1 else data.shape[1]-1

    data = data[d1_min-1:d1_max+3,d2_min-1:d2_max+3,d3_min:d3_max+1]
    print(data.shape)
    roi = roi[d1_min:d1_max+1,d2_min:d2_max+1,d3_min:d3_max+1]
    # data = data[d1_min:d1_max+1,d2_min:d2_max+1,:]
    # roi = roi[d1_min:d1_max+1,d2_min:d2_max+1,:]

    print("--Cutting size:", data.shape)
    return data, roi

def make_h5_data(o_data, o_roi=None, label=None, h5_save_path=None,count = None,count1 = None):
    print('#Make h5 data...')
    data = o_data
    roi = o_roi
    if (h5_save_path):
        for i, divided_data in enumerate(data):
            if not os.path.exists(os.path.join(h5_save_path, str(count))):
                os.makedirs(os.path.join(h5_save_path, str(count)))
            save_file_name = os.path.join(h5_save_path, str(count), str(count)  +'_'+str(i+1)+ '.h5')
            with h5py.File(save_file_name, 'a') as f:
                print("--h5 file path:", save_file_name,'    -label:', label, '    -size:', divided_data.shape)
                f['Data'] = divided_data
                f['Label'] = [label]

def make_h5_data_new(o_data, o_roi=None, label=None, h5_save_path=None, count=None, count1=None):
    print('#Make h5 data...')
    data = o_data
    roi = o_roi
    if (h5_save_path):
        if not os.path.exists(os.path.join(h5_save_path, str(count))):
            os.makedirs(os.path.join(h5_save_path, str(count)))
    save_file_name = os.path.join(h5_save_path, str(count),str(count) + '_' + str(count) + '.h5')
    with h5py.File(save_file_name, 'a') as f:
        print("--h5 file path:", save_file_name, '    -label:', label, '    -size:', data.shape)
        f['Data'] = data
        f['Label'] = [label]
        f['ROI'] = roi
        print('kokokokokokok')

def linear_normalizing(o_data):
    print('#Linear_normalizing...')
    data = o_data
    minn = np.min(data)
    maxx = np.max(data)
    data = ((data - minn)+math.exp(-20)) / ((maxx - minn)+math.exp(-20))

    return data

def block_dividing(o_data, deep = None, step = None):
    print('#Block dividing...')
    data = o_data
    data_group = []
    o_data_deep = data.shape[2]

    if o_data_deep <= deep:
        tmp_data = np.zeros((data.shape[0], data.shape[1], deep))
        tmp_data[:, :, 0:o_data_deep] = data
        blocks = 1
        tmp_data = tmp_data
        data_group.append(tmp_data)

    else:
        blocks = (o_data_deep - deep) // step + 2
        if (o_data_deep - deep) % step == 0:
            blocks -= 1
        for i in range(blocks-1):
            tmp_data = data[:, :, (0 + i * step): (deep + i * step)]
            data_group.append(tmp_data)
        # tmp_data = np.zeros((data.shape[0],data.shape[1],deep))
        # tmp_data[:,:,0:(o_data_deep-(deep+i*step))] = data[:,:,(deep+i*step):o_data_deep]
        tmp_data = data[:, :, o_data_deep -deep:o_data_deep]
        data_group.append(tmp_data)

    print("--Block size:", tmp_data.shape,
          " Divided number:(%d)"%(blocks))

    return data_group, blocks

if __name__ == "__main__":

    """Setting the path"""
    roi_root = r'Y:\newnas_1\huyilan202124\LvJieGeng-BCa\SUNYATSEN-H\ZhongShanErYuan\NII\DWI NII'
    data_root = r'Y:\newnas_1\huyilan202124\LvJieGeng-BCa\SUNYATSEN-H\ZhongShanErYuan\DWI_nii_img'
    save_root = r"Y:\newnas_1\huyilan202124\LvJieGeng-BCa\SUNYATSEN-H\ZhongShanErYuan\h5"
    h5_save_path = os.path.join(save_root, 'dwi_h5')
    if not os.path.exists(h5_save_path):
        os.makedirs(h5_save_path)
    reshapesize = (128, 128)
    deep = 8
    step = 8
    count = 0


    """Reading the label information"""
    label_list =pd.read_excel(r"Y:\newnas_1\huyilan202124\LvJieGeng-BCa\SUNYATSEN-H\ZhongShanErYuan\Label.xlsx")
    label_list = np.array(label_list)
    # print("label_list:{}".format(label_list))

    """Readint the image path and list"""
    data_all_list = os.listdir(data_root)
    data_all_list.sort()
    # print(data_all_list)
    temp = []
    all_information = []

    """Reading the roi path and list"""
    roi_list = os.listdir(roi_root)
    roi_list.sort()
    # print(roi_list)

    """Processing the data"""
    for filename in roi_list:
        count += 1
        case_name = filename.split("_")[0]
        save_name = filename.split(".")[0]
        print("case_name:{}".format(case_name))
        roi_data_path = os.path.join(roi_root, filename)
        img_data_path = os.path.join(data_root, case_name + ".nii.gz")

        data_metrix = nii_loader(img_data_path)
        roi_metrix = nii_loader(roi_data_path)

        label = label_list[label_list[:, 0] == filename, 1]
        print(label)
        roi_arr = np.array(roi_metrix.dataobj, dtype='float32')
        roi_arr[roi_arr < 0.5] = 0
        roi_arr[roi_arr >= 0.5] = 1
        img_arr = np.array(data_metrix.dataobj, dtype='float32')

        # 使用布尔索引将大于 1000 的数置为 1000
        img_arr[img_arr > 1500] = 1500

        # 添加小波预处理步骤
        img_arr = wavelet_preprocessing(img_arr)

        img_arr, roi_arr = ROI_cutting(img_arr, roi_arr, expend_voxel=0)
        img_arr = centre_window_cropping(img_arr, reshapesize=reshapesize)
        img_arr = linear_normalizing(img_arr)
        global_max = img_arr.max()
        global_min = img_arr.min()
        img_arr = block_dividing(img_arr, deep=deep, step=step)
        make_h5_data(img_arr[0], label=label[0], h5_save_path=h5_save_path, count=count)

