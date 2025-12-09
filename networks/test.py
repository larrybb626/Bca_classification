import h5py
import numpy
import matplotlib.pyplot as plt

def read_h5_get_image_label(h5_path):
    H5_file = h5py.File(h5_path, 'r')
    image = H5_file['Data'][()]
    label = H5_file['Label'][()]
    return image,label


ex_h5_data_path = r'/nfs-data-new/LJG/BCa_DATA/expanded_DWI_49_h5/h5_data/43/43_1.h5'

image, label = read_h5_get_image_label(ex_h5_data_path)
print(image.shape)
slice = image[:,:,4]
plt.imshow(slice,cmap='gray')
plt.savefig('./h5_expand_20.png')