import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import h5py
import matplotlib.pyplot as plt
from my_process_funtion import check_image
from imgaug import augmenters as iaa
import imgaug as ia

class ImageFolder(data.Dataset):
	def __init__(self, root,image_size=224, mode='train', augmentation_prob=0.4):
		"""Initializes image paths ..and preprocessing module."""
		self.root = root
		
		# GT : Ground Truth
		self.GT_paths = root[:-1]+'_GT/'
		self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
		self.image_size = image_size
		self.mode = mode
		self.RotationDegree = [0,90,180,270]
		self.augmentation_prob = augmentation_prob
		print("image count in {} path :{}".format(self.mode,len(self.image_paths)))

	def __getitem__(self, index):
		"""Reads an image from a file and preprocesses it and returns."""
		image_path = self.image_paths[index]
		filename = image_path.split('_')[-1][:-len(".jpg")]
		GT_path = self.GT_paths + 'ISIC_' + filename + '_segmentation.png'

		image = Image.open(image_path)
		GT = Image.open(GT_path)

		aspect_ratio = image.size[1]/image.size[0]

		Transform = []

		ResizeRange = random.randint(300,320)
		Transform.append(T.Resize((int(ResizeRange*aspect_ratio),ResizeRange)))
		p_transform = random.random()

		if (self.mode == 'train') and p_transform <= self.augmentation_prob:
			RotationDegree = random.randint(0,3)
			RotationDegree = self.RotationDegree[RotationDegree]
			if (RotationDegree == 90) or (RotationDegree == 270):
				aspect_ratio = 1/aspect_ratio

			Transform.append(T.RandomRotation((RotationDegree,RotationDegree)))
						
			RotationRange = random.randint(-30,30)
			Transform.append(T.RandomRotation((RotationRange,RotationRange)))
			CropRange = random.randint(250,270)
			Transform.append(T.CenterCrop((int(CropRange*aspect_ratio),CropRange)))
			Transform = T.Compose(Transform)
			
			image = Transform(image)
			GT = Transform(GT)

			ShiftRange_left = random.randint(0,20)
			ShiftRange_upper = random.randint(0,20)
			ShiftRange_right = image.size[0] - random.randint(0,20)
			ShiftRange_lower = image.size[1] - random.randint(0,20)
			image = image.crop(box=(ShiftRange_left,ShiftRange_upper,ShiftRange_right,ShiftRange_lower))
			GT = GT.crop(box=(ShiftRange_left,ShiftRange_upper,ShiftRange_right,ShiftRange_lower))

			if random.random() < 0.5:
				image = F.hflip(image)
				GT = F.hflip(GT)

			if random.random() < 0.5:
				image = F.vflip(image)
				GT = F.vflip(GT)

			Transform = T.ColorJitter(brightness=0.2,contrast=0.2,hue=0.02)

			image = Transform(image)

			Transform =[]


		Transform.append(T.Resize((int(256*aspect_ratio)-int(256*aspect_ratio)%16,256)))
		Transform.append(T.ToTensor())
		Transform = T.Compose(Transform)
		
		image = Transform(image)
		GT = Transform(GT)

		Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		image = Norm_(image)

		return image, GT

	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.image_paths)


def get_loader(image_root,image_list, image_size, batch_size, num_workers=2, mode='train', augmentation_prob=0.4):
	"""Builds and returns Dataloader."""
	
	dataset = mydata_Folder(root=image_root, imagelist=image_list, image_size=image_size, mode=mode,augmentation_prob=augmentation_prob)
	# dataset = mydata_Folder_withLinchuang(root=image_root , imagelist = image_list, image_size =image_size, mode=mode,augmentation_prob=augmentation_prob)
	data_loader = data.DataLoader(dataset=dataset,
								  shuffle = True,
								  batch_size=batch_size,
								  num_workers=num_workers)
	return data_loader


################################################
class mydata_Folder(data.Dataset):
	def __init__(self, root, imagelist, image_size=224, mode='train', augmentation_prob=0.4):
		"""Initializes image paths and preprocessing module."""
		self.root = root
		self.imagelist = imagelist

		self.patients_paths = list(map(lambda x: os.path.join(root, x), imagelist))
		self.image_paths = []
		for patients_paths in self.patients_paths:
			for image in os.listdir(patients_paths):
				self.image_paths.append(os.path.join(patients_paths, image))

		self.image_size = image_size
		self.mode = mode
		self.augmentation_prob = augmentation_prob
		# self.seq = iaa.Sequential([
		# 	iaa.Fliplr(0.5),
		# 	iaa.Crop(px=(0,40)),
		# 	iaa.CropAndPad(percent=(0.9,1.0),keep_size = 1),
		# 	iaa.GaussianBlur(sigma=(0,1.5)),
		# 	iaa.ContrastNormalization((0.9, 1.1)),
		# 	iaa.Affine(rotate=(0, 80))
		# ])
		sometimes = lambda aug: iaa.Sometimes(0.9, aug)  # 设定随机函数,50%几率扩增,or
		self.seq = iaa.Sequential(
			[
				iaa.Fliplr(0.5),  # 50%图像进行水平翻转
				iaa.Flipud(0.5),  # 50%图像做垂直翻转
				sometimes(iaa.Crop(percent=(0, 0.15))),  # 对随机的一部分图像做crop操作 crop的幅度为0到10%
				sometimes(iaa.Affine(  # 对一部分图像做仿射变换
					scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # 图像缩放为80%到120%之间
					translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # 平移±20%之间
					rotate=(-45, 45),  # 旋转±45度之间
					shear=(-16, 16),  # 剪切变换±16度，（矩形变平行四边形）
					order=[0, 1],  # 使用最邻近差值或者双线性差值
					#cval=(0, 255),
					mode=ia.ALL,  # 边缘填充
				)),
				# 使用下面的0个到5个之间的方法去增强图像
				iaa.SomeOf((0, 5),[
					iaa.Sharpen(alpha=(0, 1.0), lightness=(0.8, 1.2)),# 锐化处理
					sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),# 扭曲图像的局部区域
					iaa.contrast.LinearContrast((0.8, 1.2), per_channel=0.5),# 改变对比度
					iaa.OneOf([
						iaa.GaussianBlur((0, 1.5)),
						iaa.AverageBlur(k=(2, 5)),  # 核大小2~7之间，k=((5, 7), (1, 3))时，核高度5~7，宽度1~3
						# iaa.MedianBlur(k=(3, 11)),
					]),# 用高斯模糊，均值模糊，中值模糊中的一种增强
					iaa.AdditiveGaussianNoise(
						loc=0, scale=(0.0, 0.05), per_channel=0.5
					),# 加入高斯噪声
					iaa.AdditiveLaplaceNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5),

					# 边缘检测，将检测到的赋值0或者255然后叠在原图上(不确定)
					# sometimes(iaa.OneOf([
					#     iaa.EdgeDetect(alpha=(0, 0.7)),
					#     iaa.DirectedEdgeDetect(
					#         alpha=(0, 0.7), direction=(0.0, 1.0)
					#     ),
					# ])),

					# 浮雕效果(很奇怪的操作,不确定能不能用)
					# iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

					iaa.OneOf([
						iaa.Dropout((0.01, 0.1), per_channel=0.5),
						iaa.CoarseDropout(
							(0.03, 0.15), size_percent=(0.02, 0.05),
							per_channel=0.2
						),
					]),# 将1%到10%的像素设置为黑色或者将3%到15%的像素用原图大小2%到5%的黑色方块覆盖
				],

				random_order=True  # 随机的顺序把这些操作用在图像上
				)
			],
			random_order=True  # 随机的顺序把这些操作用在图像上
		)

		print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

	def __getitem__(self, index):
		"""Reads an image from a file and preprocesses it and returns."""
		image_path = self.image_paths[index]

		H5_file = h5py.File(image_path, 'r')
		image = H5_file['Data'][()]
		label = H5_file['Label'][()]
		#label = H5_file['Label']
		H5_file.close()

		image = np.array(image, dtype='float32')
		label = np.array(label, dtype='float32')
		image_o = image
		label_o = label

		p_transform = random.random()
		if (self.mode == 'train') and p_transform <= self.augmentation_prob:
			# print('增强前输入', image.shape)
			image = self.seq.augment_image(image)
			# print('增强后输出', image.shape)

			# image_t = (image - image.min()) / (image.max() - image.min())
			# check_image(image_o, step = 1,data_name = image_path,show_slices = False)
			# check_image(image_t, step = 1,data_name = image_path,show_slices = False)
		if image.max() != image.min():
			image = (image - image.min()) / (image.max() - image.min() )
		else:
			image = image
		image = image[:, :, ::-1].copy()
		image = np.transpose(image, (2, 0, 1))
		image = np.expand_dims(image, axis=0)
		label = np.expand_dims(label, axis=0)

		# image = torch.from_numpy(image)
		# label = torch.from_numpy(label)

		# print(self.mode,image.shape,label)

		return image_path, image, label

	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.image_paths)
