from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import scipy.io as scio
import torch
import random
import numpy as np
#import cv2


def load_img(img_path):

	img = Image.open(img_path).convert('RGB')
	img = img.resize((224,224),Image.ANTIALIAS)
	return img

class SignDataset(Dataset):
	def __init__(self,mat_file,img_folder,window_size,transform =None):
		self.data = scio.loadmat(mat_file)
		self.img_folder = img_folder
		self.transform = transform
		self.sign = self.data['sign'][0]
		self.img = self.data['img_file']
		self.transform_img = transforms.Compose([self.transform])
		self.window_size=window_size
		self.length = len(self.img)
		self.positive=[]
		self.negative = []
		for i in range(len(self.sign)):
			if self.sign[i] ==1:
				self.positive.append(self.img[i])
			else:
				self.negative.append(self.img[i])
		random.shuffle(self.positive)
		random.shuffle(self.negative)
		self.data=[]
		self.length = min(len(self.positive),len(self.negative))
		print('pos:',len(self.positive))
		print('nega:',len(self.negative))
		for i in range(self.length):
			self.data.append((self.positive[i],1.0))
			self.data.append((self.negative[i],0.0))

		#random.shuffle(self.data)

	def __getitem__(self,idx):

		
		imgs_1 = []
		#imgs_2 = []
		for i in range(self.window_size):
			# j = int( idx+i-(self.window_size-1)/2)
		# 	#print(j)
		# 	if j<0:
		# 		j=0
		# 	elif j>self.length-1:
		# 		j = self.length-1


			#imgpath_1 = self.img_folder + self.data[idx][0][i].split()[0]
			imgpath_1 = self.img_folder + self.data[idx][0][i].split()[0]


			pic_1 = load_img(imgpath_1)
			#pic_2 = load_img(imgpath_2)

			pic_1 = self.transform_img(pic_1)# 3 * 224 * 224
			#print(pic.shape)
			#pic_2 = self.transform_img(pic_2)
			imgs_1.append(pic_1.unsqueeze(0))
			#imgs_2.append(pic_2.unsqueeze(0))

		imgseq_1 = torch.cat((imgs_1[0:self.window_size]),0).squeeze()
		#imgseq_2 = torch.cat((imgs_2[0:self.window_size]),0).squeeze()
		#gseq.shape)

		
		
		#sample = {'image':imgseq,'gt_height':gt_height}

		return imgseq_1,self.data[idx][1]
		#return 0
	def __len__(self):
		return self.length*2

# dataset = EyeBrowDataset('/dresden/users/rx43/ASL/eyebrow_detecting/dataset/train_all.mat','/dresden/users/rx43/ASL/eyebrow_detecting/',7,
# 	transform = transforms.Compose([transforms.ToTensor(),
# 	transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
# 	)
#
# print(dataset[15][0].shape)
