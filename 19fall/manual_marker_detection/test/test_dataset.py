from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import scipy.io as scio
import torch
import random
import numpy as np


def load_img(img_path):

	img = Image.open(img_path).convert('RGB')
	img = img.resize((224,224),Image.ANTIALIAS)
	return img

class Test_dataset(Dataset):
	def __init__(self,mat_file,img_folder, of_folder,window_size,transform =None):
		self.data = scio.loadmat(mat_file)
		self.img_folder = img_folder
		self.of_folder = of_folder
		self.transform = transform
		self.sign = self.data['sign'][0]
		self.img = self.data['img_file']
		self.transform_img = transforms.Compose([self.transform])
		self.window_size=window_size
		self.length = len(self.img)
		print(self.length)

		#random.shuffle(self.data)

	def __getitem__(self,idx):
		imgs_1 = []
		flows_1 = []
		#imgs_2 = []
		start = self.img[idx][0].split('/')[-1].split('.')[0][5:]
		start = int(start)
		end = self.img[idx][-1].split('/')[-1].split('.')[0][5:]
		end = int(end)
		for i in range(self.window_size):
			imgpath_1 = self.img_folder + self.img[idx][i]
			pic_1 = load_img(imgpath_1)
			pic_1 = self.transform_img(pic_1)# 3 * 224 * 224
			imgs_1.append(pic_1.unsqueeze(0))

			flow_path = self.of_folder + self.img[idx][i]
			flow_path = flow_path.replace("jpg", "npy")
			flow = np.load(flow_path,allow_pickle=True)
			flows_1.append(torch.from_numpy(flow).unsqueeze(0))

		imgseq_1 = torch.cat(imgs_1).squeeze()
		flowseq_1 = torch.cat(flows_1).squeeze() #shape(10,224,224,2)
		#imgseq_2 = torch.cat((imgs_2[0:self.window_size]),0).squeeze()
		#gseq.shape)
		
		#sample = {'image':imgseq,'gt_height':gt_height}
		imgseq_1 = imgseq_1.view(imgseq_1.shape[1], imgseq_1.shape[0], imgseq_1.shape[-2], imgseq_1.shape[-1])
		flowseq_1 = flowseq_1.view(flowseq_1.shape[-1], flowseq_1.shape[0], flowseq_1.shape[1], flowseq_1.shape[2])

		return (imgseq_1,flowseq_1),(start,end)
		#return 0
	def __len__(self):
		return self.length