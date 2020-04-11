from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import scipy.io as scio


def load_img(img_path):
	
	img = Image.open(img_path).convert('RGB')
	img = img.resize((256,256),Image.ANTIALIAS)
	return img

class EyeBrowDataset(Dataset):
	def __init__(self,mat_file,img_folder,window_size,transform =None):
		self.data = scio.loadmat(mat_file)
		self.img_folder = img_folder
		self.transform = transform
		self.eyebrow = self.data['gt'][0]
		self.img = self.data['img_file']
		self.transform_img = transforms.Compose([self.transform])
		self.window_size=window_size
		self.length = len(self.img)
	

	

	def __getitem__(self,idx):

		gt_height = (self.eyebrow[idx]+2)/4
		imgs = []
		for i in range(self.window_size):
			j = int( idx+i-(self.window_size-1)/2)
			if j<0:
				j=0
			elif j>self.length-1:
				j = self.length-1


			imgpath = self.img[j]
			while(imgpath[-1] == ' '):
				imgpath = imgpath[:-1]
			


			try:
				pic = load_img(imgpath)
				
			except FileNotFoundError:
				try:
					print('failed(1)')
					pic = load_img('backupimg/'+imgpath)
				except:
					print('failed(2)')
					pic = Image.new("RGB", (256, 256))
					


			imgs.append(self.transform_img(pic))
		imgs = [m.resize_([3,1,256,256]) for m in imgs]

		imgseq = torch.cat((imgs[0:self.window_size]),1)

		
		
		sample = {'image':imgseq,'gt_height':gt_height}


		return imgseq,gt_height
	def __len__(self):
		return len(self.eyebrow)