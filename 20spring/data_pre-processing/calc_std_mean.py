from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import h5py
import scipy.io as scio
import cv2
import numpy as np
import torch

class MyDataset(Dataset):
	def __init__(self,mat_file,h5):
		self.data = scio.loadmat(mat_file)['data']
		self.h5 = h5
		self.padd = self.padding()
		self.miss=set()
	def padding(self):
		image = cv2.imread('padding.png')
		image = cv2.resize(image,(128,128),interpolation = cv2.INTER_CUBIC)
		image = np.array(image)
		return image   

	def __getitem__(self, index):
		x = self.data[index]
		imgargs = x[0]
		height = x[1]
		imgs=[]
		for arg in imgargs:
			# print(arg)
			if ''.join(arg[0].split())=='padd':
				img=self.padd
			else:
				uid , fid =''.join(arg[0].split()),''.join(arg[1].split())
				
				key = uid+'/cropped_'+fid+'.png'
				#print(key)
				
				img=self.h5[str(key)][()]
			img = torch.tensor(img,dtype = torch.float)
			imgs.append(img.unsqueeze(0))
		imgseq = torch.cat(imgs,0).squeeze()        
		return imgseq
	def __len__(self):
		return len(self.data)

	
hf = h5py.File('cropped_frame.h5','r')
mat_file='train_dataset.mat'


dataset = MyDataset(mat_file,hf)
loader = DataLoader(
	dataset,
	batch_size=10,
	num_workers=0,
	shuffle=False
)

def online_mean_and_sd(loader):
	"""Compute the mean and sd in an online fashion

		Var[x] = E[X^2] - E^2[X]
	"""
	cnt = 0.
	fst_moment = torch.empty(3)
	snd_moment = torch.empty(3)
	mean = 0.
	std = 0.
	nb_samples = 0.
	print(len(loader))
	for i,data in enumerate(loader):
		if i%100 ==99:
			print(mean)
			print(std)
			print(nb_samples)
		#print(data.shape)
		batch_samples = data.size(0)
		data = data.view(batch_samples,-1, data.size(4))
		#print(data.shape)
		mean += data.mean(1).sum(0)
		std += data.std(1).sum(0)
		nb_samples += batch_samples
	mean/=nb_samples
	std/=nb_samples
	return mean,std
		# cnt = 0.
		#print(1)
		# if i%10==0:
		# 	print(i,'/',len(loader))
		# data = data.squeeze(0)
		# # print(data.shape)
		# b, h, w,c = data.shape
		# nb_pixels = b * h * w
		# nb_pixels = float(nb_pixels)
		# sum_ = torch.sum(data, dim=[0, 1, 2],dtype = torch.float)
		# sum_of_square = torch.sum(data ** 2, dim=[0, 1, 2],dtype = torch.float)
		
		# fst_moment = ((cnt * fst_moment + sum_) / (cnt + nb_pixels))
		# snd_moment = ((cnt * snd_moment + sum_of_square) / (cnt + nb_pixels))

		# cnt += nb_pixels
		# print(fst_moment)
		# print(torch.sqrt(snd_moment - fst_moment ** 2))
	# return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)

	
miss = []
mean, std = online_mean_and_sd(loader)
# mean = 0.
# std = 0.
# nb_samples = 0.
# print(len(loader))
# for i,data in enumerate(loader):
# 	print(data)
# 	batch_samples = data.size(0)
# 	data = data.view(batch_samples, data.size(1), -1)
# 	mean += data.mean(2).sum(0)
# 	std += data.std(2).sum(0)
# 	nb_samples += batch_samples

# mean /= nb_samples
# std /= nb_samples

scio.savemat('miss.mat',{'miss':miss})
#scio.savemat('mean_std.mat',{'mean':mean,'std':std})