import scipy.io as scio
import os

height = []

imgs = []
file = os.listdir('dataset/validation')

for i in file:
	data = scio.loadmat('dataset/validation/'+i)
	imgseq = data['img_file']
	gt = data['gt'][0]
	height.extend(gt)
	imgs.extend(imgseq)


scio.savemat('dataset/validation/validation_all.mat',{'gt':height,'img_file':imgs})


# d = scio.loadmat('dataset/train/train_all.mat')
# print(d['gt'].shape)
# print(d['img_file'].shape)