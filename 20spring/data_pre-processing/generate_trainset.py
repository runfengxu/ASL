import os
import scipy.io as scio
train_files = os.listdir('train')
train_dataset = []
window_size=11
for file in train_files:
	print(file)
	uid = file[:-4]
	f = scio.loadmat('train/'+file)
	x = f['f_id'][0]
	eb = f['eb_v'][0]    
	n=len(x)
	for idx in range(len(eb)):
		height = eb[idx]
		imgs=[] 
		for i in range(window_size):
			j = int(idx+i-(window_size-1)/2)
			if  j<0 or j>n-1:
				img = ('padd',-1)
			else:
				img = (uid,j)
			imgs.append(img)
		train_dataset.append((imgs,height))

scio.savemat('train_dataset.mat',{'data':train_dataset})
		
		