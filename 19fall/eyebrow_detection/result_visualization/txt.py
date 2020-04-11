import matplotlib.pyplot as plt   
import os
import numpy as np
import matplotlib.image as mpimg
import json
import scipy.io as scio
import pandas as pd



def enlarge(x):
	for i in range(len(x)):
		x[i]=x[i]*4-2
	return x




def smooth(x,window_size):
	y =x.copy()
	for i in range(len(x)):
		add = 0
		for j in range(int(i-(window_size-1)/2),int(i+(window_size-1)/2+1)):
			
			if j<0:
				j=0
			elif j>len(x)-1:
				j=len(x)-1
			add+=y[j]
		y[i]=add/window_size
	return y


def extract_label(x):
	pass

def main():
	sets = os.listdir('dataset2')

	for folder in sets:
		previous = scio.loadmat('dataset/'+folder+'.mat')
		frame = previous['frame'][0].tolist()
		d = {'frame':[],'eyebrow_height':[]}
		files = os.listdir('dataset2/'+folder)
		for file in files:
			

			part1 = file.split('cam2')[0]
			part2 = file.split('cam2')[1]
			new_name = part1+'cam1'+part2
			start_frame = eval(file.split('_')[-1].split('to')[0])
			end_frame = eval(file.split('_')[-1].split('to')[1][:-4])
			duration = end_frame-start_frame



		

			data = scio.loadmat('dataset2/'+folder+'/'+file)
			gt = enlarge(data['gt'][0])
			img_file = list(data['imgpath'])
			result = enlarge(data['predict'][0])
			for i in range(len(gt)):
				img_index =int(img_file[i].split('frame')[-1].split('.jpg')[0])
				if img_index in frame:
					
					d['frame'].append(int(img_index))
					d['eyebrow_height'].append(result[i])
				
			#print('finish ',file)
		df = pd.DataFrame(data=d)
		df = df.sort_values('frame')

		df.to_csv('dataset/'+folder+'.txt', header=None, index=None, sep=' ', mode='a')
		print('finish'+folder)

main()