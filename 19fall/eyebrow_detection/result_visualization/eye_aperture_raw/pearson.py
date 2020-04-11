import matplotlib.pyplot as plt   
import os
import numpy as np
import matplotlib.image as mpimg
import json
import scipy.io as scio
from matplotlib import gridspec
import pandas as pd
import argparse

def enlarge(x):
	for i in range(len(x)):
		x[i]=x[i]*2-1
	for i in range(len(x)):
		if x[i]>0:
			x[i]=x[i]
		if x[i]<=0:
			x[i]=x[i]*0.5
	return x

def distance(x1,x2,y1,y2,z1,z2):

	d1 = np.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)

	
	return d1
def normalize(x):
	for i in range(len(x)):
		x[i]= ((x[i]-18)/(45-18))
	return(x)

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

def get_others(file,l,img_file):
	eyebrow = []
	df = pd.read_csv('openface/'+file+'.csv')
	
	for i in range(l):
		idx = eval(img_file[i].split('frame')[-1].split('.jpg')[0])
		

		
		x1 = float(df[' X_41'][idx])
		x2 = float(df[' X_40'][idx])
		x3 = float(df[' X_39'][idx])
		x4 = float(df[' X_42'][idx])
		x5 = float(df[' X_47'][idx])
		x6 = float(df[' X_46'][idx])

		X1 = float(df[' X_19'][idx])
		X2 = float(df[' X_20'][idx])
		X3 = float(df[' X_21'][idx])
		X4 = float(df[' X_22'][idx])
		X5 = float(df[' X_23'][idx])
		X6 = float(df[' X_24'][idx])
	
		y1 = float(df[' Y_41'][idx])
		y2 = float(df[' Y_40'][idx])
		y3 = float(df[' Y_39'][idx])
		y4 = float(df[' Y_42'][idx])
		y5 = float(df[' Y_47'][idx])
		y6 = float(df[' Y_46'][idx])

		Y1 = float(df[' Y_19'][idx])
		Y2 = float(df[' Y_20'][idx])
		Y3 = float(df[' Y_21'][idx])
		Y4 = float(df[' Y_22'][idx])
		Y5 = float(df[' Y_23'][idx])
		Y6 = float(df[' Y_24'][idx])
	
		z1 = float(df[' Z_41'][idx])
		z2 = float(df[' Z_40'][idx])
		z3 = float(df[' Z_39'][idx])
		z4 = float(df[' Z_42'][idx])
		z5 = float(df[' Z_47'][idx])
		z6 = float(df[' Z_46'][idx])
		Z1 = float(df[' Z_19'][idx])
		Z2 = float(df[' Z_20'][idx])
		Z3 = float(df[' Z_21'][idx])
		Z4 = float(df[' Z_22'][idx])
		Z5 = float(df[' Z_23'][idx])
		Z6 = float(df[' Z_24'][idx])

		d1 = distance(x1,X1,y1,Y1,z1,Z1)
		d2 =distance(x2,X2,y2,Y2,z2,Z2)
		d3 =distance(x3,X3,y3,Y3,z3,Z3)
		d4 = distance(x4,X4,y4,Y4,z4,Z4)
		d5 = distance(x5,X5,y5,Y5,z5,Z5)
		d6 = distance(x6,X6,y6,Y6,z6,Z6)

		eyebrow.append((d1+d2+d3+d4+d5+d6)/6)
	
		
	return eyebrow



def main(f):
	# print(f)
	file = f
	part1 = file.split('cam2')[0]
	part2 = file.split('cam2')[1]
	new_name = part1+'cam1'+part2.split('_')[0]
	new_name_2 = file.split('ss3')[0]+'ss3'
	start_frame = eval(file.split('_')[-1].split('to')[0])
	end_frame = eval(file.split('_')[-1].split('to')[1][:-4])
	duration = end_frame-start_frame
	


	data = scio.loadmat('dataset/'+file)
	gt = data['gt'][0]
	img_file = list(data['path'])
		
	lm =get_others(new_name_2,len(gt),img_file)

	return gt,lm



files= os.listdir('dataset')
groundtruth=np.array([])
landmark=np.array([])
for i in range(len(files)):
	
	gt,lm = main(files[i])
	groundtruth =np.concatenate((groundtruth,gt))
	landmark=np.concatenate((lm,landmark))
	print('finish:',files[i])
	
# ave1 = np.average(groundtruth)
# ave2 = np.average(landmark)

# denominator= np.sqrt(np.sum((groundtruth-ave1)**2))*np.sqrt(np.sum((landmark-ave2)**2))
# numerator = np.sum((groundtruth-ave1)*(landmark-ave2))
print(np.abs(groundtruth,enlarge(normalize(landmark))).mean())
#print(numerator/denominator)