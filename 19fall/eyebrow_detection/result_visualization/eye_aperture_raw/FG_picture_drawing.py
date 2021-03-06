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

	try:
		sign = scio.loadmat('sign/'+'_'.join(file.split('_')[:2])+'.mp4/'+str(start_frame)+'to'+str(end_frame))
	except:
		sign = scio.loadmat('sign/'+new_name+'.mp4/'+str(start_frame)+'to'+str(end_frame))
	manual= sign['sign']

	special = sign['special']
	special = sorted(special, key=lambda x: x[1])


	if not os.path.exists('draw/carol/'+file[:-4]):
		os.mkdir('draw/carol/'+file[:-4])


	f1 = open('gt_onoffset/'+''.join(f.split('cam2-for-ss3'))[:-4]+'.txt')
	label = json.load(f1)
	f1.close()

	f2 = open('pred_onoffset/'+''.join(f.split('cam2-for-ss3'))[:-4]+'.txt')
	label2 = json.load(f2)
	f2.close()


	data = scio.loadmat('dataset/'+file)
	gt = data['gt'][0]
	img_file = list(data['path'])
	
	for i in range(len(img_file)):
		img_file[i] = img_file[i].split()[0]

	result = data['pred'][0]

	img_pre ='frame'+img_file[0].split('frame')[1]+'frame'
	
	
	lm =enlarge(normalize(get_others(new_name_2,len(gt),img_file))) #lm is eye_brow_height calculated by landmark

	print(max(lm))
	print(min(lm))
		
	fig = plt.figure(1)  
	
	gs = gridspec.GridSpec(2,3,height_ratios=[1.4,2.7]) # 3 pics on upper subgraph, 1


	# ax 1 to 3 are pics
	gs.update(wspace=0.0,hspace = 0)
	ax1 = plt.subplot(gs[0,0])

	
	image  = mpimg.imread(img_file[0])
	ax1.imshow(image)
	ax1.axis('off')
	ax1.spines['top'].set_visible(False)
	ax1.spines['right'].set_visible(False)
	ax1.spines['bottom'].set_visible(False)
	ax1.spines['left'].set_visible(False)
	
	ax2 = plt.subplot(gs[0,1])
	image  = mpimg.imread(img_file[10])
	ax2.imshow(image)
	ax2.axis('off')
	ax2.spines['top'].set_visible(False)
	ax2.spines['right'].set_visible(False)
	ax2.spines['bottom'].set_visible(False)
	ax2.spines['left'].set_visible(False)
	
	ax3 = plt.subplot(gs[0,2])	
	image  = mpimg.imread(img_file[-5])
	ax3.imshow(image)
	ax3.axis('off')
	ax3.spines['top'].set_visible(False)
	ax3.spines['right'].set_visible(False)
	ax3.spines['bottom'].set_visible(False)
	ax3.spines['left'].set_visible(False)
	
	x = np.arange(len(gt))
	y1 = smooth(result,5) # eye_brow prediction
	y2 = smooth(lm,5)	# eye_brow from landmark
	
	ax4 = plt.subplot(gs[1,:])
	plt.xlim((0, len(x)))
	plt.ylim((-3.5,1.2))


	ax4.plot(x[3:-3],y1[3:-3],color = 'red',label='deep learning',lw=1.5)
	ax4.plot(x[3:-3],y2[3:-3],color= 'lime',label='landmark',lw=1.5)
	plt.xlabel('frame',fontsize=10)
	plt.ylabel('Eyebrow Movement Intensity',fontsize=10)
	ax4.tick_params(axis='both',labelsize=100)
	plt.setp(ax4.get_xticklabels(), visible=False)
	plt.setp(ax4.get_yticklabels(), visible=False)
	


	plt.tick_params(labelsize=4)

	e=0
	# Draw manual marker
	for l in range(len(manual)):
		s=int(manual[l][1])-start_frame
		if s<e+5:
			continue
		e=int(manual[l][2])-start_frame
		ax4.hlines(-3.2,s,e,color = 'magenta',lw = 3)
		ax4.text((s),-3.1,manual[l][0].split()[0][1:-1], ha = 'left' ,va = 'bottom',fontsize=9,rotation = -90)

		ax4.vlines(s, -3.5, 1.3, colors = "darkturquoise", linestyles = 'dotted',lw = 0.5)
		ax4.vlines(e, -3.5, 1.3, colors = 'darkturquoise', linestyles = 'dotted',lw = 0.5)

	
	# draw human-annotated on-offset
	for keys,values in label.items():
		count=1
		# print (keys)\
		motion = keys
		height={'further raised':0.8,
        'raised':0.0,
        'slightly raised':0.2,
        'left raised/right furrowed':0,
        'right raised/left furrowed':0,
        'left raised/right lowered':-0.4,
        'right raised/left lowered':-0.4,
        'raised-furrowed':0.3,
        'slightly lowered':-0.2,
        'lowered':-0.2,
        'further lowered':-0.7,
       }

		for value in values:
			count+=1
			s1,s2,s3,s4 = value
			ax4.hlines(+height[motion]-0.6-count%2*0.2,max(s2,3),s3,color ='red' ,lw=3)
			if s1!=None:
				ax4.hlines(height[motion]-0.6-count%2*0.2,max(s1,3),s2,color ='green' ,lw=3)
			if s4!=None:
				ax4.hlines(height[motion]-0.6-count%2*0.2,s3,s4,color = 'blue',lw=3)
			ax4.text(s2,height[motion]-0.55-count%2*0.2,'human-annotation', ha = 'left' ,va = 'bottom',fontsize=10)
	# draw predicted on-off set
	for keys,values in label2.items():
		count=1

		motion = keys
		height={'further raised':0.8,
        'raised':0.0,
        'slightly raised':0.2,
        'left raised/right furrowed':0,
        'right raised/left furrowed':0,
        'left raised/right lowered':-0.4,
        'right raised/left lowered':-0.4,
        'raised-furrowed':0.3,
        'slightly lowered':-0.2,
        'lowered':-0.2,
        'further lowered':-0.7,
       }
		for value in values:
			count+=1
			s1,s2,s3,s4 = value
			ax4.hlines(height[motion]-1.1-count%2*0.2,max(s2,3),s3,color = 'red',lw=3)
			if s1!=None:
				ax4.hlines(height[motion]-1.1-count%2*0.2,max(s1,3),s2,color = 'green',lw=3)
			if s4!=None:
				ax4.hlines(height[motion]-1.1-count%2*0.2,s3,s4,color = 'blue',lw=3)
			ax4.text(s2,height[motion]-1.05-count%2*0.2,'prediction', ha = 'left' ,va = 'bottom',fontsize=10)
	plt.tick_params(labelsize=4)
	plt.rc('xtick',labelsize=4)
	plt.rc('ytick',labelsize=4)
	plt.savefig('draw/carol/'+file[:-4]+'/%d'%i,dpi = 400)
	plt.close()
	
	print('finish ',file)




files= os.listdir('dataset')

for i in range(len(files)):
	try:
		main(files[i])
	except:
		continue
	