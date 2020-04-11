import matplotlib.pyplot as plt   
import os
import numpy as np
import matplotlib.image as mpimg
import json
import scipy.io as scio
from matplotlib import gridspec
import pandas as pd
import argparse



def distance(x1,x2,x3,x4,x5,x6,x7,x8,y1,y2,y3,y4,y5,y6,y7,y8):

	d1 = np.sqrt(((x1+x2)/2-(x3+x4)/2)**2+((y1+y2)/2-(y3+y4)/2)**2)
	d2 = np.sqrt(((x5+x6)/2-(x7+x8)/2)**2+((y5+y6)/2-(y7+y8)/2)**2)
	
	return (d1+d2)/2
def normalize(x,maxi,mini):
	for i in range(len(x)):
		x[i]= (x[i]-mini)/(maxi-mini)
	return(x)

def enlarge(x):
	for i in range(len(x)):
		x[i]=x[i]*2-1
	for i in range(len(x)):
		if x[i]>0:
			x[i]=x[i]
		if x[i]<=0:
			x[i]=x[i]*0.5
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


def get_others(file,l,img_file):
	eyeAperture1 = []
	eyeAperture2=[]
	yaw = []
	pitch = []
	roll= []
	df = pd.read_csv('openface/'+file+'.csv',header= None)
	df2 = pd.read_csv('eye_aperture_raw/'+file+'.txt',header= None)
	df2[0] = df2[0].map(lambda x: x.split('frameID: ')[-1])
	df2[1] = df2[1].map(lambda x: x.split('eyeAperture: ')[-1])
	# df[2] = df[2].map(lambda x: x.split('pitch: ')[-1])
	# df[3] = df[3].map(lambda x: x.split('yaw: ')[-1])
	# df[4] = df[4].map(lambda x: x.split('roll: ')[-1])
	for i in range(l):
		idx = eval(img_file[i].split('frame')[-1].split('.jpg')[0])
		index1 = idx+1

		index2 = df2[df2[0]==str(idx)].index.tolist()
		x1 = float(df.iloc[index1,336])
		x2 = float(df.iloc[index1,337])
		x3 = float(df.iloc[index1,340])
		x4 = float(df.iloc[index1,339])

		y1 = float(df.iloc[index1,404])
		y2 = float(df.iloc[index1,405])
		y3 = float(df.iloc[index1,408])
		y4 = float(df.iloc[index1,407])

		x5 = float(df.iloc[index1,342])
		x6 = float(df.iloc[index1,343])
		x7 = float(df.iloc[index1,346])
		x8 = float(df.iloc[index1,345])
		y5 = float(df.iloc[index1,410])
		y6 = float(df.iloc[index1,411])
		y7 = float(df.iloc[index1,414])
		y8 = float(df.iloc[index1,413])
		
		eyeAperture2.append(distance(x1,x2,x3,x4,x5,x6,x7,x8,y1,y2,y3,y4,y5,y6,y7,y8))
		yaw.append(float(df.iloc[index1,642])*180/3.14)
		# print(index1)
		# print(df.iloc[index1,641])
		# print(type(df.iloc[index1,641]))
		pitch.append(float(df.iloc[index1,641])*180/3.14*-1)
		roll.append(float(df.iloc[index1,640])*180/3.14)
		if len(index2)>0:
			index2 = index2[0]
			
			eyeAperture1.append(eval(df2.iloc[index2,1].split('eyeAperture:')[-1]))
			
		else:
			try:
				eyeAperture1.append(eyeAperture1[-1])
			except:
				eyeAperture1.append(distance(x1,x2,x3,x4,x5,x6,x7,x8,y1,y2,y3,y4,y5,y6,y7,y8)/5)
			
	return eyeAperture1,eyeAperture2,yaw,pitch,roll



def main(f):
	file = f
	part1 = file.split('cam2')[0]
	part2 = file.split('cam2')[1]
	new_name = part1+'cam1'+part2
	new_name_2 = file.split('ss3')[0]+'ss3'
	start_frame = eval(file.split('_')[-1].split('to')[0])
	end_frame = eval(file.split('_')[-1].split('to')[1][:-4])
	duration = end_frame-start_frame
	# m = scio.loadmat('info/'+m_file)
	# g = scio.loadmat('info/'+g_file)
	sign = scio.loadmat('sign/'+'_'.join(file.split('_')[:2])+'.mp4/'+str(start_frame)+'to'+str(end_frame))
	manual= sign['sign']

	special = sign['special']
	special = sorted(special, key=lambda x: x[1])


	if not os.path.exists('draw/carol/'+file[:-4]):
		os.mkdir('draw/carol/'+file[:-4])

	if os.path.exists('label/'+new_name[:-4]+'.txt'):
		f = open('label/'+new_name[:-4]+'.txt')
		label = json.load(f)
		f.close()

	else:
		print('not found')


	data = scio.loadmat('dataset/'+file)
	gt = enlarge(data['gt'][0])
	img_file = list(data['path'])
	result = enlarge(data['pred'][0])

	img_pre ='frame'+img_file[0].split('frame')[1]+'frame'
	
	eyeAperture1,eyeAperture2,yaw,pitch,roll = get_others(new_name_2,len(gt),img_file)

	pitch = smooth(pitch,5)
	yaw = smooth(yaw,5)
	roll = smooth(roll,5)
	eyeAperture1 = normalize(smooth(eyeAperture1,5),3.5,0)
	#print(max(eyeAperture1))
	#print(min(eyeAperture1))
	for i in range(len(gt)):
		
		fig = plt.figure(1,figsize=(7.8,3))
		gs = gridspec.GridSpec(2,2)
		gs.update(wspace=0,hspace = 0.05)
		ax1 = plt.subplot(gs[:,0])

		# if img_num[-1]=='g':
		#     new_img = img_pre+str(int(img_num[:-4])-2)+'.jpg'
		# else:
		#     new_img = img_pre+str(int(img_num[:-5])-2)+'.jpg'
		image  = mpimg.imread(img_file[i])
		ax1.imshow(image)
		ax1.axis('off')
		ax1.spines['top'].set_visible(False)
		ax1.spines['right'].set_visible(False)
		ax1.spines['bottom'].set_visible(False)
		ax1.spines['left'].set_visible(False)

		x = np.arange(len(gt))
		y = gt
		y2 = result
		y5 = smooth(y2,5)
		y7 = smooth(y2,7)
		sub = duration-len(y)


		ax2 = plt.subplot(gs[0,1])
		plt.xlim((0, len(x)))
		plt.ylim((-2,1.05))
	


		ax2.plot(x[3:-3],y2[3:-3],color = 'red',label='eyebrow',lw=1.5)
		#ax2.plot(x[3:-3],eyeAperture1[3:-3],color= 'lime',label='aperture',lw=1.5)
		ax2.plot(x[3:-3],eyeAperture1[3:-3],color= 'lime',label='aperture',lw=1.5)
		ax2.legend(bbox_to_anchor=(0.8, 0.4), loc=2, borderaxespad=0., prop={'size': 6})
		ax2.vlines(i, -2.5, 2.5, colors = "black", linestyles = "-",lw = 0.5)
		ax2.text(2.6,-1.8,i,fontsize=8,verticalalignment="center",horizontalalignment="center")
		plt.setp(ax2.get_xticklabels(), visible=False)
		if len(special)>0:
			for k in range(len(special)):
				s=int(special[k][1])
				e=int(special[k][2])
				if s-start_frame<0:
					s = 0
				else: 
					try:
						s = img_file.index(img_pre+str(s)+'.jpg')-1
					except:
						s = img_file.index(img_pre+str(s)+'.jpg ')-1
				if e-start_frame<len(y)-1:
					try:
						e = img_file.index(img_pre+str(e)+'.jpg')-1
					except:
						e = img_file.index(img_pre+str(e)+'.jpg ')-1
				else:
					e = len(y)-4
				
				ax2.hlines(-1.4+(k%2)*0.8,s,e,color = 'blue',lw =3 )
				ax2.text((0.3*s+0.7*e),-1.7+(k%2)*0.8,special[k][0].split()[0][1:-1], ha = 'center' ,va = 'bottom',fontsize=7,)

				ax2.vlines(s, -2.5, 2.5, colors = "darkturquoise", linestyles ='dotted',lw = 0.5)
				ax2.vlines(e, -2.5, 2.5, colors = "darkturquoise", linestyles = 'dotted',lw = 0.5)
	 



		plt.tick_params(labelsize=4)

		ax3 =plt.subplot(gs[1,1])
		plt.xlim((0, len(x)))

		plt.ylim((-30,30))
		
		for l in range(len(manual)):
			s=int(manual[l][1])-start_frame
			e=int(manual[l][2])-start_frame
			ax3.hlines(25.5,s,e,color = 'magenta',lw = 3)
			ax3.text((s),24,manual[l][0].split()[0][1:-1], ha = 'left' ,va = 'top',fontsize=5,rotation = -90)

			ax3.vlines(s, -30, 30, colors = "darkturquoise", linestyles = 'dotted',lw = 0.5)
			ax3.vlines(e, -30, 30, colors = 'darkturquoise', linestyles = 'dotted',lw = 0.5)

		
		ax3.plot(x[3:-3],pitch[3:-3],color = 'blue',label='pitch',lw=1)
		ax3.plot(x[3:-3],yaw[3:-3],color = 'red' ,label='yaw',lw=1)
		ax3.plot(x[3:-3],roll[3:-3],label='roll',color = 'lime',lw=1)
		ax3.legend(bbox_to_anchor=(0.8, 0.4), loc=2, borderaxespad=0., prop={'size': 6})
		




		ax3.vlines(i, -30,30, colors = "black", linestyles = "-",lw = 0.5)
		




		plt.tick_params(labelsize=4)

		plt.rc('xtick',labelsize=4)
		plt.rc('ytick',labelsize=4)

		
		plt.subplots_adjust(top = 0.92, bottom = 0.08, right =0.99, left = 0, hspace = 0.05, wspace = 0)
		plt.margins(0,0)

		plt.savefig('draw/carol/'+file[:-4]+'/%d'%i,dpi = 200)
		plt.close()
		print(i)
	print('finish ',file)





files= os.listdir('dataset')
file_name={}
for file in files:
	file_name[file] = int(file.split('_')[-1].split('to')[0])
files = sorted(file_name.items(), key=lambda x: x[1])
count = 1
for i in range(len(files)):
	main(files[i][0])
	count+=1
	


