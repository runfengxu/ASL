import matplotlib.pyplot as plt   
import os
import numpy as np
import matplotlib.image as mpimg
import json
import scipy.io as scio
from matplotlib import gridspec
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


def get_others(file,l,img_file):
	eyeAperture = []
	yaw = []
	pitch = []
	roll= []
	df = pd.read_csv('eye_aperture_raw/'+file+'.txt',header= None)
	df[0] = df[0].map(lambda x: x.split('frameID: ')[-1])
	df[1] = df[1].map(lambda x: x.split('eyeAperture: ')[-1])
	df[2] = df[2].map(lambda x: x.split('pitch: ')[-1])
	df[3] = df[3].map(lambda x: x.split('yaw: ')[-1])
	df[4] = df[4].map(lambda x: x.split('roll: ')[-1])
	for i in range(l):
		index = eval(img_file[i].split('frame')[-1][:-4])
		index = df[df[0]==str(index)].index.tolist()
		if len(index)>0:
			index = index[0]
			
			eyeAperture.append(eval(df.iloc[index,1].split('eyeAperture:')[-1]))
			yaw.append(eval(df.iloc[index,3]))
			pitch.append(eval(df.iloc[index,2]))
			roll.append(eval(df.iloc[index,4]))
		else:
			eyeAperture.append(0)
			yaw.append(0)
			pitch.append(0)
			roll.append(0)
	return eyeAperture,yaw,pitch,roll



def main():
	sets = os.listdir('dataset')

	for file in sets:
		part1 = file.split('cam2')[0]
		part2 = file.split('cam2')[1]
		new_name = part1+'cam1'+part2
		new_name_2 = file.split('ss3')[0]+'ss3'
		start_frame = eval(file.split('_')[-1].split('to')[0])
		end_frame = eval(file.split('_')[-1].split('to')[1][:-4])
		duration = end_frame-start_frame



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
		img_file = list(data['imgpath'])
		result = enlarge(data['predict'][0])

		img_pre ='frame'+img_file[0].split('frame')[1]+'frame'
		
		eyeAperture,yaw,pitch,roll = get_others(new_name_2,len(gt),img_file)



		for i in range(len(gt)):
			
			fig = plt.figure(1,figsize=(10,3))
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
			x = np.arange(len(gt))
			y = gt
			y2 = result
			y5 = smooth(y2,5)
			y7 = smooth(y2,7)
			sub = duration-len(y)

			ax2 = plt.subplot(gs[0,1])
			plt.xlim((0, len(x)))

			for keys in label:
				s,e = eval(keys)
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
					e = len(y)-1
				#print(keys)

				#print(s,e)
				ax2.hlines(-2.1,s,e,color = 'red')
				ax2.text((0.3*s+0.7*e),-1.7,label[keys],fontsize=7,verticalalignment="center",horizontalalignment="center")

				ax2.vlines(s, -2.5, 2.5, colors = "lightseagreen", linestyles =(0, (1, 1)))
				ax2.vlines(e, -2.5, 2.5, colors = "lightseagreen", linestyles = (0, (1, 1)))
			#plt.plot(x,y,color = 'r')
			#plt.plot(x[3:-3],y2[3:-3],color = 'b',label='origin')
			#plt.plot(x,y5,color = 'c')
			ax2.plot(x[3:-3],y5[3:-3],color = 'orange',label='eyebrow',lw=1.5)
			ax2.plot(x[3:-3],eyeAperture[3:-3],color= 'lime',label='aperture',lw=1.5)
			ax2.legend(bbox_to_anchor=(0.8, 0.4), loc=2, borderaxespad=0., prop={'size': 6})
			ax2.vlines(i, -2.5, 2.5, colors = "c", linestyles = "-")
			ax2.text(4,-2,i,fontsize=8,verticalalignment="center",horizontalalignment="center")
			plt.setp(ax2.get_xticklabels(), visible=False)

			plt.tick_params(labelsize=4)

			ax3 =plt.subplot(gs[1,1])
			plt.xlim((0, len(x)))

			
			
			
			ax3.plot(x[3:-3],pitch[3:-3],color = 'cyan',label='pitch',lw=1)
			ax3.plot(x[3:-3],yaw[3:-3],label='yaw',lw=1)
			ax3.plot(x[3:-3],roll[3:-3],label='roll',lw=1)
			ax3.legend(bbox_to_anchor=(0.8, 0.4), loc=2, borderaxespad=0., prop={'size': 6})
			ax3.vlines(i, -0.5,0.5, colors = "c", linestyles = "-")
			
			plt.tick_params(labelsize=4)

			plt.rc('xtick',labelsize=4)
			plt.rc('ytick',labelsize=4)

			plt.savefig('draw/carol/'+file[:-4]+'/%d'%i,dpi = 800)
			plt.close()
			print(i)
		print('finish ',file)


main()