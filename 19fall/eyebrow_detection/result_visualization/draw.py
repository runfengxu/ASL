import matplotlib.pyplot as plt   
import os
import numpy as np
import matplotlib.image as mpimg
import json
import scipy.io as scio



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

	for file in sets:
		part1 = file.split('cam2')[0]
		part2 = file.split('cam2')[1]
		new_name = part1+'cam1'+part2
		start_frame = eval(file.split('_')[-1].split('to')[0])
		end_frame = eval(file.split('_')[-1].split('to')[1][:-4])
		duration = end_frame-start_frame



		if not os.path.exists('draw/dimitris/'+file[:-4]):
			os.mkdir('draw/dimitris/'+file[:-4])

		if os.path.exists('label/'+new_name[:-4]+'.txt'):
			f = open('label/'+new_name[:-4]+'.txt')
			label = json.load(f)
			f.close()

		else:
			print('not found')


		data = scio.loadmat('dataset2/'+file)
		gt = enlarge(data['gt'][0])
		img_file = list(data['imgpath'])
		result = enlarge(data['predict'][0])
		img_pre ='frame'+img_file[0].split('frame')[1]+'frame'

		for i in range(len(gt)):
			plt.figure(1,figsize=(10,3))
			plt.subplot(1,2,1)
			# img_num=img_file[i].split('frame')[-1]
			# if img_num[-1]=='g':
			# 	new_img = img_pre+str(int(img_num[:-4])-2)+'.jpg'
			# else:
			# 	new_img = img_pre+str(int(img_num[:-5])-2)+'.jpg'
			image  = mpimg.imread(img_file[i])
			plt.imshow(image)
			plt.axis('off')
			x = np.arange(len(gt))
			y = gt
			y2 = result
			y3 = smooth(y2,3)-0.5
			#y7 = smooth(y2,7)-1
			sub = duration-len(y)

			plt.subplot(1,2,2)
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

				#print(s,e)
				plt.hlines(-2,s,e,color = 'red')
				plt.text((0.3*s+0.7*e),-1.7,label[keys],fontsize=8,verticalalignment="top",horizontalalignment="center")
			
				plt.vlines(s, -2.5, 2.5, colors = "blue", linestyles = "dashed")
				plt.vlines(e, -2.5, 2.5, colors = "blue", linestyles = "dashed")
			plt.plot(x[3:-3],y[3:-3],color = 'r',label='gt')
			plt.plot(x[3:-3],y2[3:-3],color = 'b',label='pre')
			#plt.plot(x[3:-3],y3[3:-3],color = 'c',label='smooth_by_3')

			#plt.plot(x,y7,color = 'green',label = 'smooth_by_7')
			plt.legend()
			plt.vlines(i, -2.5, 2.5, colors = "c", linestyles = "-")
			plt.text(4,-2,i,fontsize=8,verticalalignment="top",horizontalalignment="right")
			plt.savefig('draw/dimitris/'+file[:-4]+'/%d'%i,dpi = 640)
			plt.close()

		print('finish ',file)


main()