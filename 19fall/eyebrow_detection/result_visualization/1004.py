import matplotlib.pyplot as plt   
import os
import numpy as np
import matplotlib.image as mpimg
import json
import scipy.io as scio

def main():
	sets = os.listdir('new_gt')


	for video in sets:
		part1 = video.split('cam1')[0]
		part2 = video.split('cam1')[1]
		new_name = part1+'cam2'+part2
		
		files = os.listdir('new_gt/'+video)

		for file in files:


			if not os.path.exists('draw/dimitris/'+video+file):
				os.mkdir('draw/dimitris/'+video+file)

			new_gt = scio.loadmat('new_gt/'+video+'/'+file)['eb_v'][0]
			old_gt = scio.loadmat('old_gt/'+video+'/'+file)['eb_v'][0]
			
			img_file = scio.loadmat('new_gt/'+video+'/'+file)['f_id'][0]
			img_pre ='frames/'+new_name+'/frame'

			for i in range(len(old_gt)-2):
				plt.figure(1,figsize=(10,3))
				plt.subplot(1,2,1)
				# img_num=img_file[i].split('frame')[-1]
				# if img_num[-1]=='g':
				# 	new_img = img_pre+str(int(img_num[:-4])-2)+'.jpg'
				# else:
				# 	new_img = img_pre+str(int(img_num[:-5])-2)+'.jpg'
				image  = mpimg.imread(img_pre+str(img_file[i])+'.jpg')
				plt.imshow(image)
				plt.axis('off')
				x = np.arange(len(old_gt)-1)
				y = new_gt
				y2 = old_gt

			

				plt.subplot(1,2,2)
				plt.plot(x[:len(y)-1],y[:-1],color = 'r',label='new')
				plt.plot(x,y2[:-1],color = 'b',label='old')
				#plt.plot(x[3:-3],y3[3:-3],color = 'c',label='smooth_by_3')

				#plt.plot(x,y7,color = 'green',label = 'smooth_by_7')
				plt.legend()
				plt.vlines(i, -2.5, 2.5, colors = "c", linestyles = "-")
				plt.text(4,-2,i,fontsize=8,verticalalignment="top",horizontalalignment="right")
				plt.savefig('draw/dimitris/'+video+file+'/%d'%i,dpi = 320)
				plt.close()
				print(i)

			print('finish ',file)


main()