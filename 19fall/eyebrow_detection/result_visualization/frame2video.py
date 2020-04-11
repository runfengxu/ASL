import os
import cv2
from PIL import Image
import numpy as np

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
size = (1560,600)
files = os.listdir('draw/carol')
for file in files:
	
	path = "draw/carol/"+file+'/'
	filelist =[f for f in os.listdir(path)]
	filelist.sort(key = lambda x: int(x[:-4]))
	index = 0
	f = filelist[0]
	img = cv2.imread(path+f)
	h=img.shape[0]
	w = img.shape[1]
	vw = cv2.VideoWriter('carol/'+file+'.avi', fourcc=fourcc, fps=10, frameSize=(w,h))

	for f in filelist:
	    f_read = cv2.imread(path+f)
	    f_img = Image.fromarray(f_read)  
	    f_rs = f_img  #.resize([1560,600],resample=Image.NONE)
	    f_out = np.array(f_rs)
	    # cv2.imwrite("file"+str(index)+".jpg",f_out)
	    # index+=1
	    vw.write(f_out)
	vw.release()
	print('finish:',file)


# fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# size = (1560,600)
# files = os.listdir('draw/dimitris')
# for file in files:
# 	vw = cv2.VideoWriter('dimi/'+file+'.avi', fourcc=fourcc, fps=10, frameSize=size)

# 	path = "draw/dimitris/"+file+'/'
# 	filelist =[f for f in os.listdir(path)]
# 	filelist.sort(key = lambda x: int(x[:-4]))
# 	index = 0
# 	for f in filelist[4:-4]:
# 	    f_read = cv2.imread(path+f)
# 	    f_img = Image.fromarray(f_read)  
# 	    f_rs = f_img.resize([1560,600],resample=Image.NONE)
# 	    f_out = np.array(f_rs)
# 	    # cv2.imwrite("file"+str(index)+".jpg",f_out)
# 	    # index+=1
# 	    vw.write(f_out)
# 	vw.release()
# 	print('finish:',file)