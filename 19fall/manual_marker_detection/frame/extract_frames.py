import cv2
import os



def FrameCapture(path1,path2):
	vidObj = cv2.VideoCapture(path1)
	count = 0
	success = 1
	while success:
		success,image = vidObj.read()
		cv2.imwrite(path2+'/frame%d.jpg' % count,image)
		count+=1
		print(path1,count)

if __name__ =='__main__':

	files = os.listdir('videos')
	for file in files:
		if file[-3:] =='mp4':
			if not os.path.exists(file[:-3]):
				os.makedirs(file[:-3])
			path1 = 'videos/'+file
			path2 = file[:-3]
			FrameCapture(path1,path2)


	# path = 'videos/2011-12-01_0043-cam2-for-ss3.mp4'
	# FrameCapture(path)