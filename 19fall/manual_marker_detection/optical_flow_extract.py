import scipy.io as scio
import os
import numpy as np
import matplotlib.image as mpimg
import cv2
import threading

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


folders=['2011-12-01_0043-cam1-for-ss3',
       '2011-12-08_0045-cam1-for-ss3',
       '2011-12-08_0046-cam1-for-ss3',
       '2012-01-27_0050-cam1-for-ss3',
       '2012-01-27_0051-cam1-for-ss3',
       '2012-01-27_0052-cam1-for-ss3',
       '2012-01-27_0053-cam1-for-ss3',
       '2012-01-27_0055-cam1-for-ss3',
       '2012-01-27_0056-cam1-for-ss3',
       '2012-02-09_0057-cam1-for-ss3',
       '2012-02-09_0058-cam1-for-ss3',
       '2012-02-09_0059-cam1-for-ss3',
       '2012-02-09_0061-cam1-for-ss3',
       '2012-02-14_0063-cam1-for-ss3',
       '2012-02-14_0065-cam1-for-ss3',
       '2013-06-27_CB_0107-cam1-for-ss3',
       '2013-06-27_CB_0110-cam1-for-ss3',
       '2013-06-27_CB_0111-cam1-for-ss3',
       '2013-06-27_CB_0112-cam1-for-ss3']

optical_flow = cv2.DualTVL1OpticalFlow_create()
x = np.zeros((224, 224,2), dtype=np.float32)
for folder in folders:
	rgb_frames = os.listdir('frame/'+folder)
	os.mkdir('optical_flow_matrix/'+folder)
	for i in range(len(rgb_frames)-1):
		prvs = cv2.imread("C:/Users/cbim v left/Desktop/manual/frame/"+folder+'/frame%d.jpg'%i, 0)
		next = cv2.imread("C:/Users/cbim v left/Desktop/manual/frame/"+folder+'/frame%d.jpg'%(i+1),0)
		prvs = cv2.resize(prvs,(224,224))
		next = cv2.resize(next,(224,224))
		flow = optical_flow.calc(prvs, next, None)
		#nomalization
		x=np.array([-1.0,-1.0],dtype=np.float32)
		x = np.expand_dims(x, axis =0)
		x = np.expand_dims(x, axis =0)
		flow = (flow-x)/2

		# hsv_image = draw_hsv(flow)
		# cv2.imwrite("C:/Users/cbim v left/Desktop/manual/optical_flow_frames/"+folder+'/frame%d.jpg'%(i+1),hsv_image)
		np.save("C:/Users/cbim v left/Desktop/manual/optical_flow_matrix/"+folder+'/frame%d'%(i), flow)
	np.save("C:/Users/cbim v left/Desktop/manual/optical_flow_matrix/"+folder+'/frame%d'%(len(rgb_frames)-1), x)