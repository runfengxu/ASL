import os
import numpy as np
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

x = np.zeros((224, 224,2), dtype=np.float32)
for folder in folders:
       rgb_frames = os.listdir('optical_flow_matrix/'+folder)
       for i in range(len(rgb_frames)):
              os.rename("C:/Users/cbim v left/Desktop/manual/optical_flow_matrix/"+folder+'/frame%d.npy'%(i+1), 
                     "C:/Users/cbim v left/Desktop/manual/optical_flow_matrix/"+folder+'/frame%d.npy'%(i))
       print("finish 1")
       np.save("C:/Users/cbim v left/Desktop/manual/optical_flow_matrix/"+folder+'/frame%d'%(len(rgb_frames)), x)