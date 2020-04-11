# store image data and tags to h5 file

import cv2
import json
import h5py
import shutil
import numpy as np
import os
import random
import scipy.io as scio
from multiprocessing import Pool

def load_json(pth):
    with open(pth) as f:
        data  = json.load(f)
        return data


def padding():

    image = cv2.imread('/dresden/users/rx43/20spring/padding.png')
    image = cv2.resize(image,(128,128),interpolation = cv2.INTER_CUBIC)
    image = np.array(image)
    return image


def get_img(uid,fid):
    image_pth = "/dresden/gpu2/yc984/ASL/grammar_marker/new_data/cropped_frame/{}/cropped_{}.png".format(uid,fid)
    image = cv2.imread(image_pth)
    image = cv2.resize(image,(128,128),interpolation = cv2.INTER_CUBIC)
    image = np.array(image)
    return image



def pro_images(idx,uid,eb,n):

    
    
        
    imgs=[] 
    for i in range(window_size):
        j = int(idx+i-(window_size-1)/2)
        if  j<0 or j>n::
            img = padd
        else:
            img = get_img(uid,j)
        imgs.append(img)
    input_data = np.array(imgs)
    return input_data,eb[idx]
    
            
if __name__ == '__main__':
    padd = padding()
    HEIGHT = 128
    WIDTH = 128
    CHANNELS = 3
    window_size = 11
    SHAPE  = (HEIGHT,WIDTH,CHANNELS)
    
    p = Pool(32)
    hf= h5py.File('train.h5','a')
    # set each utterance as a group
    train_files =  os.listdir('/dresden/users/rx43/20spring/train')
    for file in train_file:
        print(file)
        uid = file[:-4]
        f = scio.loadmat('/dresden/users/rx43/20spring/train/'+file)
        x = f['f_id'][0]
        eb = f['eb_v'][0]    
        
        for idx in range(len(x)):

            input_data,height =    pro_images(idx,uid,eb,len(x))
            
            xset = hf.create_dataset(
            name = 'X'+uid+'_'+str(idx),
            data = input_data,
            shape = input_data.shape,
            maxshape = input_data.shape,
            compression = "gzip",
            compression_opts =9)

            yset = hf.create_dataset(
            name = 'Y'+uid+'_'+str(idx),
            data = height,
            shape=(1,),
            maxshape=(None,),
            compression="gzip",
        compression_opts=9)
