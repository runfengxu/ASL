import scipy.io as scio
import os
import numpy as np
import pandas as pd
from math import ceil
import random

def intersection(s1,e1,s2,e2):
    s = max(s1,s2)
    e = min(e1,e2)
    if e-s>0:
        return (e-2)
    else:
        return 0

def check_sign(s,e,sign):
    status = 0
    for item in sign:
        sign_start = item[0]
        sign_end = item[1]
        overlap = intersection(sign_start,sign_end,s,e)
        percentage = overlap/(sign_end-sign_start)
        if percentage>0.8:
            status = 1
    return status







def generate_sequence(video_name,file_name,start,end):
    
    pics_path =[]
    df = pd.read_csv('../sign/csv/'+file)
    sign = []
    gt = []


    # read all sign
    for i in range(len(df)):
        s2 = ceil((df.iloc[i,-2]))
        e2 = ceil((df.iloc[i,-1]))
        if e2>s2:
            sign.append((s2,e2))


            # for a clip, check if anyny sign is in this clip
    for j in range(start+5,end-6,1):
        img = []
        for i in range(-5,6):
            img.append(video_name+'/frame'+str(j+i)+'.jpg')
        pics_path.append(img)
        
        # check if any sign in this range
        if (check_sign(j-5,j+5,sign)):
            gt.append(1)
        else:
            gt.append(0)
    


    file_name = file_name[:-4]+'.mat'
    scio.savemat(file_name,{'sign':gt,'img_file':pics_path})


    print('file:',file,)

def generate_multi_classification():
    d1={'Classifiers':1,
 'Fingerspelled Signs':2,
 'Gestures':3,
 'Lexical Signs':4,
 'Loan Signs':5}
    files = os.listdir('../sign/label_type')
    gt = []
    for file in files:
        print(file)
        video = '_'.join(file.split('_')[:-2])
        start_frame = eval(file.split('_')[-1].split('to')[0])
        end_frame = eval(file.split('_')[-1].split('to')[1][:-4])
        df = pd.read_csv('../sign/label_type/'+file)
        for i in range(len(df)):
            s = ceil((df.iloc[i,-2]))
            e = ceil((df.iloc[i,-1]))
            if s-5>start_frame and e+5<end_frame: 
                img = []
                for i in range(s-5,e+5):
                    img.append(video+'/frame'+str(i)+'.jpg')
                gt.append((df['SIGN_TYPE'][i],img))
        print('finish:',file)
    random.shuffle(gt)
    n = ceil(0.9*len(gt))
    classes=[]
    pics_path=[]
    for i in gt[:n]:
        classes.append(i[0])
        pics_path.append(i[1])
    file_name = 'classify_train.mat'
    scio.savemat(file_name,{'classid':classes,'img_file':pics_path})

    classes=[]
    pics_path=[]
    for i in gt[n:]:
        classes.append(i[0])
        pics_path.append(i[1])
    file_name = 'classify_vali.mat'
    scio.savemat(file_name,{'classid':classes,'img_file':pics_path})

        
# #main  

# files = os.listdir('../sign/csv')

# #for every utterance , create a mat file
# for file in files:
#     video = '_'.join(file.split('_')[:-2])
#     start_frame = eval(file.split('_')[-1].split('to')[0])
#     end_frame = eval(file.split('_')[-1].split('to')[1][:-4])

#     generate_sequence(video,file,start_frame,end_frame)
generate_multi_classification()

