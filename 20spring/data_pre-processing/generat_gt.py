
import json
import numpy as np
import scipy.io as scio
import os
import math
from os import path

height={"'further raised'":1.0,
        "'raised'":0.8,
        "'slightly raised'":0.4,
        "'slightly lowered'":-0.3,
        "'lowered'":-0.4,
        "'further lowered'":-0.5,
       }


jsons = os.listdir('eyebrows')
for jso in jsons:
    with open('eyebrows/'+jso) as f:
        data = json.load(f)

    uid=data['uid']
    length=data['length']
    
    f_id=[i for i in range(length)]
    eb_v=[0 for i in range(length)]
    for i in range(len(data['eyebrow_action'])):
        action=data['eyebrow_action'][i]['action']
        t1= data['eyebrow_action'][i]['onset']
        t2= data['eyebrow_action'][i]['s']
        t3= data['eyebrow_action'][i]['e']
        t4= data['eyebrow_action'][i]['offset']
        h = height[action]
        eb_v[t2:t3]=np.linspace(h,h,t3-t2)
        next_t2=None
        if i+1 < len(data['eyebrow_action']): #check the gap between two actions
            next_t2 = data['eyebrow_action'][i+1]['s']
            next_h = height[data['eyebrow_action'][i+1]['action']]
        if t1 != None and eb_v[t2-1] == 0:

            t1 = t1[0]
            print(t1,t2)
            eb_v[t1:t2]=np.linspace(0,h,t2-t1)
        if next_t2 and (next_t2 - t3 <= 20):
            print(next_t2)
          
            eb_v[t3:next_t2] = np.linspace(h,next_h,next_t2-t3)
            continue

        elif t4 != None and eb_v[t3] == 0:
            t4=t4[1]
            eb_v[t3:t4]=np.linspace(h,0,t4-t3)
        elif t4 == None :
            eb_v = eb_v[:t3]
            break


              
          
    f_id=(np.array([f_id]))
    eb_v=(np.array([eb_v]))
                  
    file= 'gt/'+str(uid)+'.mat'
    print(file)
    scio.savemat(file,{'f_id':f_id,'eb_v':eb_v})
            

