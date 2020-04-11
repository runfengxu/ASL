import scipy.io as scio
import os
import numpy as np



def generate_sequence(folder,file):
    eyebrow_height=[]
    pics_path =[]
    f = scio.loadmat('gt/'+folder+'/'+file)
    t = f['f_id'][0]
    v = f['eb_v'][0]
    missed  = 0
    for i in range(len(v)):
        if os.path.exists('frames/'+'cam2'.join(folder.split('cam1'))+'/frame'+str(t[i])+'.jpg'):
            eyebrow_height.append(v[i])
            pics_path.append('frames/'+'cam2'.join(folder.split('cam1'))+'/frame'+str(t[i])+'.jpg')
        else:
            #print(folder+'_'+file+str(i))
            missed+=1

    file_name = 'dataset/'+'cam2'.join(folder.split('cam1'))+'_'+file+'.mat'
    scio.savemat(file_name,{'gt':eyebrow_height,'img_file':pics_path})


    print('folder:',folder,'file:',file,'   missed:',missed)

            

folders = os.listdir('gt')

for folder in folders:
    files = os.listdir('gt/'+folder)
    for file in files:
        generate_sequence(folder,file)







    #     if file[-3:]== 'mat':
    #         utterance_id = file.split('_')[0]
    #         start_frame = file.split('_')[1].split('to')[0]

    #         l[utterance_id]=scio.loadmat(file)['eb_v']
        
    # return l
        


# def data_alignment():

#     newdict={}
#     d = []

#     numbers = {}  # key:utterance idx(0-714), value: img number
#     for i in range(715):
#         numbers[str(i)]=0


#     pics = os.listdir('image/')


#     for pic in pics:
#         index = pic.split('im')[1].split('_')[1]

#         numbers[index]+=1

#     count = 0       

#     for i in range(715):

#         uid=raw['raw_info'][i][0][0].split('ID')[1].split('_')[1]
#         frames = len(l[uid][0])
#         imgs = numbers[str(i)]

#         if frames == imgs:
#             newdict[str(i)]=l[uid][0]
#         elif frames > imgs:
#             newdict[str(i)]=l[uid][0][:imgs]
#         elif frames < imgs:
#             newdict[str(i)]=np.append(l[uid][0],[l[uid][0][-1]]*int(imgs-frames))
#     return newdict





# def main():
# 	extract_gt_sequence()
# 	raw = scio.loadmat('raw/raw_info.mat')
# 	data_alignment()

# 	for i in range(715):
# 	    eyebrow_height=[]
# 	    img_file = []
# 	    for j in range(len(newdict[str(i)])):
# 	        eyebrow_height.append(newdict[str(i)][j])
# 	        file_name = 'face_im_'+str(i)+'_'+str(j)+'.jpg'
# 	        img_file.append(file_name)
# 	    file = 'dataset/data_ut_'+str(i)+'.mat'
# 	    scio.savemat(file,{'gt':eyebrow_height,'img_file':img_file})


# main()
