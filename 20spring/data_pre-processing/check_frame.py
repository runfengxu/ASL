#check whether a frame is in the server or not
import os 

file1  = os.listdir('/dresden/gpu2/yc984/ASL/grammar_marker/new_data/cropped_frame')
file2 = os.listdir('/dresden/users/rx43/20spring/eyebrows')

for i in range(len(file2)):
	file2[i] = file2[i][:-5]

file1=set(file1)
file2 = set(file2)

inter =file2.intersection(file1)

file2.difference_update(inter)
file1.difference_update(inter)

for file in file2:
	os.remove('/dresden/users/rx43/20spring/eyebrows/'+file+'.json')
