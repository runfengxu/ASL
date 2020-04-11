from xml.dom.minidom import parse
import xml.dom.minidom
import json
import numpy as np
import scipy.io as scio
import os
import math
from os import path

	



DOMTree = xml.dom.minidom.parse('283_68124copy.xml')
collection= DOMTree.documentElement

utterances=collection.getElementsByTagName('UTTERANCE')

videos=['2011-12-01_0043-cam1-for-ss3',
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

l={}
for video in videos:
	l[video]=[]

	
	
ut={}
s=[]
count =0
for utterance in utterances:
	count+=1
	ut={}
	uid = utterance.getAttribute('ID')
	stf= int(math.ceil(eval(utterance.getAttribute('START_FRAME'))/1001))
	edf= int(math.ceil(eval(utterance.getAttribute('END_FRAME'))/1001))
	ut['utterance_id']=uid
	ut['start_frame']=stf
	ut['end_frame']=edf
	ut['eyebrow_motion']=[]
	
	a = utterance.getElementsByTagName('NON_MANUALS')
	video = utterance.getElementsByTagName('MEDIA-FILES')
	video = video[0].getElementsByTagName('MEDIA-FILE')[0].getAttribute('NAME')
	print(video)
	video= str(video)[:-4]
	# if not os.path.exists(video):
	# 	os.mkdir(video)
	
	
	b=a[0].getElementsByTagName('NON_MANUAL')
	for c in b:
		label=c.getElementsByTagName('LABEL')[0]
		if label.childNodes[0].data=='eye brows':
			motion=str(c.getElementsByTagName('VALUE')[0].childNodes[0].nodeValue)
			t2=int (math.ceil(eval(c.getAttribute('START_FRAME'))/1001))
			t3=int(math.ceil(eval(c.getAttribute('END_FRAME'))/1001))
			
			ONSET=c.getElementsByTagName('ONSET')
			OFFSET=c.getElementsByTagName('OFFSET')
			if len(ONSET)>0:
				if ONSET[0].hasAttribute('START_FRAME'):
					ONSET_START_FRAME=ONSET[0].getAttribute('START_FRAME')
					t1=int(math.ceil(eval(ONSET_START_FRAME)/1001))
			else:
				t1=None
			if len(OFFSET)>0:
				if OFFSET[0].hasAttribute('END_FRAME'):
					OFFSET_END_FRAME=OFFSET[0].getAttribute('END_FRAME')
					t4=int(math.ceil(eval(OFFSET_END_FRAME)/1001))
			else:
				t4=None
				
			ut['eyebrow_motion'].append(str((motion,t1,t2,t3,t4)))
			
	l[video].append(ut)
			
height={'further raised':2,
		'raised':1,
		'slightly raised':0.8,
		'left raised/right furrowed':0.2,
		'right raised/left furrowed':0.2,
		'left raised/right lowered':-0.2,
		'right raised/left lowered':-0.2,
		'raised-furrowed':0.5,
		'slightly lowered':-0.8,
		'lowered':-1,
		'further lowered':-2,
	   }


for v in videos:
	for dicts in l[v]:
		label={}
		uid=dicts['utterance_id']
		s=dicts['start_frame']
		e=dicts['end_frame']
		#f_id=[i for i in range(s,e)]
		eb_v=[0 for i in range(s,e)]
		for motion in dicts['eyebrow_motion']:
			action=eval(motion)[0]
			t1=eval(motion)[1]
			t2=eval(motion)[2]
			t3=eval(motion)[3]
			t4=eval(motion)[4]
			#h = height[action]
			#eb_v[t2-s:t3-s]=np.linspace(h,h,t3-t2)
			#if t1 != None:
			 #   eb_v[t1-s:t2-s]=np.linspace(0,h,t2-t1)
			#if t4 != None:
			 #   eb_v[t3-s:t4-s]=np.linspace(h,0,t4-t3)
				
			label[str((t1,t2,t3,t4))]=action
		#f_id=(np.array([f_id]))
		#eb_v=(np.array([eb_v]))
			
			
		
		
		
		
		file= 'cam2'.join(str(v).split('cam1'))+'_'+str(s)+'to'+str(e)+'.txt'
		print(file)
		with open('label/'+file, 'w') as outfile:
			json.dump(label, outfile)
		
			

