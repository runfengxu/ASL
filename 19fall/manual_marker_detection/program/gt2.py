from xml.dom.minidom import parse
import xml.dom.minidom
import json
import numpy as np
import scipy.io as scio
import os
import math
from os import path
import pandas as pd

	


def get_node_value(element,name):
	if len(element.getElementsByTagName(name)[0].childNodes)>0:
		return element.getElementsByTagName(name)[0].childNodes[0].nodeValue
	else:
		return None



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
	
	a = utterance.getElementsByTagName('MANUALS')
	video = utterance.getElementsByTagName('MEDIA-FILES')
	video = video[0].getElementsByTagName('MEDIA-FILE')[0].getAttribute('NAME')
	video= str(video)[:-4]
	# if not os.path.exists(video):
	# 	os.mkdir(video)
	
	
	b=a[0].getElementsByTagName('SIGN')
	for c in b:
		ID = c.getAttribute('ID')
		LABEL=get_node_value(c,'LABEL')
		SIGN_TYPE = get_node_value(c,'SIGN_TYPE')
		TWO_HANDED = get_node_value(c,'TWO_HANDED')
		PASSIVE_BASE_ARM =  get_node_value(c,'PASSIVE_BASE_ARM')
		MARKED_HANDS = get_node_value(c,'MARKED_HANDS')
		START_END_HS = get_node_value(c,'START_END_HS')
		START_END_LEFT_HS =get_node_value(c,'START_END_LEFT_HS')
		START_END_RIGHT_HS = get_node_value(c,'START_END_RIGHT_HS')
		D_START_HS = get_node_value(c,'D_START_HS')
		ND_START_HS = get_node_value(c,'ND_START_HS')
		D_END_HS = get_node_value(c,'D_END_HS')
		ND_END_HS = get_node_value(c,'ND_END_HS')
		TWOHANDED_HANDSHAPES = get_node_value(c,'TWOHANDED_HANDSHAPES')
		DOMINANT_HAND_START_FRAME=c.getElementsByTagName('DOMINANT_HAND')[0].getAttribute('START_FRAME')
		DOMINANT_HAND_END_FRAME=c.getElementsByTagName('DOMINANT_HAND')[0].getAttribute('END_FRAME')




		l[video].append([ID,LABEL,SIGN_TYPE,TWO_HANDED,PASSIVE_BASE_ARM,MARKED_HANDS,
			START_END_HS,START_END_LEFT_HS,START_END_RIGHT_HS,D_START_HS,ND_START_HS,
			D_END_HS ,ND_END_HS,TWOHANDED_HANDSHAPES,DOMINANT_HAND_START_FRAME,DOMINANT_HAND_END_FRAME])
	#print('finish utterance:'+uid)
	
#print(l)


for v in videos:
	df = pd.DataFrame(l[v],columns  =['ID','LABEL','SIGN_TYPE','TWO_HANDED','PASSIVE_BASE_ARM','MARKED_HANDS','START_END_HS','START_END_LEFT_HS','START_END_RIGHT_HS','D_START_HS','ND_START_HS','D_END_HS ','ND_END_HS','TWOHANDED_HANDSHAPES','DOMINANT_HAND_START_FRAME','DOMINANT_HAND_END_FRAME']) 
	
	df.to_csv(r'sign/'+v+'.csv')