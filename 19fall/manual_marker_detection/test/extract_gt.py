import os
import csv
import numpy as np

csv_files = os.listdir("./csv")
for file in csv_files:
	file_name = file.split('.')[0]
	utt_start = eval(file.split('_')[-1].split('to')[0])
	utt_end = eval(file.split('_')[-1].split('to')[1][:-4])
	utt_length = utt_end-utt_start+1
	with open("./csv/"+file, encoding="utf8") as csvfile:
		gt = np.zeros(utt_length)
		utterance_data = csv.reader(csvfile, delimiter=',')
		next(utterance_data)
		for row in utterance_data:
			start = int(float(row[15]))
			end = int(float(row[16]))
			for i in range(start, end+1):
				gt[i-utt_start] = 1
		np.save('./gt/'+file_name, gt)