from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import torch
import numpy as np
from OF_C3D2 import TwoStream_Fusion2
from test_dataset import Test_dataset
import os
from torch.autograd import Variable

def main():
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ['CUDA_VISIBLE_DEVICES']='1'
	torch.manual_seed(666)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	D_xs = TwoStream_Fusion2()
	D_xs.load_state_dict(torch.load("C:/Users/cbim v left/Desktop/manual/new_log/epoch_28_val_loss_0.32.pth"),strict=False)
	matfiles = os.listdir("./dataset/")
	# matfiles = ["2011-12-01_0043-cam1-for-ss3_17587203_5382to5466.mat"]
	cuda = True
	if cuda:
		print('lets use',torch.cuda.device_count(),"gpus")
		D_xs = torch.nn.DataParallel(D_xs).cuda()
	for matfile in matfiles:
		print(matfile)
		start_frame = eval(matfile.split('_')[-1].split('to')[0])
		end_frame = eval(matfile.split('_')[-1].split('to')[1][:-4])
		utt_length = end_frame-start_frame+1
		predicts = np.zeros(utt_length)
		count = np.zeros(utt_length)

		test_data = Test_dataset('./dataset/'+matfile,'../frame/','../optical_flow_matrix/',
											11,
											transforms.Compose([
											transforms.ToTensor(),
											transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
											]))
		test_loader = torch.utils.data.DataLoader(test_data, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
		with torch.no_grad():
			for  i,(data1, interval) in enumerate(test_loader):
				if len(interval[0])<2:
					continue
				x_1 = data1[0]
				x_2 = data1[1]
				x_1 = x_1.cuda()
				x_2 = x_2.cuda()
				score=D_xs(x_1,x_2)
				score = score.tolist()

				start1 = interval[0][0]-start_frame
				end1 = interval[1][0]-start_frame
				start2 = interval[0][1]-start_frame
				end2 = interval[1][1]-start_frame
				score1 = score[0]
				score2 = score[1]
				# start3 = interval[0][2]-start_frame
				# end3 = interval[1][2]-start_frame
				# start4 = interval[0][3]-start_frame
				# end4 = interval[1][3]-start_frame
				# score3 = score[2]
				# score4 = score[3]
				for i in range(start1,end1+1):
					predicts[i] += score1
					count[i] += 1
				for j in range(start2,end2+1):
					predicts[j] += score2
					count[j] += 1
				# for k in range(start3,end3+1):
				# 	predicts[k] += score2
				# 	count[k] += 1
				# for l in range(start4,end4+1):
				# 	predicts[l] += score4
				# 	count[l] += 1
		predicts = np.divide(predicts, count, out=np.zeros_like(predicts), where=count!=0)
		results = []
		for i in predicts:
			results.append(i)
		np.save("./predicts/"+matfile[:-4], results)

main()
