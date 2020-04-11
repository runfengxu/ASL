from dataset import SignDataset
from resnet_lstm2 import resnet_lstm
import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from datetime import datetime
from loger import Logger
import os
from sklearn.metrics import f1_score
from OF_dataset import OpticalF_dataset
from OF_dataset import TwoStream_dataset
from OF_C3D2 import TwoStream_Fusion2

parser = argparse.ArgumentParser()

parser.add_argument("-b", "--batch_size", type=int, default=10)  # 16
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
# parser.add_argument("-lr_s", "--learning_rate_s", type=float, default=1e-5)
parser.add_argument("-m", "--momentum", type=float, default=0.5)  # 0.5
parser.add_argument("-m2", "--momentum2", type=float, default=0.9)
parser.add_argument("-gm_x", "--gamma_vx", type=float, default=0.5)
parser.add_argument("-gm_xx", "--gamma_x", type=float, default=0.5)
parser.add_argument("-lm", "--lamda", type=float, default=0.001)
parser.add_argument("-fn", "--filter_number", type=int, default=16)
parser.add_argument("-z", "--input_size", type=int, default=16)
parser.add_argument("-em", "--embedding", type=int, default=16)
parser.add_argument('--outf', default='../model',
					help='folder to output images and model checkpoints')
parser.add_argument('--modelf', default='./output_ASL', help='folder to output images and model checkpoints')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
					help='number of data loading workers (default: 8)')
parser.add_argument('-w', '--window_size', default=7, type=int,
					help='window_size (default: 7)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
					help='number of total epochs to run')  # 1000

parser.add_argument('--patiences', default=10, type=int,
					help='number of epochs to tolerate the no improvement of val_loss')  # 1000


def weights_init(m):


	classname = m.__class__.__name__

	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0,0.02)

	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0,0.02)
		m.bias.data.fill_(0)
	elif classname.find('LayerNorm') != -1:
		m.weight.data.normal_(1.0,0.02)
		m.bias.data.fill_(0)
	elif classname.find('Linear') != -1:
		m.weight.data.normal_(0.0,0.02)
		m.bias.data.fill_(0)

def load_model(net, path, name):
	state_dict = torch.load('%s/%s' % (path, name))
	own_state = net.state_dict()
	for name, param in state_dict.items():
		if name not in own_state:
			print('not load weights %s' % name)
			continue
		own_state[name].copy_(param)
		print('load weights %s' % name)



def main():
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ['CUDA_VISIBLE_DEVICES']='1'
	# print(0)
	args  =parser.parse_args()
	torch.manual_seed(666)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	
	#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	#print(1)
	#torch.cuda.set_device(0)


	#print(2)

	train_data = TwoStream_dataset('../dataset/train.mat','../frame/','../optical_flow_matrix/',
										args.window_size,
										transforms.Compose([
										transforms.ToTensor(),
										transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
										]))
	vali_data = TwoStream_dataset('../dataset/validation.mat','../frame/','../optical_flow_matrix/',
										args.window_size,
										transforms.Compose([
										transforms.ToTensor(),
										transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
										]))
	train_loader = torch.utils.data.DataLoader(train_data,batch_size = args.batch_size,shuffle = True,num_workers = args.workers,pin_memory = True)
	vali_loader = torch.utils.data.DataLoader(vali_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
	D_xs = TwoStream_Fusion2()
	D_xs.apply(weights_init)
	args.cuda = True
	if args.cuda:
		print('lets use',torch.cuda.device_count(),"gpus")
		D_xs = torch.nn.DataParallel(D_xs).cuda()

	lr = args.learning_rate
	D_xs_solver = optim.Adam(D_xs.parameters(),lr = lr)
	BCE = nn.BCELoss().cuda()
	e_shift =0 
	min_val_loss = 99999
	max_accuracy = 0
	no_improve_epoch = 0
	now = datetime.now()
#log
	log_path = '../log/lr_{}_time_{}'.format(args.learning_rate,now.strftime("%Y%m%d-%H%M%S"))
	try:
		os.mkdir(log_path)
	except:
		pass
	#writer = SummaryWriter(log_path)
	logger = Logger(log_path)
#train
	mes_sum=0
	n_iter = 0
	for epoch in range(args.epochs):
		print(1)

		D_xs.train()
		print('len of train_loader',len(train_loader))
		train_epoch_score = 0
		train_predictions = []
		train_gt = []
		for i,(data1,v1) in enumerate(train_loader):
			if len(data1[0])<args.batch_size:
				continue
			n_iter+=2
			D_xs.zero_grad()
			x_1 =data1[0]
			x_2 = data1[1]
			x_1 = x_1.cuda()
			x_2 = x_2.cuda()

			# x_2 = data2
			# x_2 = x_2.to(device)
			vv_1 = v1.type(torch.FloatTensor)
			vv_1= vv_1.cuda()
			vv_1 = Variable(vv_1,requires_grad =False)
			# vv_2 = v2.type(torch.FloatTensor)
			# vv_2= vv_2.cuda()
			# vv_2 = Variable(vv_2,requires_grad =False)
			
			
			
			# score_2 = D_xs(x_2)
			# v_loss=L1Loss(score_2,vv_2)
			# v_loss.backward()
			# D_xs_solver.step()
			# mes_sum+=v_loss.item()
			# print('nega:',v_loss.item())
			score_1 = D_xs(x_1,x_2)
			
			v_loss=BCE(score_1,vv_1)
			v_loss.backward()
			D_xs_solver.step()
			mes_sum+=v_loss.item()

			for item in score_1.tolist():
					if item > 0.5:
						train_predictions.append(1)
					else:
						train_predictions.append(0)

			train_gt += vv_1.tolist()

			# pre = round(list(score_1))
			# print(pre)
			# print(vv_1)
			# print(score_1)
			# print('loss:',v_loss.item())
			if i%3 ==0:
				#print(vv_1)
				#print(score_1)

				#writer.add_scalar('train/loss',mes_sum/10,n_iter)
				info = { 'loss': v_loss.item()}

				for tag, value in info.items():
					logger.scalar_summary(tag, value, (i+epoch*1000))
				# for tag, value in D_xs.named_parameters():
				# 	tag = tag.replace('.','/')
				# 	logger.histo_summary(tag, value.data.cpu().numpy(), i)
				# 	logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), i)
				print('epoch:[%2d] [%4d/%4d] loss: %.4f'  % (epoch + e_shift, i, (len(train_data)/args.batch_size), mes_sum/10))
				mes_sum=0
		
		train_epoch_score = sum(i == j for i, j in zip(train_predictions, train_gt))		
		train_accuracy = train_epoch_score/(len(train_loader)*args.batch_size)
		train_f1_score = f1_score(train_gt, train_predictions)
		print("Train accuracy for epoch:[%2d]: %.4f" %(epoch, train_accuracy))
		print("Train F1 score for epoch:[%2d]: %.4f" %(epoch, train_f1_score))
		info2 = {'train_accuracy' : train_accuracy, "train_f1_score" : train_f1_score}
		for tag, value in info2.items():
			logger.scalar_summary(tag, value, epoch)

		mse_sum = 0
		D_xs.eval()
		
		vali_epoch_score = 0
		vali_predictions = []
		vali_gt = []
		with torch.no_grad():
			for  i,(data1,v1) in enumerate(vali_loader):
				x_1 = data1[0]
				x_2 = data1[1]

				x_1 = x_1.cuda()
				x_2 = x_2.cuda()

				
				vv_1 = v1.type(torch.FloatTensor)
				vv_1= vv_1.cuda()
				vv_1 = Variable(vv_1,requires_grad =False)
				
				
				score=D_xs(x_1,x_2)
				v_loss =BCE(score,vv_1)
				mse_sum = mse_sum+v_loss.item()

				for item in score.tolist():
					if item > 0.5:
						vali_predictions.append(1)
					else:
						vali_predictions.append(0)

				vali_gt += vv_1.tolist()

			val_loss = mse_sum/float(i+1)
			print('epoch:[%2d] ,val_mse :%.6f  '%(epoch+e_shift,val_loss))
			vali_epoch_score = sum(i == j for i, j in zip(vali_predictions, vali_gt))
			vali_accuracy = vali_epoch_score/(len(vali_loader)*args.batch_size)
			vali_f1_score = f1_score(vali_gt, vali_predictions)
			#writer.add_scalar('Test/Loss',val_loss,n_iter)
			print("validation accuracy for epoch:[%2d]: %.4f" %(epoch, vali_accuracy))
			print("validation F1 score for epoch:[%2d]: %.4f" %(epoch, vali_f1_score))
			info3 = { 'vali_loss': mse_sum, 'vali_accuracy': vali_accuracy, 'vali_f1_score' : vali_f1_score}

			for tag, value in info3.items():
				logger.scalar_summary(tag, value, epoch)
			if max_accuracy < vali_accuracy:
				max_accuracy = vali_accuracy
				no_improve_epoch = 0
				max_accuracy = round(max_accuracy,4)
				torch.save(D_xs.state_dict(),'{}/epoch_{}_max_accuracy_{}.pth'.format(args.outf, epoch + e_shift, max_accuracy))
				print("performance improve, saved the new model......")
			else:
				no_improve_epoch+=1
			if no_improve_epoch>args.patiences:
				print('stop training')
				break


main()