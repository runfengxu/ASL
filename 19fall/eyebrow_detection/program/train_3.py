from dataset import EyeBrowDataset
from network import Net
import time
import argparse
import os
from PIL import Image
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.autograd as autograd
import scipy.io as sio
from torch.autograd import Variable
import torch.nn.functional as F
from itertools import *
import pdb
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--data_dir", type=str, default="./Data")
#parser.add_argument("-d", "--data_dir", type=str, default="./getTrainList")
parser.add_argument("-trd", "--dataset", type=str, default="celeba"  )
parser.add_argument("-tro", "--data_opt", type=str, default="crop")
parser.add_argument("-trs", "--data_size", type=int, default=8) # 64
parser.add_argument("-ns", "--nsnapshot", type=int, default=10)
parser.add_argument("-b", "--batch_size", type=int, default=16) # 16
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
parser.add_argument("-lr_s", "--learning_rate_s", type=float, default=1e-5)
parser.add_argument("-m" , "--momentum", type=float, default=0.) # 0.5
parser.add_argument("-m2", "--momentum2", type=float, default=0.9) # 0.999
parser.add_argument("-gm_x", "--gamma_vx", type=float, default=0.5) # 0.5 0.7
parser.add_argument("-gm_xx", "--gamma_x", type=float, default=0.5) # 0.5 0.3
parser.add_argument("-lm", "--lamda", type=float, default=0.001)
parser.add_argument("-fn", "--filter_number", type=int, default=16) # 64
parser.add_argument("-z",  "--input_size", type=int, default=16)
parser.add_argument("-em", "--embedding", type=int, default=16)
parser.add_argument('--outf', default='./output_ASL', help='folder to output images and model checkpoints')
parser.add_argument('--modelf', default='./output_ASL', help='folder to output images and model checkpoints')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
					help='number of data loading workers (default: 8)')
parser.add_argument('-w', '--window_size', default=7, type=int,
					help='window_size (default: 7)')
parser.add_argument('--epochs', default=5, type=int, metavar='N',
					help='number of total epochs to run') # 1000

parser.add_argument('--patiences', default=10, type=int,
					help='number of epochs to tolerate the no improvement of val_loss') # 1000



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

def L1_loss(x, y):
	return torch.mean(torch.sum(torch.abs(x-y), 1))


def load_model(net, path, name):
	state_dict = torch.load('%s/%s' % (path,name))
	own_state = net.state_dict()
	for name, param in state_dict.items():
		if name not in own_state:
			print('not load weights %s' % name)
			continue
		own_state[name].copy_(param)
		print('load weights %s' % name)
def main():
	args = parser.parse_args()
	
	D_xs = Net(7)
	D_xs.apply(weights_init)
	
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	x = torch.FloatTensor(args.batch_size,3,args.window_size,256,256)
	
	args.cuda = True
	if args.cuda:
		#print("Let's use", torch.cuda.device_count(), "GPUs!")
		D_xs = torch.nn.DataParallel(D_xs).cuda()
		#D_xs = D_xs.cuda()
	
		x =x.cuda()
	x = Variable(x)
	#D_xs.to(device)
	
	lr = args.learning_rate
	lr_s = args.learning_rate_s
	ourBetas = [args.momentum,args.momentum2]
	batch_size = args.batch_size
	snapshot = args.nsnapshot
	

	D_xs_solver = optim.Adam(D_xs.parameters(), lr = lr, betas = ourBetas)

	cudnn.benchmark = True
	l1Loss = nn.L1Loss().cuda()
	e_shift = 0
	min_val_loss = 99999
	no_improve_epoch = 0
	
	
	for epoch in range(args.epochs):

		path = 'dataset/'
		
			
		train_loader = torch.utils.data.DataLoader(
								EyeBrowDataset('dataset/train_all.mat','',args.window_size,
									transform = transforms.Compose([transforms.ToTensor(),
									transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
									), batch_size = args.batch_size, shuffle = True, num_workers = args.workers, pin_memory = True )
		
		
		print('len of train_loader',len(train_loader))
			
		for i, (data, value) in enumerate(train_loader):
			
			if len(data)<args.batch_size:
				continue

			D_xs.zero_grad()

			imgseq = data
			
			x.data.resize_(imgseq.size()).copy_(imgseq)
			vv = value.type(torch.FloatTensor)
			vv = vv.cuda()
			#vv = vv.to(device)
			vv = Variable(vv,requires_grad=False)
		
			#x= x.to(device)
			score = D_xs(x)
			
			v_loss = l1Loss(score,vv)
			v_loss.backward()
			D_xs_solver.step()

			print('epoch:[%2d] [%4d/%4d] loss: %.4f' % (epoch+e_shift,i,len(data),v_loss.item()))
			mes_sum=0




		path = 'dataset/'
	
		
		vali_loader = torch.utils.data.DataLoader(
								EyeBrowDataset('dataset/validation_all.mat','',args.window_size,
									transform = transforms.Compose([transforms.ToTensor(),
									transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
									), batch_size = args.batch_size, shuffle = True, num_workers =args.workers, pin_memory = True )
		
		
		print('len of train_loader',len(vali_loader))
			
		for i, (data, value) in enumerate(vali_loader):
			
		

			imgseq = data
			with torch.no_grad():
				x.data.resize_(imgseq.size()).copy_(imgseq)
			vv = value.type(torch.FloatTensor)
			vv = vv.cuda()
			vv = Variable(vv,requires_grad=False)
			
			score = D_xs(x)
			
			v_loss = l1Loss(score,vv)

			mse_sum = mse_sum + v_loss.item()
			
			

			print('epoch:[%2d] [%4d/%4d] loss: %.4f' % (epoch+e_shift,i,len(data),v_loss.item()))
	
		val_loss = mse_sum/float(i+1)
		print("*** Epoch: [%2d], "
				  "val_mse: %.6f ***"
				  % (epoch + e_shift, val_loss))
		
		# if performance improve save the new model
		# if performance does not increase for patiences epochs, stop training
		if val_loss < min_val_loss:
			min_val_loss = val_loss
			no_improve_epoch = 0
			val_loss = round(val_loss,2)
			torch.save(D_xs.state_dict(), '{}/netD_xs_epoch_{}_val_loss_{}.pth'.format(args.outf, epoch+e_shift,val_loss))
			print("performance improve, saved the new model......")
		else:
			no_improve_epoch += 1
		
		if no_improve_epoch > args.patiences:
			print("stop training....")
			break



main()








