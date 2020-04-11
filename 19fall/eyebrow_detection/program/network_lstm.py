import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
	def __init__(self,window_size):
		super(Net,self).__init__()
		self.conv = nn.Conv3d(3,64,(1,3,3),stride=(1,1,1),padding=(0,1,1))
		self.resBlock0 = residualBlock_down(64, 64) 
		self.resBlock1 = residualBlock_down(64, 128)
		self.resBlock2 = residualBlock_down(128, 128) 
		self.resBlock3 = residualBlock_down(128, 256) 
		self.resBlock4 = residualBlock_down(256, 256)
		self.resBlock5 = residualBlock_down(256,128)
		self.bilstm1 = BiRNN(128*4*4, 64, 2, 32) 
		
		self.window_size = window_size
		self.fc_s = nn.Linear(32, 16)


	def num_flat_features(self,x):
		size = x.size()[1:]
		num_feature = 1
		for s in size:
			num_feature*=s
		return num_feature

	def forward(self,x):
		x = self.conv(x)
		
		x = self.resBlock0(x)
		x = self.resBlock1(x) 
		x = self.resBlock2(x)
		x = self.resBlock3(x)
		x = self.resBlock4(x)
		x = self.resBlock5(x)
		
		x = x.view(-1, self.window_size, 128*4*4)

		

		a_fea = self.bilstm1(x)

	
		
		
	
		attn = torch.sigmoid(self.fc_s(a_fea))
		
		s = torch.sum(attn, 1)
		return s


class conv_mean_pool(nn.Module):
	def __init__(self,inplanes,outplanes):
		super(conv_mean_pool,self).__init__()
		self.conv = nn.Conv3d(inplanes,outplanes,(1,3,3),stride=(1,1,1),padding =(0,1,1))
		self.pooling = nn.AvgPool3d((1,2,2))

	def forward(self,x):
		out = x
		out = self.conv(out)
		out = self.pooling(out)

		return out

class mean_pool_conv(nn.Module):
	def __init__(self,inplanes,outplanes):
		super(mean_pool_conv,self).__init__()
		self.conv = nn.Conv3d(inplanes,outplanes,(1,3,3),stride=(1,1,1),padding = (0,1,1))
		self.pooling = nn.AvgPool3d((1,2,2))

	def forward(self,x):
		out = x
		out = self.pooling(out)
		out = self.conv(out)
		return out

class residualBlock_down(nn.Module):
	def __init__(self,inplanes,outplanes):
		super(residualBlock_down,self).__init__()
		self.conv_shortcut =mean_pool_conv(inplanes,outplanes)
		self.conv1 = nn.Conv3d(inplanes,outplanes,(1,3,3),stride =(1,1,1),padding = (0,1,1))
		self.conv2 = conv_mean_pool(outplanes,outplanes)
		self.ReLU = nn.ReLU()

	def forward(self,x):
		shortcut = self.conv_shortcut(x)

		out = x
		out = self.ReLU(out)
		out = self.conv1(out)
		out = self.ReLU(out)
		out = self.conv2(out)

		return shortcut+out

class BiRNN(nn.Module):
	def __init__(self,input_size,hidden_size,num_layers,output_size):
		super(BiRNN,self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,bidirectional = True)
		self.fc = nn.Linear(hidden_size*2,output_size)

	def forward(self,x):
		out, _ = self.lstm(x)
		out = self.fc(out[:,-1,:])
		return out


