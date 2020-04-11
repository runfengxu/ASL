from dataset import EyeBrowDataset
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


parser = argparse.ArgumentParser()


parser.add_argument("-b", "--batch_size", type=int, default=256) # 16
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
#parser.add_argument("-lr_s", "--learning_rate_s", type=float, default=1e-5)
parser.add_argument("-m" , "--momentum", type=float, default=0.5) # 0.5
parser.add_argument("-m2", "--momentum2", type=float, default=0.9) 
parser.add_argument("-gm_x", "--gamma_vx", type=float, default=0.5) 
parser.add_argument("-gm_xx", "--gamma_x", type=float, default=0.5) 
parser.add_argument("-lm", "--lamda", type=float, default=0.001)
parser.add_argument("-fn", "--filter_number", type=int, default=16) 
parser.add_argument("-z",  "--input_size", type=int, default=16)
parser.add_argument("-em", "--embedding", type=int, default=16)
parser.add_argument('--outf', default='/gpu2/yc984/ASL/eyebrowheight/model/cnn', help='folder to output images and model checkpoints')
parser.add_argument('--modelf', default='./output_ASL', help='folder to output images and model checkpoints')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
					help='number of data loading workers (default: 8)')
parser.add_argument('-w', '--window_size', default=1, type=int,
					help='window_size (default: 7)')
parser.add_argument('--epochs', default=15, type=int, metavar='N',
					help='number of total epochs to run') # 1000

parser.add_argument('--patiences', default=10, type=int,
					help='number of epochs to tolerate the no improvement of val_loss') # 1000



# def weights_init(m):
#
#
# 	classname = m.__class__.__name__
#
# 	if classname.find('Conv') != -1:
# 		m.weight.data.normal_(0.0,0.02)
#
# 	elif classname.find('BatchNorm') != -1:
# 		m.weight.data.normal_(1.0,0.02)
# 		m.bias.data.fill_(0)
# 	elif classname.find('LayerNorm') != -1:
# 		m.weight.data.normal_(1.0,0.02)
# 		m.bias.data.fill_(0)
# 	elif classname.find('Linear') != -1:
# 		m.weight.data.normal_(0.0,0.02)
# 		m.bias.data.fill_(0)

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

	print(1)
	args = parser.parse_args()
	
	torch.manual_seed(666)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	train_loader = torch.utils.data.DataLoader(
		EyeBrowDataset('/dresden/users/rx43/ASL/eyebrow_detecting/dataset/train_all.mat',
					   '/dresden/users/rx43/ASL/eyebrow_detecting/', args.window_size,
					   transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
					   ])), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)



	vali_loader = torch.utils.data.DataLoader(
		EyeBrowDataset('/dresden/users/rx43/ASL/eyebrow_detecting/dataset/validation_all.mat',
					   '/dresden/users/rx43/ASL/eyebrow_detecting/', args.window_size,
					   transform=transforms.Compose([transforms.ToTensor(),
													 transforms.Normalize([0.485, 0.456, 0.406],
																		  [0.229, 0.224,
																		   0.225])])), batch_size=args.batch_size,
		shuffle=True, num_workers=args.workers, pin_memory=True)

	#D_xs = Net(args.window_size)

	D_xs = models.resnet50()
	num_ftrs = D_xs.fc.in_features
	D_xs.fc = nn.Linear(num_ftrs, 1)

	
	#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	#torch.cuda.set_device(0)
	#x = torch.FloatTensor(args.batch_size,3,args.window_size,256,256)
	
	args.cuda = True
	if args.cuda:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		D_xs = torch.nn.DataParallel(D_xs).cuda()
		#D_xs = D_xs.cuda()
	
		#x =x.cuda()
	#x = Variable(x)
	#D_xs.to(device)

	print(2)
	
	lr = args.learning_rate
	#lr_s = args.learning_rate_s
	

	D_xs_solver = optim.Adam(D_xs.parameters(), lr = lr,)

	l1Loss = nn.L1Loss().cuda()
	e_shift = 0
	min_val_loss = 99999
	no_improve_epoch = 0
	now = datetime.now()
	print('begin')

	log_path = 'log/lr_{}_time:{}'.format(args.learning_rate, now.strftime("%Y%m%d-%H%M%S"))
	# try:
	# 	os.mkdir(log_path)
	# except:
	# 	pass
	writer = SummaryWriter(log_path)

	mes_sum = 0
	for epoch in range(args.epochs):
		
		print('len of train_loader',len(train_loader))
		for i, (data, value) in enumerate(train_loader):
			
			if len(data)<args.batch_size:
				continue
			D_xs.zero_grad()

			imgseq = data

			x = data
			vv = value.type(torch.FloatTensor)
			vv = vv.cuda()
			#vv = vv.to(device)
			vv = Variable(vv,requires_grad=False)

			#x= x.to(device)
			score = D_xs(x)
			#score = nn.Sigmoid(score)

			v_loss = l1Loss(score,vv)
			v_loss.backward()
			D_xs_solver.step()
			mes_sum+=v_loss.item()

			if i%10 == 0:
				# s.append(mes_sum)
				# scio.savemat('log/log_after'+str(i)+'_iteration.mat',{'loss':s})

				writer.add_scalar('train/loss',mes_sum / 10,i)

				mes_sum =0
			#print(score,vv)



				print('epoch:[%2d] [%4d/%4d] loss: %.4f' % (epoch+e_shift,i,len(data),v_loss.item()))
			



		print('do evaluation........')
	

		print('len of train_loader',len(vali_loader))
		mse_sum = 0
		with torch.no_grad():
			for i, (data, value) in enumerate(vali_loader):


				x = data
				vv = value.type(torch.FloatTensor)
				vv = vv.cuda()
				vv = Variable(vv,requires_grad=False)

				score = D_xs(x)

				v_loss = l1Loss(score,vv)

				mse_sum = mse_sum + v_loss.item()



				#print('epoch:[%2d] [%4d/%4d] loss: %.4f' % (epoch+e_shift,i,len(data),v_loss.item()))

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
				torch.save(D_xs.state_dict(), '{}/epoch_{}_val_loss_{}.pth'.format(args.outf, epoch+e_shift,val_loss))
				print("performance improve, saved the new model......")
			else:
				no_improve_epoch += 1

			if no_improve_epoch > args.patiences:
				print("stop training....")
				break



main()








