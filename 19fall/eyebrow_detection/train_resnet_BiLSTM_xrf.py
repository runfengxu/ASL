from new_dataset import EyeBrowDataset
from resnet_BiLSTM import resnet_lstm
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
import os

parser = argparse.ArgumentParser()

parser.add_argument("-b", "--batch_size", type=int, default=8)  # 16
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)

parser.add_argument('--cuda', action='store_true', help='enables cuda')

parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')

parser.add_argument('-w', '--window_size', default=7, type=int,
                    help='window_size (default: 7)')

parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')  # 1000

parser.add_argument('--patiences', default=10, type=int,
                    help='number of epochs to tolerate the no improvement of val_loss')  # 1000


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
    state_dict = torch.load('%s/%s' % (path, name))
    own_state = net.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print('not load weights %s' % name)
            continue
        own_state[name].copy_(param)
        print('load weights %s' % name)


def main():
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    torch.manual_seed(666)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    upper_face_fold = ''
    #whole_face_fold = '/dresden/users/rx43/ASL/eyebrow_detecting/'

    img_fold = upper_face_fold

    # model = Net(args.window_size)

    model = resnet_lstm(args.window_size)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)
    # x = torch.FloatTensor(args.batch_size,3,args.window_size,256,256)

    args.cuda = True
    if args.cuda:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model).cuda()
    # model = model.cuda()

    # x =x.cuda()
    # x = Variable(x)
    # model.to(device)
    state_dict = torch.load('epoch_97_val_loss_0.01.pth')
    model.load_state_dict(state_dict)



    lr = args.learning_rate


   
    e_shift = 0

  

    mes_sum = 0
    n_iter = 0
 
    model.eval()

    with torch.no_grad():
        files = os.listdir('dataset/test')
        for file in files:
            test_data =  EyeBrowDataset('dataset/test/'+file, '',
                       args.window_size,
                       transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                       ]))
       
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
            #print(4)
            result =[]
            gt=[]
            imgpaths=[]
            for i, (data, height,imgpath) in enumerate(test_loader):
                if len(data)<args.batch_size:
                    continue
                
                
                score = model(data)

                result.append(score.item())
                gt.append(value.iten())
                imgpaths.append(imgpath[0])
            scio.savemat('result/'+file[:-4],{'gt':gt,'predict':result,'imgpath':imgpaths})
            print('finished:',file)


main()








