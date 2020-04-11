from dataset import EyeBrowDataset
from resnet_lstm import resnet_lstm
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
import scipy.io as scio

parser = argparse.ArgumentParser()

parser.add_argument("-b", "--batch_size", type=int, default=1)  # 16
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
parser.add_argument('--outf', default='/gpu2/yc984/ASL/eyebrowheight/model/lstm_resnet',
                    help='folder to output images and model checkpoints')
parser.add_argument('--modelf', default='./output_ASL', help='folder to output images and model checkpoints')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('-w', '--window_size', default=7, type=int,
                    help='window_size (default: 7)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')  # 1000

parser.add_argument('--patiences', default=10, type=int,
                    help='number of epochs to tolerate the no improvement of val_loss')  # 1000



def load_model(net, path):
    state_dict = torch.load('%s' %path)
    own_state = net.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print('not load weights %s' % name)
            continue
        own_state[name].copy_(param)
        print('load weights %s' % name)


def main():
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    print(1)
    args = parser.parse_args()

    torch.manual_seed(666)
   

 

    D_xs = resnet_lstm(args.window_size)

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)
    # x = torch.FloatTensor(args.batch_size,3,args.window_size,256,256)

    #args.cuda = True
    #if args.cuda:
        #print("Let's use", torch.cuda.device_count(), "GPUs!")
        #D_xs = torch.nn.DataParallel(D_xs).cuda()
        # D_xs = D_xs.cuda()

    # x =x.cuda()
    # x = Variable(x)
    # D_xs.to(device)

    print(2)

    lr = args.learning_rate
    # lr_s = args.learning_rate_s

    #D_xs_solver = optim.Adam(D_xs.parameters(), lr=lr, )

    # l1Loss = nn.L1Loss().cuda()
    e_shift = 0
    min_val_loss = 99999
    no_improve_epoch = 0
    now = datetime.now()
    print('begin')



    #kwargs = {'map_location':lambda storage, loc:storage.cuda(gpu_id)}
    # def load_gpus(model,model_path,kwargs):
    #     state_dict = torch.load(model_path,**kwargs)
    # # create new OrderedDict that does not contain `module.`
    #     from collections import OrderedDict
    #     new_state_dict = OrderedDict()
    #     for k, v in state_dict.items():
    #         name = k[7:] # remove `module.`
    #         new_state_dict[name] = v
    # # load params
    #     model.load_state_dict(new_state_dict)
    #     return model

    # load_gpus(D_xs,'epoch_3_val_loss_0.06.pth',kwargs)
    #D_xs.load_state_dict('epoch_3_val_loss_0.06.pth')

    

    #device = torch.device('cpu')
   
  
    #D_xs=D_xs.cuda()
    D_xs = torch.nn.DataParallel(D_xs).cuda()
    #D_xs.load_state_dict(torch.load('epoch_3_val_loss_0.06.pth',map_location=device))
    state_dict =torch.load('epoch_296_train_loss_0.189.pth')
    D_xs.load_state_dict(state_dict)



    #D_xs=D_xs.cpu()
    D_xs.eval()


    print(3)


    with torch.no_grad():
        files = os.listdir('dataset/train')
        for file in files:
            train_data =  EyeBrowDataset('dataset/train/'+file, '',
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
            for i, (data, value,imgpath) in enumerate(train_loader):
                if len(data)<args.batch_size:
                    continue
                x = data
                # vv = value.type(torch.FloatTensor)
                # vv = vv.cuda()
                # vv = Variable(vv, requires_grad=False)

                score = D_xs(x)
                
                # print(value.size())
                
                
                # print(score.data.numpy())


                result.append(score.item())
                gt.append(value.item())
                imgpaths.append(imgpath[0])
                #print('i:',i)

            scio.savemat('result/'+file[:-4],{'gt':gt,'predict':result,'imgpath':imgpaths})
            print('finished:',file)

            


main()








