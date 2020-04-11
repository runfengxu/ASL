from torchvision import models
import torch.nn as nn
import torch

class my_resnet(nn.Module):
    def __init__(self):
        super(my_resnet, self).__init__()
        resnet = models.resnet50(norm_layer = nn.BatchNorm2d)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

    def forward(self, x):
        x = self.resnet(x)
        return x


class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiRNN, self).__init__()
        self.input_size=input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)

        self.fc = nn.Linear(hidden_size , 1)  # 2 for bich.abs(x-y), 1)direction

    def forward(self, x):
        # Set initial states
        # h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)).cuda() # 2 for bidirection
        # c0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)).cuda() # 2 for bidirection
        # Forward propagate LSTM
        # out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        out, _ = self.lstm(x)
        #print('out:',out.shape)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :]).squeeze()
        return out

class resnet_lstm(nn.Module):
    def __init__(self,window_size):
        super(resnet_lstm, self).__init__()
        self.resnet = my_resnet()
        self.bilstm = BiRNN(2048, 256, 2)
        self.window_size = window_size

    def forward(self, x):
        # batch t chanel h w
        #print(x.shape)
        x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
        #print(x.shape)
        x= self.resnet(x)
        x = x.view(-1, self.window_size, x.shape[-3], x.shape[-2], x.shape[-1]).squeeze()
        #print(x.shape)
         #batch, t, ft_dim
        #x.reshape([1,self.window_size,512])
        #x = x.squeeze()
        #x = x.view(-1, self.window_size, 256*4*4) #5*4096  # 5 to 7, 9, or 11
        #s = self.fc_s(x)
        x = self.bilstm(x) #
        #print(x.shape)
        x = torch.sigmoid(x)
        print(x.shape)
        return x

class C3D(nn.Module):
    def __init__(self):
        super(C3D,self).__init__()
        self.conv1=nn.Conv3d(3, 64, kernel_size=(2, 3, 3), stride=1)
        self.relu=nn.LeakyReLU()
        self.maxpool1 = nn.MaxPool3d((1,2,2))
        self.conv2=nn.Conv3d(64, 128, kernel_size=(2, 3, 3), stride=1)
        self.maxpool2 = nn.MaxPool3d((2,2,2))
        self.conv3=nn.Conv3d(128, 256, kernel_size=(2, 3, 3), stride=1)
        self.maxpool3 = nn.MaxPool3d((2,2,2))
        self.conv4=nn.Conv3d(256, 256, kernel_size=(1, 3, 3), stride=1)
        self.fc5 = nn.Linear(147456, 256)
        self.batch0=nn.BatchNorm1d(256)
        self.drop=nn.Dropout(p=0.15)
        self.fc6 = nn.Linear(256, 256)
        self.batch1=nn.BatchNorm1d(256)
        self.fc7 = nn.Linear(256, 1)
        self.drop=nn.Dropout(p=0.15)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[2], x.shape[1], x.shape[-2], x.shape[-1])
        x = self.maxpool1(self.relu(self.conv1(x)))    #shape(10, 64, 10, 111, 111)
        #print(x.shape)
        x = self.maxpool2(self.relu(self.conv2(x)))    #shape(10, 128, 4, 54, 54)
        #print(x.shape)
        x = self.maxpool3(self.relu(self.conv3(x)))    #shape(10, 256, 1, 26, 26)
        #print(x.shape)
        x = self.conv4(x)                              #shape(10, 256, 1, 24, 24)
        #print(x.shape)
        x = x.view(x.size(0), -1)                      #shape(10, 147456)
        #print(x.shape)
        x = self.batch0(self.relu(self.fc5(x)))        #SHAPE()
        x = self.drop(x)
        #print(x.shape)
        x = self.batch1(self.relu(self.fc6(x)))
        x = self.drop(x)
        x = self.fc7(x)
        x = x.squeeze()
        x = torch.sigmoid(x)
        
        return 

class C3D2(nn.Module):
    def __init__(self):
        super(C3D2,self).__init__()
        self.conv1=nn.Conv3d(3, 64, kernel_size=(3, 3, 3), stride=1,padding=(1,0,0))
        self.relu=nn.LeakyReLU()
        self.maxpool1 = nn.MaxPool3d((1,2,2))

        self.batchnorm1 = nn.BatchNorm3d(64)
        
        self.conv2=nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1,padding=(1,0,0))
        self.maxpool2 = nn.MaxPool3d((2,2,2))
        
        self.batchnorm2 = nn.BatchNorm3d(128)
        
        self.conv3a=nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=1,padding=(1,0,0))
        self.conv3b=nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=1,padding=(1,0,0))
        
        self.batchnorm3 = nn.BatchNorm3d(256)
        
        self.conv4a=nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=1,padding=(1,0,0))
        self.conv4b=nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=1,padding=(1,0,0))
        self.maxpool3 = nn.MaxPool3d((2,3,3))
        self.batchnorm4 = nn.BatchNorm3d(512)
        
        self.conv5a=nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=1,padding=(1,0,0))
        self.conv5b=nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=1,padding=(1,0,0))

        
        self.fc5 = nn.Linear(25088, 256)

        self.batch0=nn.BatchNorm1d((256))
        
        self.drop=nn.Dropout(p=0.15)
        
        self.fc6 = nn.Linear(256, 256)
        
        self.batch1=nn.BatchNorm1d(256)
        self.fc7 = nn.Linear(256, 1)
        self.drop=nn.Dropout(p=0.15)

    def forward(self, x):
        #shape(10,11,3,224,224)   bacth,window,channel,h,w
        x = x.view(x.shape[0], x.shape[2], x.shape[1], x.shape[-2], x.shape[-1])
       
        x = self.maxpool1(self.relu(self.batchnorm1(self.conv1(x))))    #shape(10, 64, 10, 111, 111)
       
        x = self.maxpool2(self.relu(self.batchnorm2(self.conv2(x))))    #shape(10, 128, 4, 54, 54)
       
        x = self.maxpool2(self.conv3b(self.relu(self.batchnorm3(self.conv3a(x)))))   #shape(10, 256, 1, 26, 26)
        
        x = self.maxpool3(self.conv4b(self.relu(self.batchnorm4(self.conv4a(x))))) 
        #x = self.maxpool2(self.conv5b(self.relu(self.conv5a(x)))) 
       
        #shape(batchsize, 512, 1, 4, 4)
        #print(x.shape)
        x = x.view(x.size(0), -1)                      #shape(10, 147456)
       
        x = self.batch0(self.relu(self.fc5(x)))        #SHAPE()
        x = self.drop(x)
        #print(x.shape)
        x = self.batch1(self.relu(self.fc6(x)))
        #x = self.drop(x)
        x = self.fc7(x)
        x = x.squeeze()
        x = torch.sigmoid(x)
        
        return x
