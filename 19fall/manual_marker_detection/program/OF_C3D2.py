from torchvision import models
import torch.nn as nn
import torch

class C3D(nn.Module):
    def __init__(self, modality):
        super(C3D,self).__init__()
        if modality == "RGB":
            self.conv1=nn.Conv3d(3, 64, kernel_size=(3, 3, 3), stride=1,padding=(1,0,0))
        elif modality == "OF": 
            self.conv1=nn.Conv3d(2, 64, kernel_size=(3, 3, 3), stride=1,padding=(1,0,0))
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
        #shape(10,11,2,224,224)   bacth,window,channel,h,w
       
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
        # x = self.fc7(x)
        # x = x.squeeze()
        # x = torch.sigmoid(x)
        
        return x


class TwoStream_Fusion(nn.Module):
    def __init__(self):
        super(TwoStream_Fusion,self).__init__()

        self.RGB_stream = C3D("RGB")
        self.OF_stream = C3D("OF")
        self.fc6 = nn.Linear(512, 512)
        self.relu=nn.LeakyReLU()
        self.batch1=nn.BatchNorm1d(512)
        self.fc7 = nn.Linear(512, 256)
        self.batch2=nn.BatchNorm1d(256)
        self.drop=nn.Dropout(p=0.15)
        self.fc8=nn.Linear(256,1)

    def forward(self, x1,x2):
        #shape(10,11,2,224,224)   bacth,window,channel,h,w
        x1 = self.RGB_stream(x1)
        x2 = self.OF_stream(x2)
        x = torch.cat((x1,x2),-1)
        #x = self.maxpool2(self.conv3b(self.relu(self.batchnorm3(self.conv3a(x)))))   #shape(10, 256, 1, 26, 26)
        
        #x = self.maxpool3(self.conv4b(self.relu(self.batchnorm4(self.conv4a(x))))) 
        #x = self.maxpool2(self.conv5b(self.relu(self.conv5a(x)))) 
       
        #shape(batchsize, 512, 1, 4, 4)
        #print(x.shape)
        #x = x.view(x.size(0), -1)                      #shape(10, 147456)
        #x = self.batch0(self.relu(self.fc5(x)))        #SHAPE()
        #x = self.drop(x)
        #print(x.shape)
        x = self.batch1(self.relu(self.fc6(x)))
        x = self.drop(x)
        x = self.batch2(self.relu(self.fc7(x)))
        x = self.fc8(x)
        x = x.squeeze()
        x = torch.sigmoid(x)
        
        return x

class C3D2(nn.Module):
    def __init__(self, modality):
        super(C3D2,self).__init__()
        if modality == "RGB":
            self.conv1=nn.Conv3d(3, 64, kernel_size=(3, 3, 3), stride=1,padding=(1,0,0))
        elif modality == "OF": 
            self.conv1=nn.Conv3d(2, 64, kernel_size=(3, 3, 3), stride=1,padding=(1,0,0))
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
        #shape(10,11,2,224,224)   bacth,window,channel,h,w
       
        x = self.maxpool1(self.relu(self.batchnorm1(self.conv1(x))))    #shape(10, 64, 10, 111, 111)
       
        x = self.maxpool2(self.relu(self.batchnorm2(self.conv2(x))))    #shape(10, 128, 4, 54, 54)
       
        x = self.maxpool2(self.conv3b(self.relu(self.batchnorm3(self.conv3a(x)))))   #shape(10, 256, 1, 26, 26)
        
        
        return x


class TwoStream_Fusion2(nn.Module):
    def __init__(self):
        super(TwoStream_Fusion2,self).__init__()

        self.RGB_stream = C3D2("RGB")
        self.OF_stream = C3D2("OF")
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

    def forward(self, x1,x2):
        #shape(10,11,2,224,224)   bacth,window,channel,h,w
        x1 = self.RGB_stream(x1)
        x2 = self.OF_stream(x2)
        x = x1+x2
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