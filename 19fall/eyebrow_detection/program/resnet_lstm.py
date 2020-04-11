from torchvision import models
import torch.nn as nn

class my_resnet(nn.Module):
    def __init__(self):
        super(my_resnet, self).__init__()
        resnet = models.resnet34(norm_layer = nn.BatchNorm2d)
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
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)  # 2 for bich.abs(x-y), 1)direction

    def forward(self, x):
        # Set initial states
        # h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)).cuda() # 2 for bidirection
        # c0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)).cuda() # 2 for bidirection
        # Forward propagate LSTM
        # out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        out, _ = self.lstm(x)
        #print(out.shape)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :]).squeeze()
        return out

class resnet_lstm(nn.Module):
    def __init__(self,window_size):
        super(resnet_lstm, self).__init__()
        self.resnet = my_resnet()
        self.bilstm = BiRNN(512, 256, 2)
        self.window_size = window_size

    def forward(self, x):
        # batch t chanel h w
        #print(x.shape)
        x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
        #print(x.shape)
        x = self.resnet(x)
        x = x.view(-1, self.window_size, x.shape[-3], x.shape[-2], x.shape[-1]).squeeze() #batch, t, ft_dim
        x = x.resize_([1,7,512])

        #x = x.view(-1, self.window_size, 256*4*4) #5*4096  # 5 to 7, 9, or 11
        #s = self.fc_s(x)
        x = self.bilstm(x) #
        1

        return x