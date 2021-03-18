import sys
import torch
import numpy as np
# from .cifar import get

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class SemiNet(torch.nn.Module):
    def __init__(self,in_channels=3,taskcla=None):
        super(SemiNet,self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels,32,kernel_size=3,stride=1, padding=1)
        self.GN1 = torch.nn.GroupNorm(32,32)
        self.BN1 = torch.nn.BatchNorm2d(32)

        self.conv2=torch.nn.Conv2d(32,64,kernel_size=3,stride=1, padding=1)

        self.conv3=torch.nn.Conv2d(64,128,kernel_size=3,stride=1, padding=1)
        self.GN2 = torch.nn.GroupNorm(32,128)
        self.BN2 = torch.nn.BatchNorm2d(128)

        self.conv4=torch.nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)

        self.conv5=torch.nn.Conv2d(128,256,kernel_size=3,stride=1, padding=1)
        self.GN3 = torch.nn.GroupNorm(32,256)
        self.BN3 = torch.nn.BatchNorm2d(256)

        self.conv6=torch.nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1)

        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()

        self.drop1=torch.nn.Dropout(0.05)
        self.drop2=torch.nn.Dropout(0.1)
        self.drop3=torch.nn.Dropout(0.1)

        self.fc1=torch.nn.Linear(4096,1024)
        self.fc2=torch.nn.Linear(1024,512)
        self.fc3 = torch.nn.Linear(512,10)

        return


    def forward(self,x):

        h = self.relu(self.BN1((self.conv1(x))))

        h = self.maxpool(self.relu(self.conv2(h)))

        h = self.relu(self.BN2((self.conv3(h))))

        h= self.drop1(self.maxpool(self.relu(self.conv4(h))))

        h = self.relu(self.BN3((self.conv5(h))))

        h = self.maxpool(self.relu(self.conv6(h)))

        h=h.view(x.size(0),-1)

        h=self.drop2(self.relu(self.fc1(h)))
        h=self.drop3(self.relu(self.fc2(h)))

        y=self.fc3(h)

        return y
