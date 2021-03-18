import sys
import torch
import numpy as np
# from .cifar import get

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class Net(torch.nn.Module):
    def __init__(self,inputsize,taskcla=None):
        super(Net,self).__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla

        self.conv1=torch.nn.Conv2d(ncha,64,kernel_size=size//8)
        s=compute_conv_output_size(size,size//8)
        s=s//2
        self.conv2=torch.nn.Conv2d(64,128,kernel_size=size//10)
        s=compute_conv_output_size(s,size//10)
        s=s//2
        self.conv3=torch.nn.Conv2d(128,256,kernel_size=2)
        s=compute_conv_output_size(s,2)
        s=s//2
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()

        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)
        # self.fc1=torch.nn.Linear(256*s*s,2048)
        self.fc1=torch.nn.Linear(2304,2048)
        self.fc2=torch.nn.Linear(2048,2048)
        self.last=torch.nn.ModuleList()

        self.last = torch.nn.Linear(2048,10)
        # data,taskcla,size = get()
        # for t,n in taskcla:
        #     print('t n ',t,n)
        #     self.last.append(torch.nn.Linear(2048,n))

        return

    # def forward(self,x):
    #     h=self.maxpool(self.drop1(self.relu(self.conv1(x))))
    #     h=self.maxpool(self.drop1(self.relu(self.conv2(h))))
    #     h=self.maxpool(self.drop2(self.relu(self.conv3(h))))
    #     h=h.view(x.size(0),-1)
    #     h=self.drop2(self.relu(self.fc1(h)))
    #     h=self.drop2(self.relu(self.fc2(h)))
    #     y=self.last(h)
    #     # for i in range(2):
    #     #     y.append(self.last[i](h))
    #     return y

    def forward(self,x):
        h=self.maxpool(self.drop1(self.relu(self.conv1(x))))
        h=self.maxpool(self.drop1(self.relu(self.conv2(h))))
        h=self.maxpool(self.drop2(self.relu(self.conv3(h))))
        h=h.view(x.size(0),-1)
        h=self.drop2(self.relu(self.fc1(h)))
        h=self.drop2(self.relu(self.fc2(h)))
        y=self.last(h)
        # for i in range(2):
        #     y.append(self.last[i](h))
        return y
