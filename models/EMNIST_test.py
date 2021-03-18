import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from EMNIST_model import *
# Training settings
batch_size = 64


train_dataset = datasets.EMNIST(root='./data/',
                               train=True,
                               transform=transforms.ToTensor(),split='byclass',
                               download=True)

test_dataset = datasets.EMNIST(root='./data/',
                              train=False,split='byclass',
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):


        output = model(data)
        #output:64*10

        loss = F.nll_loss(output, target)

        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
