import torch
import torch.nn as nn
import sys

n_channel=32

cfg = {
    'VGG9':  [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, input_size=96, num_class=10):
        super(VGG, self).__init__()
        self.input_size = input_size
        self.features = self._make_layers(cfg[vgg_name])
        self.n_maps = cfg[vgg_name][-2]
        self.fc = self._make_fc_layers()
        self.classifier1 = nn.Linear(self.n_maps, 128)
        self.classifier2 = nn.Linear(128, num_class)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        # print('out.size',out.size(),self.n_maps)
        out = self.fc(out)
        out = self.classifier1(out)
        out = self.classifier2(out)
        return out

    def _make_fc_layers(self):
        layers = []
        # print(self.n_maps*self.input_size*self.input_size, self.n_maps)
        # print('self.input_size',self.input_size)
        # layers += [nn.Linear(self.n_maps*self.input_size*self.input_size, self.n_maps),
        #            nn.BatchNorm1d(self.n_maps),
        #            nn.ReLU(inplace=True)]
        layers += [nn.Linear(4608, self.n_maps),
                   nn.BatchNorm1d(self.n_maps),
                   nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
                self.input_size = self.input_size // 2
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

def VGG9(input_size, num_class):
    return VGG('VGG9', input_size, num_class)

def VGG16(input_size, num_class):
    return VGG('VGG16', input_size, num_class)

def VGG19(input_size, num_class):
    return VGG('VGG19', input_size, num_class)

def VGG11(input_size, num_class):
    return VGG('VGG11', input_size, num_class)

def test():
    #net = VGG('VGG11', input_size=96,num_class=10)
    net = VGG('VGG11', input_size=32,num_class=10)
    print(net)
    x = torch.randn(128, 3, 96, 96)
    # x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())
# test()
