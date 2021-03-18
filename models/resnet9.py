import torch.nn as nn
# from base import Base

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class Base(nn.Module):
    def __init__(self):
        super().__init__()

    def train_step(self, batch, model, device):
        images, labels = batch
        images = images.to(device)
        output = model(images)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output, labels)
        return loss

    def validation_step(self, batch, model, device):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        acc = accuracy(output, labels)
        return acc

def conv_bn_relu_pool(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        # nn.BatchNorm2d(out_channels),
        nn.GroupNorm(32,out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(Base):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.prep = conv_bn_relu_pool(in_channels, 64)
        self.layer1_head = conv_bn_relu_pool(64, 128, pool=True)
        self.layer1_residual = nn.Sequential(conv_bn_relu_pool(128, 128), conv_bn_relu_pool(128, 128))
        self.layer2 = conv_bn_relu_pool(128, 256, pool=True)
        self.layer3_head = conv_bn_relu_pool(256, 512, pool=True)
        self.layer3_residual = nn.Sequential(conv_bn_relu_pool(512, 512), conv_bn_relu_pool(512, 512))
        self.MaxPool2d = nn.Sequential(
            nn.MaxPool2d(4))
        self.linear = nn.Linear(512, num_classes)
        # self.classifier = nn.Sequential(
        #     nn.MaxPool2d(4),
        #     nn.Flatten(),
        #     nn.Linear(512, num_classes))


    def forward(self, x):
        x = self.prep(x)
        x = self.layer1_head(x)
        x = self.layer1_residual(x) + x
        x = self.layer2(x)
        x = self.layer3_head(x)
        x = self.layer3_residual(x) + x
        x = self.MaxPool2d(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
