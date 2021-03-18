import torch
import torch.nn as nn


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
