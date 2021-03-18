import os
import numpy as np
import time
import argparse
# import logging

#from mpi4py import MPI
from math import ceil
from random import Random
#import networkx as nx

import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.multiprocessing import Process
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import torchvision.models as IMG_models

from models import *
from models.Semi_net import SemiNet

# logging.basicConfig(level=logging.INFO)


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234, isNonIID=False, alpha=0):
        self.data = data
        self.partitions = []
        self.ratio = [0] * len(sizes)
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)


        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

        if isNonIID:
            self.partitions, self.ratio = self.__getDirichletData__(data, sizes, seed, alpha)

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

    def __getNonIIDdata__(self, data, sizes, seed):
        labelList = data.train_labels
        rng = Random()
        rng.seed(seed)
        a = [(label, idx) for idx, label in enumerate(labelList)]
        # Same Part
        labelIdxDict = dict()
        for label, idx in a:
            labelIdxDict.setdefault(label,[])
            labelIdxDict[label].append(idx)
        labelNum = len(labelIdxDict)
        labelNameList = [key for key in labelIdxDict]
        labelIdxPointer = [0] * labelNum
        # sizes = number of nodes
        partitions = [list() for i in range(len(sizes))]
        eachPartitionLen= int(len(labelList)/len(sizes))
        majorLabelNumPerPartition = ceil(labelNum/len(partitions))
        basicLabelRatio = 0.4

        interval = 1
        labelPointer = 0

        #basic part
        for partPointer in range(len(partitions)):
            requiredLabelList = list()
            for _ in range(majorLabelNumPerPartition):
                requiredLabelList.append(labelPointer)
                labelPointer += interval
                if labelPointer > labelNum - 1:
                    labelPointer = interval
                    interval += 1
            for labelIdx in requiredLabelList:
                start = labelIdxPointer[labelIdx]
                idxIncrement = int(basicLabelRatio*len(labelIdxDict[labelNameList[labelIdx]]))
                partitions[partPointer].extend(labelIdxDict[labelNameList[labelIdx]][start:start+ idxIncrement])
                labelIdxPointer[labelIdx] += idxIncrement

        #random part
        remainLabels = list()
        for labelIdx in range(labelNum):
            remainLabels.extend(labelIdxDict[labelNameList[labelIdx]][labelIdxPointer[labelIdx]:])
        rng.shuffle(remainLabels)
        for partPointer in range(len(partitions)):
            idxIncrement = eachPartitionLen - len(partitions[partPointer])
            partitions[partPointer].extend(remainLabels[:idxIncrement])
            rng.shuffle(partitions[partPointer])
            remainLabels = remainLabels[idxIncrement:]
        return partitions

    def __getDirichletData__(self, data, psizes, seed, alpha):
        sizes = len(psizes)
        labelList = data.targets
        rng = Random()
        rng.seed(seed)
        a = [(label, idx) for idx, label in enumerate(labelList)]
        # Same Part
        labelIdxDict = dict()
        for label, idx in a:
            labelIdxDict.setdefault(label,[])
            labelIdxDict[label].append(idx)
        labelNum = len(labelIdxDict) #10
        labelNameList = [key for key in labelIdxDict]
        # rng.shuffle(labelNameList)
        labelIdxPointer = [0] * labelNum
        # sizes = number of nodes
        partitions = [list() for i in range(sizes)] # of size (m)
        np.random.seed(seed)
        distribution = np.random.dirichlet([alpha] * sizes, labelNum).tolist() # of size (10, m)

        # basic part
        for row_id, dist in enumerate(distribution):
            subDictList = labelIdxDict[labelNameList[row_id]]
            rng.shuffle(subDictList)
            totalNum = len(subDictList)
            dist = self.handlePartition(dist, totalNum)
            for i in range(len(dist)-1):
                partitions[i].extend(subDictList[dist[i]:dist[i+1]+1])

        #random part
        a = [len(partitions[i]) for i in range(len(partitions))]
        ratio = [a[i]/sum(a) for i in range(len(a))]
        return partitions, ratio

    def handlePartition(self, plist, length):
        newList = [0]
        canary = 0
        for i in range(len(plist)):
            canary = int(canary + length*plist[i])
            newList.append(canary)
        return newList

def partition_dataset(rank, size, args):
    print('==> load train data')
    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data',
                                                train=True,
                                                download=True,
                                                transform=transform_train)

        partition_sizes = [1.0 / size for _ in range(size)]
        partition = DataPartitioner(trainset, partition_sizes, isNonIID=args.NIID, alpha=args.alpha)
        ratio = partition.ratio
        partition = partition.use(rank)
        train_loader = torch.utils.data.DataLoader(partition,
                                                batch_size=args.bs,
                                                shuffle=True,
                                                pin_memory=True)

        print('==> load test data')
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(root='./data',
                                            train=False,
                                            download=True,
                                            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                batch_size=64,
                                                shuffle=False,
                                                num_workers=size)

    if args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR100(root='/users/jianyuw1/AdaDSGD/data',
                                                train=True,
                                                download=True,
                                                transform=transform_train)

        partition_sizes = [1.0 / size for _ in range(size)]
        partition = DataPartitioner(trainset, partition_sizes, isNonIID=False)
        ratio = partition.ratio
        partition = partition.use(rank)
        train_loader = torch.utils.data.DataLoader(partition,
                                                batch_size=args.bs,
                                                shuffle=True,
                                                pin_memory=True)

        print('==> load test data')
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR100(root='/users/jianyuw1/AdaDSGD/data',
                                            train=False,
                                            download=True,
                                            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                batch_size=64,
                                                shuffle=False,
                                                num_workers=size)


    elif args.dataset == 'imagenet':
        datadir = '/datasets/shared/imagenet/ILSVRC2015/Data/'
        traindir = os.path.join(datadir, 'CLS-LOC/train/')
        #valdir = os.path.join(datadir, 'CLS-LOC/')
        #testdir = os.path.join(datadir, 'CLS-LOC/')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        partition_sizes = [1.0 / size for _ in range(size)]
        partition = DataPartitioner(train_dataset, partition_sizes, isNonIID=False)
        ratio = partition.ratio
        partition = partition.use(rank)

        train_loader = torch.utils.data.DataLoader(
            partition, batch_size=args.bs, shuffle=True,
             pin_memory=True)
        '''
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.bs, shuffle=False,
            pin_memory=True)
        val_loader = None
        '''
        test_loader = None

    if args.dataset == 'emnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = torchvision.datasets.EMNIST(root='/users/jianyuw1/AdaDSGD/data',
                                              split = 'balanced',
                                              train=True,
                                              download=True,
                                              transform=transform_train)
        partition_sizes = [1.0 / size for _ in range(size)]
        partition = DataPartitioner(train_dataset, partition_sizes, isNonIID=False)
        ratio = partition.ratio
        partition = partition.use(rank)

        train_loader = torch.utils.data.DataLoader(
            partition, batch_size=args.bs, shuffle=True,
             pin_memory=True)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        testset = torchvision.datasets.EMNIST(root='/users/jianyuw1/AdaDSGD/data',
                                             split = 'balanced',
                                             train=False,
                                             download=True,
                                             transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                batch_size=64,
                                                shuffle=False,
                                                num_workers=size)



    return train_loader, test_loader

def select_model(num_class, args):
    if args.model == 'VGG':
        model = VGG(16, num_class)
    elif args.model == 'res':
        if args.dataset == 'cifar10':
            # model = resnet.ResNet(34, num_class)
            model = ResNet18()
        elif args.dataset == 'imagenet':
            model = IMG_models.resnet18()
    elif args.model == 'wrn':
        num_class = 10
        model = Wide_ResNet(28,10,0,num_class)
    elif args.model == 'mlp':
        if args.dataset == 'emnist':
            model = MNIST_MLP(47)
    elif args.model == 'res_gn':
            # model = resnet.ResNet(34, num_class)
        model = ResNet18_gn()
    elif args.model == 'res_ln':
        if args.dataset == 'cifar10':
            # model = resnet.ResNet(34, num_class)
            model = ResNet18_LN()
    elif args.model == 'EMNIST_model':
        if args.dataset == 'emnist':
            # model = resnet.ResNet(34, num_class)
            model = Net()
    elif args.model == 'UserSemi_model':

        model = SemiNet()

    return model

def comp_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Meter(object):
    """ Computes and stores the average, variance, and current value """

    def __init__(self, init_dict=None, ptag='Time', stateful=False,
                 csv_format=True):
        """
        :param init_dict: Dictionary to initialize meter values
        :param ptag: Print tag used in __str__() to identify meter
        :param stateful: Whether to store value history and compute MAD
        """
        self.reset()
        self.ptag = ptag
        self.value_history = None
        self.stateful = stateful
        if self.stateful:
            self.value_history = []
        self.csv_format = csv_format
        if init_dict is not None:
            for key in init_dict:
                try:
                    # TODO: add type checking to init_dict values
                    self.__dict__[key] = init_dict[key]
                except Exception:
                    print('(Warning) Invalid key {} in init_dict'.format(key))

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.std = 0
        self.sqsum = 0
        self.mad = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sqsum += (val ** 2) * n
        if self.count > 1:
            self.std = ((self.sqsum - (self.sum ** 2) / self.count)
                        / (self.count - 1)
                        ) ** 0.5
        if self.stateful:
            self.value_history.append(val)
            mad = 0
            for v in self.value_history:
                mad += abs(v - self.avg)
            self.mad = mad / len(self.value_history)

    def __str__(self):
        if self.csv_format:
            if self.stateful:
                return str('{dm.val:.3f},{dm.avg:.3f},{dm.mad:.3f}'
                           .format(dm=self))
            else:
                return str('{dm.val:.3f},{dm.avg:.3f},{dm.std:.3f}'
                           .format(dm=self))
        else:
            if self.stateful:
                return str(self.ptag) + \
                       str(': {dm.val:.3f} ({dm.avg:.3f} +- {dm.mad:.3f})'
                           .format(dm=self))
            else:
                return str(self.ptag) + \
                       str(': {dm.val:.3f} ({dm.avg:.3f} +- {dm.std:.3f})'
                           .format(dm=self))
