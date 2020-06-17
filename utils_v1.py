from torchvision import datasets, transforms
import math
import random
import torch
import copy
import csv
from datetime import datetime
import os
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn
import functools
import operator
import transform as T

class Utils():
    def target_transform(target):
        return int(target) - 1
    def get_dataset_dist(self, args):
        if args.dataset == "cifar10":
            data_dir = "./data"
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            train_dataset = datasets.CIFAR10(
                root=data_dir,
                train=True,
                download=True,
                transform=transform
            )

            test_dataset = datasets.CIFAR10(
                root=data_dir,
                train=False,
                download=True,
                transform=transform
            )
        if args.dataset == "svhn":
            data_dir = "./data"

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            train_dataset = datasets.SVHN(
                root=data_dir,
                split='train',
                download=True,
                transform=transform
            )

            test_dataset = datasets.SVHN(
                root=data_dir,
                split='test',
                download=True,
                transform=transform
            )

        if args.dataset == "emnist":
            data_dir = "./data"

            train_dataset = datasets.EMNIST(root='./data/',
                                           train=True,split='balanced',  ##byclass
                                           download=True)

            test_dataset = datasets.EMNIST(root='./data/',
                                          train=False,split='balanced')

        if args.dataset == "cifar10":
            if args.iid:
                user_idxs,server_idxs = self.iid_dist_cifar(train_dataset, args)
            else:
                user_idxs,server_idxs = self.getNonIIDdata_cifar(train_dataset, args)

        if args.dataset == "svhn":
            if args.iid:
                user_idxs,server_idxs = self.iid_dist_svhn(train_dataset, args)
            else:
                user_idxs,server_idxs = self.getNonIIDdata_svhn(train_dataset, args)

        if args.dataset == "emnist":
            if args.iid:
                user_idxs,server_idxs = self.iid_dist_emnist(train_dataset, args)
            else:
                user_idxs,server_idxs = self.getNonIIDdata_emnist(train_dataset, args)

        return train_dataset, test_dataset, user_idxs,server_idxs

    def iid_dist_cifar(self, dataset, args):
        data_per_device = (len(dataset) -  args.num_data_server)//args.num_devices# args.max_data_per_device

        num_sample_of_server = len(dataset) - args.num_devices*data_per_device
        num_sample_of_each_class_server = int(num_sample_of_server/len(dataset.classes))

        server_idxs=[]
        rest_idxs = []
        idx_target_class_list = []

        for target_class in range(len(dataset.classes)):
            idxs_target_class = np.where(np.array(dataset.targets)==target_class)[0].tolist()
            np.random.shuffle(idxs_target_class)
            idx_target_class_list.append(idxs_target_class)
            ### distribute sample of each class to the server
            server_idxs += idx_target_class_list[target_class][0:num_sample_of_each_class_server]
            ### record rest sample according to class
            rest_idxs += idx_target_class_list[target_class][num_sample_of_each_class_server:]

        users_idxs = [[] for i in range(args.num_devices)]  # Index dictionary for devices
        random.shuffle(rest_idxs)
        for i in range(args.num_devices):
            users_idxs[i] = rest_idxs[i*data_per_device:(i+1)*data_per_device]

        return users_idxs,server_idxs

    def iid_dist_emnist(self, dataset, args):
        data_per_device = (len(dataset) -  args.num_data_server)//args.num_devices# args.max_data_per_device
        num_class = 47
        num_sample_of_server = len(dataset) - args.num_devices*data_per_device
        num_sample_of_each_class_server = int(num_sample_of_server/num_class)

        server_idxs=[]
        rest_idxs = []
        idx_target_class_list = []

        for target_class in range(num_class):
            idxs_target_class = np.where(np.array(dataset.targets)==target_class)[0].tolist()
            np.random.shuffle(idxs_target_class)
            idx_target_class_list.append(idxs_target_class)
            ### distribute sample of each class to the server
            server_idxs += idx_target_class_list[target_class][0:num_sample_of_each_class_server]
            ### record rest sample according to class
            rest_idxs += idx_target_class_list[target_class][num_sample_of_each_class_server:]

        users_idxs = [[] for i in range(args.num_devices)]  # Index dictionary for devices
        random.shuffle(rest_idxs)
        for i in range(args.num_devices):
            users_idxs[i] = rest_idxs[i*data_per_device:(i+1)*data_per_device]

        return users_idxs,server_idxs

    def iid_dist_svhn(self, dataset, args):
        data_per_device = (len(dataset) -  args.num_data_server)//args.num_devices# args.max_data_per_device

        num_sample_of_server = len(dataset) - args.num_devices*data_per_device
        num_sample_of_each_class_server = int(num_sample_of_server/10)

        server_idxs=[]
        rest_idxs = []
        idx_target_class_list = []

        for target_class in range(10):
            idxs_target_class = np.where(np.array(dataset.labels)==target_class)[0].tolist()
            np.random.shuffle(idxs_target_class)
            idx_target_class_list.append(idxs_target_class)
            ### distribute sample of each class to the server
            server_idxs += idx_target_class_list[target_class][0:num_sample_of_each_class_server]
            ### record rest sample according to class
            rest_idxs += idx_target_class_list[target_class][num_sample_of_each_class_server:]

        users_idxs = [[] for i in range(args.num_devices)]  # Index dictionary for devices
        random.shuffle(rest_idxs)
        for i in range(args.num_devices):
            users_idxs[i] = rest_idxs[i*data_per_device:(i+1)*data_per_device]

        return users_idxs,server_idxs



    def getNonIIDdata_cifar(self, data, args):
        from random import Random
        from math import ceil
        labelList = data.targets
        num_sample_of_server = args.num_data_server
        server_idxs = np.arange(0,num_sample_of_server).tolist()

        ######
        num_sample_of_each_class_server = int(num_sample_of_server/len(data.classes))

        server_idxs=[]
        idx_target_class_list = []

        for target_class in range(len(data.classes)):
            idxs_target_class = np.where(np.array(data.targets)==target_class)[0].tolist()
            np.random.shuffle(idxs_target_class)
            idx_target_class_list.append(idxs_target_class)
            ### distribute sample of each class to the server
            server_idxs += idx_target_class_list[target_class][0:num_sample_of_each_class_server]

        ######
        rng = Random()
        rng.seed(2020)

        a = []
        for id in range(len(data)):
            if id not in server_idxs:
                a.append((labelList[id], id))

        # Same Part
        labelIdxDict = dict()
        for label, idx in a:
            labelIdxDict.setdefault(label,[])
            labelIdxDict[label].append(idx)
        labelNum = len(labelIdxDict)
        labelNameList = [key for key in labelIdxDict]
        labelIdxPointer = [0] * labelNum
        base_size = 10
        partitions = [list() for i in range(base_size)]
        eachPartitionLen= int(len(a)/base_size)
        majorLabelNumPerPartition = ceil(labelNum/len(partitions))
        print('majorLabelNumPerPartition',majorLabelNumPerPartition)
        basicLabelRatio = args.basicLabelRatio

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
            for labelIdx in requiredLabelList:

                start = labelIdxPointer[labelIdx]
                idxIncrement = int(eachPartitionLen*basicLabelRatio)
                partitions[partPointer].extend(labelIdxDict[labelNameList[labelIdx]][start:start+ idxIncrement])
                length_tmp = len(labelIdxDict[labelNameList[labelIdx]][start:start+ idxIncrement])
                tmp = np.arange(start).tolist()
                random.shuffle(tmp)
                index_tmp = np.array(labelIdxDict[labelNameList[labelIdx]])
                if length_tmp < idxIncrement:
                    partitions[partPointer].extend(index_tmp[tmp[0:idxIncrement-length_tmp]].tolist())

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

        if args.num_devices == 20:
            partitions_list = [list() for i in range(20)]
            for partPointer in range(10):
                index_partitions = partitions[partPointer]
                len_partitions = len(index_partitions)
                rng.shuffle(index_partitions)
                partitions_list[partPointer] = index_partitions[0:len_partitions//2]
                partitions_list[partPointer+10] = index_partitions[len_partitions//2:]
            partitions = partitions_list

        if args.num_devices == 30:
            partitions_list = [list() for i in range(30)]
            for partPointer in range(10):
                index_partitions = partitions[partPointer]
                len_partitions = len(index_partitions)
                rng.shuffle(index_partitions)
                partitions_list[partPointer] = index_partitions[0:len_partitions//3]
                partitions_list[partPointer+10] = index_partitions[len_partitions//3:len_partitions*2//3]
                partitions_list[partPointer+20] = index_partitions[len_partitions*2//3:]
            partitions = partitions_list


        return partitions,server_idxs

    def getNonIIDdata_svhn(self, data, args):
        from random import Random
        from math import ceil

        labelList = copy.deepcopy(data.labels) - 1
        num_sample_of_server = args.num_data_server
        server_idxs = np.arange(0,num_sample_of_server).tolist()

        ######
        num_sample_of_each_class_server = int(num_sample_of_server/10)

        server_idxs=[]
        idx_target_class_list = []

        for target_class in range(10):
            idxs_target_class = np.where(np.array(data.labels)==target_class)[0].tolist()
            np.random.shuffle(idxs_target_class)
            idx_target_class_list.append(idxs_target_class)
            ### distribute sample of each class to the server
            server_idxs += idx_target_class_list[target_class][0:num_sample_of_each_class_server]

        ######
        rng = Random()
        rng.seed(2020)
        a = []
        for id in range(len(data)):
            if id not in server_idxs:
                a.append((labelList[id], id))

        # Same Part
        labelIdxDict = dict()
        for label, idx in a:
            labelIdxDict.setdefault(label,[])
            labelIdxDict[label].append(idx)
        labelNum = len(labelIdxDict)
        labelNameList = [key for key in labelIdxDict]
        labelIdxPointer = [0] * labelNum

        sizes = 10#args.num_devices
        partitions = [list() for i in range(sizes)]
        eachPartitionLen= int(len(a)/sizes)
        majorLabelNumPerPartition = ceil(labelNum/len(partitions))
        print('majorLabelNumPerPartition',majorLabelNumPerPartition)
        basicLabelRatio = args.basicLabelRatio

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
            for labelIdx in requiredLabelList:
                start = labelIdxPointer[labelIdx]
                idxIncrement = int(eachPartitionLen*basicLabelRatio)
                partitions[partPointer].extend(labelIdxDict[labelNameList[labelIdx]][start:start+ idxIncrement])
                length_tmp = len(labelIdxDict[labelNameList[labelIdx]][start:start+ idxIncrement])
                tmp = np.arange(start).tolist()
                index_tmp = np.array(labelIdxDict[labelNameList[labelIdx]])
                if length_tmp < idxIncrement:
                    partitions[partPointer].extend(index_tmp[tmp[0:idxIncrement-length_tmp]].tolist())

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

        if args.num_devices == 20:
            partitions_list = [list() for i in range(20)]
            for partPointer in range(10):
                index_partitions = partitions[partPointer]
                len_partitions = len(index_partitions)
                rng.shuffle(index_partitions)
                partitions_list[partPointer] = index_partitions[0:len_partitions//2]
                partitions_list[partPointer+10] = index_partitions[len_partitions//2:]
            partitions = partitions_list

        if args.num_devices == 30:
            partitions_list = [list() for i in range(30)]
            for partPointer in range(10):
                index_partitions = partitions[partPointer]
                len_partitions = len(index_partitions)
                rng.shuffle(index_partitions)
                partitions_list[partPointer] = index_partitions[0:len_partitions//3]
                partitions_list[partPointer+10] = index_partitions[len_partitions//3:len_partitions*2//3]
                partitions_list[partPointer+20] = index_partitions[len_partitions*2//3:]
            partitions = partitions_list


        return partitions,server_idxs


    def getNonIIDdata_emnist(self, data, args):
        from random import Random
        from math import ceil
        labelList = data.targets
        labelList = labelList.cpu().numpy()#[0:50000]
        num_class = 47
        Total_labelList = data.targets
        Total_labelList = Total_labelList.cpu().numpy()#[0:50000]


        print(labelList.shape)
        num_sample_of_server = args.num_data_server#4000#len(data) - args.num_devices*args.max_data_per_device

        ######
        num_sample_of_each_class_server = int(num_sample_of_server/num_class)

        server_idxs=[]
        idx_target_class_list = []

        for target_class in range(num_class):
            idxs_target_class = np.where(np.array(labelList)==target_class)[0].tolist()
            np.random.shuffle(idxs_target_class)
            idx_target_class_list.append(idxs_target_class)
            ### distribute sample of each class to the server
            server_idxs += idx_target_class_list[target_class][0:num_sample_of_each_class_server]
        np.random.shuffle(server_idxs)
        ######
        # labelList = rest_idxs
        rng = Random()
        rng.seed(2020)

        a = []
        b = []
        devices_labelid = []
        for id in range(len(Total_labelList)):
            if id not in server_idxs:
                a.append((labelList[id], id))
                b.append(labelList[id])
                devices_labelid.append(id)

        ### sampels probility
        sample_probility_ = []
        for id in range(num_class):
            index = np.where(np.array(b)==id)[0]
            index = list(index)
            sample_probility_.append(len(index))
        sample_probility = np.array(sample_probility_)/np.sum(sample_probility_)
        print(sample_probility)

        # Same Part
        labelIdxDict = dict()
        for label, idx in a:
            labelIdxDict.setdefault(label,[])
            labelIdxDict[label].append(idx)

        labelNum = len(labelIdxDict)

        labelNameList = [key for key in labelIdxDict]
        labelIdxPointer = [0] * labelNum

        base_size = args.num_devices
        partitions = [list() for i in range(base_size)]
        eachPartitionLen= int(len(a)/base_size)
        majorLabelNumPerPartition = ceil(labelNum/len(partitions))
        basicLabelRatio = args.basicLabelRatio


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
            for labelIdx in requiredLabelList:
                idxIncrement = int(sample_probility_[labelIdx]*basicLabelRatio)
                partitions[partPointer].extend(labelIdxDict[labelIdx][0:idxIncrement])
                labelIdxDict[labelIdx][0:idxIncrement] = []


        #random part
        remainLabels = list()
        for labelIdx in range(labelNum):
            remainLabels.extend(labelIdxDict[labelIdx])

        remainLabels_class = []
        for target_class in range(num_class):
            tmp = np.where(np.array(Total_labelList[remainLabels])==target_class)[0].tolist()
            remainLabels_class.append(len(tmp))

        for partPointer in range(len(partitions)):
            for target_class in range(num_class):
                idxs_target_class = np.where(np.array(Total_labelList[remainLabels])==target_class)[0].tolist()
                num_sample_of_each_class = int(remainLabels_class[partPointer]*sample_probility[target_class])
                remainLabels_array = np.array(remainLabels)
                partitions[partPointer].extend(remainLabels_array[idxs_target_class[0:num_sample_of_each_class]].tolist())
                remainLabels = np.delete(np.array(remainLabels), idxs_target_class[0:num_sample_of_each_class]).tolist()
            rng.shuffle(partitions[partPointer])
        return partitions,server_idxs


def weight_diff(weights, layer_name):
    w_1 = copy.deepcopy(weights[0])   # Weight from first device
    w_2 = copy.deepcopy(weights[0])
    w_avg = copy.deepcopy(weights[-1])
    layer_name_1 = layer_name + '.weight'
    layer_name_2 = layer_name + '.bias'
    for i in range(len(weights)-1):
        for key in w_1.keys():
            # print(key)
            if key == layer_name_1 or key == layer_name_2:
                weights[i][key] = weights[i][key] - w_avg[key]

    w_1 = copy.deepcopy(weights[0])   # Weight from first device
    w_2 = copy.deepcopy(weights[0])

    W_2_list = []
    for key in w_2.keys():
        if key == layer_name_1 or key == layer_name_2:
            W_2_list.append(torch.norm(w_2[key])**2)

    k = 0
    for key in w_1.keys():
        if key == layer_name_1 or key == layer_name_2:
            for i in range(1, len(weights)-1):    # Other devices
                w_1[key] += weights[i][key]
                W_2_list[k] += torch.norm(weights[i][key])**2
            k += 1
    W_1_list = []
    for key in w_1.keys():
        if key == layer_name_1 or key == layer_name_2:
            W_1_list.append(torch.norm(w_1[key])**2)

    sum_ratio = []

    for i in range(len(W_1_list)):
        w2 = W_2_list[i].item()
        w1 = W_1_list[i].item()
        sum_ratio.append(w2/w1)

    mean_ratio = np.mean(sum_ratio)
    return mean_ratio
