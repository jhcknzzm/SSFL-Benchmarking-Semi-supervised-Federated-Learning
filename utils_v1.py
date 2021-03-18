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
import json
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from scipy import io
import torch

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

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

            train_dataset = datasets.EMNIST(root='./data/',
                                           train=True,
                                           transform=transform,split='balanced',  ##byclass
                                           download=True)

            test_dataset = datasets.EMNIST(root='./data/',
                                          train=False,split='balanced',
                                          transform=transform)

        if args.dataset == "cifar10":
            if args.iid:
                user_idxs,server_idxs = self.iid_dist(train_dataset, args)
                # user_idxs,server_idxs = self.iid_dist_random(train_dataset,args)
            else:
                # user_idxs,server_idxs = self.noniid_dist(train_dataset, args)
                # user_idxs,server_idxs = self.noniid_dist_2(train_dataset, args)
                # user_idxs,server_idxs = self.non_iid_dist_random(train_dataset, args)
                user_idxs,server_idxs = self.getNonIIDdata(train_dataset, args)

        if args.dataset == "svhn":
            if args.iid:
                user_idxs,server_idxs = self.iid_dist_svhn(train_dataset, args)
                # user_idxs,server_idxs = self.iid_dist_random(train_dataset,args)
            else:
                # user_idxs,server_idxs = self.noniid_dist(train_dataset, args)
                # user_idxs,server_idxs = self.noniid_dist_2(train_dataset, args)
                # user_idxs,server_idxs = self.non_iid_dist_random(train_dataset, args)
                user_idxs,server_idxs = self.getNonIIDdata_svhn(train_dataset, args)

        if args.dataset == "emnist":
            if args.iid:
                user_idxs,server_idxs = self.iid_dist_emnist(train_dataset, args)
                # user_idxs,server_idxs = self.iid_dist_random(train_dataset,args)
            else:
                # user_idxs,server_idxs = self.noniid_dist(train_dataset, args)
                # user_idxs,server_idxs = self.noniid_dist_2(train_dataset, args)
                # user_idxs,server_idxs = self.non_iid_dist_random(train_dataset, args)
                # user_idxs,server_idxs = self.getNonIIDdata_emnist(train_dataset, args)
                if args.attack:
                    user_idxs, server_idxs, attack_idxs = self.getNonIIDdata_emnist_with_attacker(train_dataset, args)
                else:
                    user_idxs,server_idxs = self.getNonIIDdata_emnist(train_dataset, args)


        if args.attack:
            return train_dataset, test_dataset, user_idxs, server_idxs, attack_idxs
        else:
            return train_dataset, test_dataset, user_idxs, server_idxs

    def iid_dist(self, dataset, args):
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

    def iid_dist_emnist(self, dataset, args):
        data_per_device = (len(dataset) -  args.num_data_server)//args.num_devices# args.max_data_per_device
        print('data_per_device',data_per_device)
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
        # sizes = number of nodes
        # sizes = args.num_devices
        base_size = args.num_devices
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
                    # interval += 1
            # print('requiredLabelList',requiredLabelList)
            for labelIdx in requiredLabelList:

                idxIncrement = int(sample_probility_[labelIdx]*basicLabelRatio)

                partitions[partPointer].extend(labelIdxDict[labelIdx][0:idxIncrement])

                labelIdxDict[labelIdx][0:idxIncrement] = []


        #random part
        remainLabels = list()
        # for labelIdx in range(labelNum):
        #     remainLabels.extend(labelIdxDict[labelNameList[labelIdx]][labelIdxPointer[labelIdx]:])
        for labelIdx in range(labelNum):
            remainLabels.extend(labelIdxDict[labelIdx])
        # print(Total_labelList[remainLabels])
        # yuyuyu
        remainLabels_class = []
        for target_class in range(num_class):
            tmp = np.where(np.array(Total_labelList[remainLabels])==target_class)[0].tolist()
            remainLabels_class.append(len(tmp))
        print(np.sum(remainLabels_class))
        print(remainLabels_class)
        # rng.shuffle(remainLabels)
        for partPointer in range(len(partitions)):

            for target_class in range(num_class):
                idxs_target_class = np.where(np.array(Total_labelList[remainLabels])==target_class)[0].tolist()
                num_sample_of_each_class = int(remainLabels_class[partPointer]*sample_probility[target_class])
                remainLabels_array = np.array(remainLabels)
                partitions[partPointer].extend(remainLabels_array[idxs_target_class[0:num_sample_of_each_class]].tolist())

                remainLabels = np.delete(np.array(remainLabels), idxs_target_class[0:num_sample_of_each_class]).tolist()
            rng.shuffle(partitions[partPointer])
        return partitions, server_idxs

    def getNonIIDdata_emnist_with_attacker(self, data, args):
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

        attacker_idxs = []
        for target_class in range(num_class):
            idxs_target_class = np.where(np.array(labelList)==target_class)[0].tolist()
            np.random.shuffle(idxs_target_class)
            idx_target_class_list.append(idxs_target_class)
            ### distribute sample of each class to the server
            server_idxs += idx_target_class_list[target_class][0:num_sample_of_each_class_server]
            attacker_idxs += idx_target_class_list[target_class][num_sample_of_each_class_server:num_sample_of_each_class_server+args.attack_num_data//num_class]
        np.random.shuffle(server_idxs)
        np.random.shuffle(attacker_idxs)

        used_idx = server_idxs + attacker_idxs
        ######
        # labelList = rest_idxs
        rng = Random()
        rng.seed(2020)

        a = []
        b = []
        devices_labelid = []
        for id in range(len(Total_labelList)):
            if id not in used_idx:
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
        # sizes = number of nodes
        # sizes = args.num_devices
        base_size = 47#args.num_devices
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
                    # interval += 1
            # print('requiredLabelList',requiredLabelList)
            for labelIdx in requiredLabelList:

                idxIncrement = int(sample_probility_[labelIdx]*basicLabelRatio)

                partitions[partPointer].extend(labelIdxDict[labelIdx][0:idxIncrement])

                labelIdxDict[labelIdx][0:idxIncrement] = []


        #random part
        remainLabels = list()
        # for labelIdx in range(labelNum):
        #     remainLabels.extend(labelIdxDict[labelNameList[labelIdx]][labelIdxPointer[labelIdx]:])
        for labelIdx in range(labelNum):
            remainLabels.extend(labelIdxDict[labelIdx])
        # print(Total_labelList[remainLabels])
        # yuyuyu
        remainLabels_class = []
        for target_class in range(num_class):
            tmp = np.where(np.array(Total_labelList[remainLabels])==target_class)[0].tolist()
            remainLabels_class.append(len(tmp))
        print(np.sum(remainLabels_class))
        print(remainLabels_class)
        # rng.shuffle(remainLabels)

        for partPointer in range(len(partitions)):

            for target_class in range(num_class):
                idxs_target_class = np.where(np.array(Total_labelList[remainLabels])==target_class)[0].tolist()
                num_sample_of_each_class = int(remainLabels_class[partPointer]*sample_probility[target_class])
                remainLabels_array = np.array(remainLabels)
                partitions[partPointer].extend(remainLabels_array[idxs_target_class[0:num_sample_of_each_class]].tolist())

                remainLabels = np.delete(np.array(remainLabels), idxs_target_class[0:num_sample_of_each_class]).tolist()
            rng.shuffle(partitions[partPointer])
        return partitions, server_idxs, attacker_idxs

    def non_iid_dist_random(self, dataset, args):
        ### Number of samples the server has
        num_sample_of_server = len(dataset) - args.num_devices*args.max_data_per_device

        import itertools
        all_class = np.arange(len(dataset.classes)).tolist()
        ### Produce all possible combinations of users' class
        all_combination = list(itertools.combinations(all_class,args.class_per_device))
        np.random.shuffle(all_combination)  ### Shuffle the order of these combinations

        idx_target_class_list = []
        ### Divide the samples of each class equally for the server
        num_sample_of_each_class_server = int(num_sample_of_server/len(dataset.classes))
        server_idxs = []
        rest_idxs = []
        for target_class in range(len(dataset.classes)):
            idxs_target_class = np.where(np.array(dataset.targets)==target_class)[0].tolist()
            np.random.shuffle(idxs_target_class)
            idx_target_class_list.append(idxs_target_class)
            ### distribute sample of each class to the server
            server_idxs += idx_target_class_list[target_class][0:num_sample_of_each_class_server]
            ### record rest sample according to class
            rest_idxs.append(idx_target_class_list[target_class][num_sample_of_each_class_server:])

        users_idxs = [[] for i in range(args.num_devices)]
        ### distribute sample of each class to users
        for user_id in range(args.num_devices):
            if user_id>=len(all_combination):
                combination_id = user_id%len(all_combination)
            else:
                combination_id = user_id
            user_class = all_combination[combination_id]
            user_idxs_tmp = []
            for user_class_i in list(user_class):
                np.random.shuffle(rest_idxs[user_class_i])
                user_idxs_tmp += rest_idxs[user_class_i][0:int(args.max_data_per_device/args.class_per_device)]
            users_idxs[user_id] = user_idxs_tmp

        return users_idxs,server_idxs

    def getNonIIDdata(self, data, args):
        from random import Random
        from math import ceil
        labelList = data.targets
        num_sample_of_server = args.num_data_server#4000#len(data) - args.num_devices*args.max_data_per_device


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
        # labelList = rest_idxs
        rng = Random()
        rng.seed(2020)
        # a = [(label, idx) for idx, label in enumerate(labelList)]
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
        # sizes = number of nodes
        # sizes = args.num_devices
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
                    # interval += 1
            for labelIdx in requiredLabelList:
                # print(len(labelIdxPointer), labelIdx)
                start = labelIdxPointer[labelIdx]
                idxIncrement = int(eachPartitionLen*basicLabelRatio)#int(basicLabelRatio*len(labelIdxDict[labelNameList[labelIdx]]))
                # print(eachPartitionLen, int(eachPartitionLen*basicLabelRatio), idxIncrement, len(labelIdxDict[labelNameList[labelIdx]][start:start+ idxIncrement]))
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
            # print(eachPartitionLen)
            idxIncrement = eachPartitionLen - len(partitions[partPointer])
            # print(idxIncrement)
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


        # server_idxs = list(set(all_index).difference(set(a)))
        return partitions,server_idxs

    def getNonIIDdata_svhn(self, data, args):
        from random import Random
        from math import ceil

        labelList = copy.deepcopy(data.labels) - 1
        num_sample_of_server = args.num_data_server#4000#len(data) - args.num_devices*args.max_data_per_device
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
        # labelList = rest_idxs
        rng = Random()
        rng.seed(2020)
        # a = [(label, idx) for idx, label in enumerate(labelList)]
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
        # sizes = number of nodes
        sizes = args.num_devices
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
                    # interval += 1
            for labelIdx in requiredLabelList:
                # print(len(labelIdxPointer), labelIdx)
                start = labelIdxPointer[labelIdx]
                idxIncrement = int(eachPartitionLen*basicLabelRatio)#int(basicLabelRatio*len(labelIdxDict[labelNameList[labelIdx]]))
                # print(eachPartitionLen, int(eachPartitionLen*basicLabelRatio), idxIncrement, len(labelIdxDict[labelNameList[labelIdx]][start:start+ idxIncrement]))
                partitions[partPointer].extend(labelIdxDict[labelNameList[labelIdx]][start:start+ idxIncrement])
                length_tmp = len(labelIdxDict[labelNameList[labelIdx]][start:start+ idxIncrement])
                tmp = np.arange(start).tolist()
                #random.shuffle(tmp)
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
            # print(eachPartitionLen)
            idxIncrement = eachPartitionLen - len(partitions[partPointer])
            # print(idxIncrement)
            partitions[partPointer].extend(remainLabels[:idxIncrement])
            rng.shuffle(partitions[partPointer])
            remainLabels = remainLabels[idxIncrement:]


        # server_idxs = list(set(all_index).difference(set(a)))
        return partitions,server_idxs

def Generate_device_server_index(args, path_device_idxs):
    utils = Utils()
    if args.attack:
        train_dataset, test_dataset, device_idxs, server_idxs, attacker_idxs = utils.get_dataset_dist(args)
        dictionary1 = {'attacker_idxs':attacker_idxs}
        np.save(path_device_idxs+"attacker_idxs.npy", dictionary1)
        # print('attacker_idxs:',attacker_idxs)
    else:
        train_dataset, test_dataset, device_idxs, server_idxs = utils.get_dataset_dist(args) ##### generate the data indexes of the users and the server

    dictionary1 = {'device_idxs':device_idxs}
    np.save(path_device_idxs+"device_idxs.npy", dictionary1)

    dictionary2 = {'server_idxs':server_idxs}
    np.save(path_device_idxs+"server_idxs.npy", dictionary2)

def Load_device_server_index(args, path_device_idxs):
    #### load data index of the users and the server
    device_ids = np.load(path_device_idxs + 'device_idxs' + '.npy', allow_pickle=True).item()
    server_idxs = np.load(path_device_idxs + 'server_idxs' + '.npy', allow_pickle=True).item()
    device_ids = device_ids['device_idxs']
    server_idxs = server_idxs['server_idxs']

    if args.attack:
        attacker_idxs = np.load(path_device_idxs + 'attacker_idxs' + '.npy', allow_pickle=True).item()
        attacker_idxs = attacker_idxs['attacker_idxs']
        return server_idxs, device_ids, attacker_idxs
    else:
        return server_idxs, device_ids

def Generate_communicate_user_list(args, path_device_idxs):
    ue_list_epoch = [[]]*args.epoch
    iterations_epoch = args.k_img//args.bs

    ue_list_epoch = np.zeros((args.epoch,iterations_epoch,args.num_comm_ue+1),dtype='int32')

    if args.num_comm_ue <= args.size - 1:
        # ue_list = np.arange(1, args.size).tolist()
        # connected_user_list = random.sample(ue_list,args.num_comm_ue)
        for e in range(args.epoch):
            for it in range(iterations_epoch):
                ue_list = np.arange(1, args.size).tolist()
                connected_user_list = random.sample(ue_list, args.num_comm_ue)
                ue_list = np.arange(1, args.size).tolist()
                random.shuffle(ue_list)
                ue_list = [0] + connected_user_list
                ue_list.sort()
                ue_list = np.array(ue_list, dtype='int32')
                ue_list_epoch[e,it,0:len(ue_list)] = ue_list

        ue_list_epoch = np.array(ue_list_epoch, dtype='int32')
        io.savemat(path_device_idxs+'ue_list_epoch.mat', {'ue_list_epoch': ue_list_epoch})
        dictionary1 = {'ue_list_epoch':ue_list_epoch.tolist()}
        np.save(path_device_idxs + "ue_list_epoch.npy", dictionary1)

def Load_communicate_user_list(args, path_device_idxs):
    if args.num_comm_ue <= args.size - 1:
        ue_list_epoches = np.load(path_device_idxs + 'ue_list_epoch' + '.npy', allow_pickle=True).item()
        ue_list_epoches = ue_list_epoches['ue_list_epoch']
    else:
        ue_list_epoches = []

    return ue_list_epoches

def Generate_Train_data_loader(args, labeled_dataset, unlabeled_dataset, train_sampler, attack_dataset=None):
    Train_data_loader = []
    if args.attack:
        for rank in range(args.size):
            Data_Loader = Assign_Train_data_loader(args, rank, labeled_dataset, unlabeled_dataset, train_sampler, attack_dataset)
            Train_data_loader.append(Data_Loader)
    else:
        for rank in range(args.size):
            Data_Loader = Assign_Train_data_loader(args, rank, labeled_dataset, unlabeled_dataset, train_sampler)
            Train_data_loader.append(Data_Loader)

    return Train_data_loader

def Generate_Test_data_loader(args, test_dataset, base_dataset, device_ids):
    Test_data_loader = []
    for rank in range(args.size):
        Data_Loader = Assign_Test_data_loader(args, rank, test_dataset, base_dataset, device_ids)
        Test_data_loader.append(Data_Loader)
    return Test_data_loader

def Assign_Train_data_loader(args, rank, labeled_dataset, unlabeled_dataset, train_sampler, attack_dataset=None):
    if attack_dataset is not None:
        if rank > 0 and rank < args.size-1:
            unlabeled_trainloader = DataLoader(
                unlabeled_dataset[rank-1],
                # sampler=train_sampler(unlabeled_dataset[rank-1]),
                batch_size=args.bs,
                num_workers=8,
                shuffle = True,
                drop_last=True)
            return unlabeled_trainloader
        else:
            if rank == 0:
                labeled_trainloader = DataLoader(
                    labeled_dataset,
                    # sampler=train_sampler(labeled_dataset),
                    shuffle = True,
                    batch_size=args.bs,
                    num_workers=8,
                    drop_last=True)
                return labeled_trainloader
            else:
                attack_trainloader = DataLoader(
                    attack_dataset,
                    # sampler=train_sampler(labeled_dataset),
                    shuffle = True,
                    batch_size=args.bs,
                    num_workers=8,
                    drop_last=True)
                return attack_trainloader

    else:
        if rank > 0:
            unlabeled_trainloader = DataLoader(
                unlabeled_dataset[rank-1],
                # sampler=train_sampler(unlabeled_dataset[rank-1]),
                batch_size=args.bs,
                num_workers=8,
                shuffle = True,
                drop_last=True)
            return unlabeled_trainloader
        else:
            labeled_trainloader = DataLoader(
                labeled_dataset,
                # sampler=train_sampler(labeled_dataset),
                shuffle = True,
                batch_size=args.bs,
                num_workers=8,
                drop_last=True)
            return labeled_trainloader


def Assign_Test_data_loader(args, rank, test_dataset, base_dataset, device_ids):
    Server_test_loader = DataLoader(test_dataset,
                             batch_size=args.bs,
                             shuffle=False)
    ### generate test loader for users

    targets = base_dataset.targets
    # print(len(set(targets)))
    if args.dataset == 'cifar10':
        num_calss = 10
    if args.dataset == 'emnist':
        num_calss = 47
    # ALLusers_test_index = []
    probability_all = []
    if rank > 0 and rank != args.size - args.attack:
        num = rank - 1

        user_targets = np.array(targets)[device_ids[num]]
        num_each_class=[]
        for i in range(num_calss):
            num_class_i = np.where(user_targets==i)
            num_each_class.append(num_class_i[0].shape[0])

        probability_all.append(num_each_class/np.sum(num_each_class))
        user_test_idxs=[]
        idx_target_class_list = []

        for target_class in range(num_calss):
            idxs_target_class = np.where(np.array(test_dataset.targets)==target_class)[0].tolist()
            np.random.shuffle(idxs_target_class)
            idx_target_class_list.append(idxs_target_class)
            # num_sample_of_each_class_server = int(num_each_class[target_class]/np.sum(num_each_class)*len(test_dataset.targets))
            num_sample_of_each_class_server = int(probability_all[0][target_class]*len(idxs_target_class))
            ### distribute sample of each class to the server
            user_test_idxs += idx_target_class_list[target_class][0:num_sample_of_each_class_server]

        test_user_sampler = torch.utils.data.sampler.SubsetRandomSampler(user_test_idxs)
        User_test_loader = torch.utils.data.DataLoader(dataset=test_dataset,sampler=test_user_sampler,num_workers=2,
                                                   batch_size=args.bs)
    if rank > 0 and rank != args.size - args.attack:
        return User_test_loader
    else:
        return Server_test_loader

def Save_acc_file(args, rank, prefix=None, acc_list=None):
    filename = "./results_v0/%s_Rank%s_%s_%s_iid%s_UE%s_%s_comUE%s_%s_bs%s_cp%s.txt" %(prefix, rank, args.experiment_name, args.dataset, args.iid,
                                                                                        args.size - 1, args.basicLabelRatio,
                                                                                        args.num_comm_ue, args.model, args.bs, args.cp)
    if filename:
        with open(filename, 'w') as f:
            json.dump(acc_list, f)

def Save_model(experiment_name, model, rank, epoch):
    path_checkpoint = "./checkpoint/%s/" %(experiment_name)
    if not os.path.exists(path_checkpoint):
        os.makedirs(path_checkpoint)
    if rank == 0:
        torch.save(model.state_dict(), path_checkpoint+'Rank%s_Epoch_%s_weights.pth' %(rank, epoch))

def Load_model_weights(experiment_name, epoch):
    path_checkpoint = './checkpoint/%s/' %(experiment_name)
    pthfile = path_checkpoint+'Rank0_Epoch_%s_weights.pth' %(epoch)
    checkpoint_weights = torch.load(pthfile)
    return checkpoint_weights

def Assign_ranks_to_threads(args):
    User_list = np.arange(args.num_comm_ue).tolist()#[i for i in args.num_comm_ue]
    increase_tmp = (args.size - 1)//len(User_list)
    ranks_list = np.arange(0, args.size - 1).tolist()

    rank_group = []
    for rank_id in range(len(User_list)):
        if rank_id == len(User_list)-1:
            ranks = ranks_list[rank_id*increase_tmp:]
        else:
            ranks = ranks_list[rank_id*increase_tmp:(rank_id+1)*increase_tmp]
        rank_group.append(ranks)

    return rank_group

def Test_each_model_with_FedAvg_weights(args, user_group, model, test_loader_list):
    for n in range(len(user_group)):
        Users_acc = []

        Users_acc1 = []
        Users_acc2 = []
        for i in range(args.epoch):
            checkpoint_weights = Load_model_weights(args.experiment_name, i)
            model.load_state_dict(checkpoint_weights)

            if args.attack:
                acc1, acc2 = test_with_attack(args, model, test_loader_list[user_group[n] + 1])
                Users_acc1.append(round(acc1*100.0,2))
                Users_acc2.append(round(acc2*100.0,2))
            else:
                acc = test(model, test_loader_list[user_group[n] + 1])
                Users_acc.append(round(acc*100.0,2))
        if args.attack:
            Save_acc_file(args, user_group[n] + 1, prefix=f'FedAvg_benign_attack{args.attack}_{args.attack_class}_{args.attack_target}', acc_list=Users_acc1)
            Save_acc_file(args, user_group[n] + 1, prefix=f'FedAvg_attack{args.attack}_{args.attack_class}_{args.attack_target}', acc_list=Users_acc2)
        else:
            Save_acc_file(args, user_group[n] + 1, prefix='FedAvg', acc_list=Users_acc)



class Sampler(Dataset):

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs

    def __getitem__(self, idx):
        return self.dataset[self.idxs[idx]]

    def __len__(self):
        return len(self.idxs)

class Sampler_no_target(Dataset):

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs

    def __getitem__(self, idx):

        return self.dataset[self.idxs[idx]][0]

    def __len__(self):
        return len(self.idxs)

class Sampler_distill(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.trans = T.Compose([
                T.ToTensor(),
            ])

    def __getitem__(self, idx):
        # tensor2np = self.dataset[idx][0].cpu().numpy()
        np2tensor = self.trans(self.dataset[idx][0])
        # print('size',np2tensor.size())
        # np2tensor = torch.from_numpy(tensor2np)
        # np2tensor = self.dataset[idx][0]
        return (np2tensor, self.dataset[idx][1])

    def __len__(self):
        return len(self.dataset)


def fed_avg(weights):
    w = copy.deepcopy(weights[0])   # Weight from first device
    for key in w.keys():
        for i in range(1, len(weights)):    # Other devices
            w[key] += weights[i][key]   # Sum up weights
        w[key] = torch.div(w[key], len(weights))    # Get average weights
    return w

def fed_weight_avg(weights,user_weight):
    w = copy.deepcopy(weights[0])   # Weight from first device
    for key in w.keys():
        w[key] = torch.div(w[key], 1/user_weight[0])
    for key in w.keys():
        for i in range(1, len(weights)):    # Other devices
            w[key] += torch.div( weights[i][key], 1/user_weight[i])

    return w

def weight_diff(weights, layer_name):
    w_1 = copy.deepcopy(weights[0])   # Weight from first device
    w_2 = copy.deepcopy(weights[0])
    w_avg = copy.deepcopy(weights[-1])
    layer_name_1 = layer_name + '.weight'
    layer_name_2 = layer_name + '.bias'
    for i in range(len(weights)-1):
        for key in w_1.keys():
            print(key)
            if key == layer_name_1 or key == layer_name_2:
                weights[i][key] = weights[i][key] - w_avg[key]
                # print(weights[i][key].max(),weights[i][key].min())
    w_1 = copy.deepcopy(weights[0])   # Weight from first device
    w_2 = copy.deepcopy(weights[0])

    W_2_list = []
    for key in w_2.keys():
        # print(key)
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
# def fed_weight_avg(weights,user_weight):
#     w = copy.deepcopy(weights[0])   # Weight from first device
#     for id in range(len(weights)):
#         for key in w.keys():
#             weights[id][key] = weights[id][key]*user_weight[id]
#     w = copy.deepcopy(weights[0])   # Weight from first device
#     for key in w.keys():
#         for i in range(1, len(weights)):    # Other devices
#             w[key] += weights[i][key]   # Sum up weights
#     return w

def cal_avg_weight_diff(weight_1, weight_2,weight_2_weight=1.0):
    w1 = copy.deepcopy(weight_1)
    w2 = copy.deepcopy(weight_2)

    diff_list = []
    for key in w2.keys():
        average = np.abs(np.array(w1[key].cpu().numpy()) - np.array(w2[key].cpu().numpy())).mean()
        diff_list.append(average)
    return sum(diff_list)#/len(diff_list)



def test(model, test_loader, cuda=True):
    """
    Get the test performance
    """
    model.eval()
    correct = 0
    total_num = 0
    with torch.no_grad():
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.data.max(
                1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            total_num += len(data)
    # print('testing_correct: ', correct / total_num, '\n')
    return correct / total_num

def test_with_attack(args, model, test_loader, cuda=True):
    """
    Get the test performance
    """
    model.eval()
    correct1 = 0
    correct2 = 0
    total_num1 = 0
    total_num2 = 0
    with torch.no_grad():
        for data, target in test_loader:
            target_tmp = target - args.attack_class
            benign_index = torch.nonzero(target_tmp)
            attack_index =  torch.eq(target, args.attack_class)
            target[attack_index] = args.attack_target
            if cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.data.max(
                1, keepdim=True)[1]  # get the index of the max log-probability
            target1 = target[benign_index]
            target2 = target[attack_index]
            pred1 = pred[benign_index]
            pred2 = pred[attack_index]
            correct1 += pred1.eq(target1.data.view_as(pred1)).cpu().sum().item()

            correct2 += pred2.eq(target2.data.view_as(pred2)).cpu().sum().item()

            total_num1 += target1.size(0)
            total_num2 += target2.size(0)
    # print('testing_correct: ', correct / total_num, '\n')
    return correct1 / total_num1, correct2 / total_num2

def test_each_class(net,testloader,cuda=True):
    """
    Get the test performance of each class
    """
    net.eval()
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data, target in testloader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            outputs = net(data)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == target).squeeze()
            for i in range(target.shape[0]):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    return np.array(class_correct) / np.array(class_total)

def test_each_class_with_attack(args, net, testloader, cuda=True):
    """
    Get the test performance of each class
    """
    net.eval()
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data, target in testloader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            outputs = net(data)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == target).squeeze()
            for i in range(target.shape[0]):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    return np.array(class_correct) / np.array(class_total)
