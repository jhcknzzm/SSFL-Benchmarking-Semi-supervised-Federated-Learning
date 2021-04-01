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
from random import Random
from comm_helpers import SyncAllreduce, SyncAllreduce_1, SyncAllreduce_2, SyncAllGather

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
                                           transform=transform,split='balanced',
                                           download=True)

            test_dataset = datasets.EMNIST(root='./data/',
                                          train=False,split='balanced',
                                          transform=transform)

        if args.dataset == "cifar10":
            if args.iid:
                user_idxs,server_idxs = self.iid_dist(train_dataset, args)
                if args.user_semi:
                    user_idxs,server_idxs = user_semi_iid_dist2(dataset, args)
            else:
                if args.user_semi:
                    user_idxs,server_idxs = self.user_semi_noniid_dist2(train_dataset, args)
                else:
                    user_idxs,server_idxs = self.getNonIIDdata(train_dataset, args)

        if args.dataset == "svhn":
            if args.iid:
                user_idxs,server_idxs = self.iid_dist_svhn(train_dataset, args)
            else:
                user_idxs,server_idxs = self.getNonIIDdata_svhn(train_dataset, args)

        if args.dataset == "emnist":
            if args.iid:
                user_idxs,server_idxs = self.iid_dist_emnist(train_dataset, args)

            else:
                user_idxs,server_idxs = self.getNonIIDdata_emnist_largeK(train_dataset, args)

        return train_dataset, test_dataset, user_idxs,server_idxs

    def iid_dist(self, dataset, args):
        rng = Random()
        rng.seed(1)
        print('iid----')
        data_per_device = (len(dataset) -  args.num_data_server)//args.num_devices

        num_sample_of_server = args.num_data_server
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
        print(len(users_idxs))
        return users_idxs,server_idxs

    def user_semi_iid_dist2(self, dataset, args):
        rng = Random()
        rng.seed(1)

        data_per_device = (len(dataset) -  args.num_data_server)//args.num_devices

        num_sample_of_server = args.num_data_server
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

        users_idxs_u = [[] for i in range(args.num_devices)]  # Index dictionary for devices
        users_idxs_x = [[] for i in range(args.num_devices)]  # Index dictionary for devices
        random.shuffle(rest_idxs)
        random.shuffle(server_idxs)
        num_x = len(server_idxs)//args.num_devices

        for i in range(args.num_devices):
            users_idxs_u[i] = rest_idxs[i*data_per_device:(i+1)*data_per_device]
            users_idxs_x[i] = server_idxs[i*num_x:(i+1)*num_x]

        return users_idxs_u, users_idxs_x

    def user_semi_noniid_dist2(self, dataset, args):
        rng = Random()
        rng.seed(1)

        data_per_device = (len(dataset) -  args.num_data_server)//args.num_devices

        num_sample_of_server = args.num_data_server
        num_sample_of_each_class_server = int(num_sample_of_server/len(dataset.classes))

        server_idxs=[]
        rest_idxs = []
        idx_target_class_list = []

        labeled_class_idx = []
        server_labels = []
        for target_class in range(len(dataset.classes)):
            idxs_target_class = np.where(np.array(dataset.targets)==target_class)[0].tolist()
            np.random.shuffle(idxs_target_class)
            idx_target_class_list.append(idxs_target_class)
            ### distribute sample of each class to the server
            server_idxs += idx_target_class_list[target_class][0:num_sample_of_each_class_server]
            server_labels += [target_class for ii_lab in range(num_sample_of_each_class_server)]
            labeled_class_idx.append(idx_target_class_list[target_class][0:num_sample_of_each_class_server])
            ### record rest sample according to class
            rest_idxs += idx_target_class_list[target_class][num_sample_of_each_class_server:]


        users_idxs_u = [[] for i in range(args.num_devices)]  # Index dictionary for devices
        users_idxs_x = [[] for i in range(args.num_devices)]  # Index dictionary for devices
        random.shuffle(rest_idxs)
        for i_ue in range(args.num_devices):
            users_idxs_u[i_ue] = rest_idxs[i_ue*data_per_device:(i_ue+1)*data_per_device]

            target_classes = np.random.choice(np.arange(len(dataset.classes)),2)


            random.shuffle(labeled_class_idx[target_classes[0]])
            random.shuffle(labeled_class_idx[target_classes[1]])
            users_idxs_x[i_ue] += labeled_class_idx[target_classes[0]][0:args.num_data_server//args.num_devices//2]
            users_idxs_x[i_ue] += labeled_class_idx[target_classes[1]][0:args.num_data_server//args.num_devices//2]


            current_classs = 0
            users_classes = [[] for i in range(args.num_devices)]  # Classes dictionary for devices
            classes_devives = [[] for i in range(len(dataset.classes))]  # Devices in each class

            # Distribute class numbers to devices
            for i in range(args.num_devices):
                next_current_class = (current_classs+2)%len(dataset.classes)
                if next_current_class > current_classs:
                    users_classes[i] = np.arange(current_classs, next_current_class)
                else:
                    users_classes[i] = np.append(
                        np.arange(current_classs, len(dataset.classes)),
                        np.arange(0, next_current_class)
                    )

                for j in users_classes[i]:
                    classes_devives[j].append(i)

                current_classs = next_current_class

            # Combine indexes and labels for sorting
            idxs_labels = np.vstack((np.array(server_idxs) ,np.array(server_labels)))

            idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]

            users_idxs = [[] for i in range(args.num_devices)]  # Index dictionary for devices

            current_idx = 0
            for i in range(len(dataset.classes)):
                if not len(classes_devives[i]):
                    continue

                send_to_device = 0
                for j in range(current_idx, len(idxs_labels[0])):
                    if idxs_labels[1, j] != i:
                        current_idx = j
                        break

                    users_idxs[classes_devives[i][send_to_device]].append(idxs_labels[0, j])
                    send_to_device = (send_to_device+1)%len(classes_devives[i])



        return users_idxs_u, users_idxs

    def iid_dist_svhn(self, dataset, args):
        data_per_device = (len(dataset) -  args.num_data_server)//args.num_devices

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
        data_per_device = (len(dataset) -  args.num_data_server)//args.num_devices
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
        labelList = labelList.cpu().numpy()
        num_class = 47
        Total_labelList = data.targets
        Total_labelList = Total_labelList.cpu().numpy()


        print(labelList.shape)
        num_sample_of_server = args.num_data_server

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
        print(np.sum(remainLabels_class))
        print(remainLabels_class)

        for partPointer in range(len(partitions)):

            for target_class in range(num_class):
                idxs_target_class = np.where(np.array(Total_labelList[remainLabels])==target_class)[0].tolist()
                num_sample_of_each_class = int(remainLabels_class[partPointer]*sample_probility[target_class])
                remainLabels_array = np.array(remainLabels)
                partitions[partPointer].extend(remainLabels_array[idxs_target_class[0:num_sample_of_each_class]].tolist())

                remainLabels = np.delete(np.array(remainLabels), idxs_target_class[0:num_sample_of_each_class]).tolist()
            rng.shuffle(partitions[partPointer])
        return partitions,server_idxs

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
        num_sample_of_server = args.num_data_server


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


        if args.num_devices == 100:
            partitions_list = [list() for i in range(100)]
            for partPointer in range(10):
                index_partitions = partitions[partPointer]
                len_partitions = len(index_partitions)
                rng.shuffle(index_partitions)
                for ii in range(10-1):
                    partitions_list[partPointer+ii*10] = index_partitions[(len_partitions*ii)//10:(len_partitions*(ii+1))//10]
                partitions_list[partPointer+90] = index_partitions[len_partitions*9//10:]
            partitions = partitions_list

        return partitions, server_idxs

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

        return partitions, server_idxs

    def getNonIIDdata_emnist_largeK(self, data, args):
        from random import Random
        from math import ceil
        labelList = data.targets
        labelList = labelList.cpu().numpy()
        num_class = 47
        Total_labelList = data.targets
        Total_labelList = Total_labelList.cpu().numpy()


        print(labelList.shape)
        num_sample_of_server = args.num_data_server

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

        base_size = 47
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
        print(np.sum(remainLabels_class))
        print(remainLabels_class)

        for partPointer in range(len(partitions)):

            for target_class in range(num_class):
                idxs_target_class = np.where(np.array(Total_labelList[remainLabels])==target_class)[0].tolist()
                num_sample_of_each_class = int(remainLabels_class[partPointer]*sample_probility[target_class])
                remainLabels_array = np.array(remainLabels)
                partitions[partPointer].extend(remainLabels_array[idxs_target_class[0:num_sample_of_each_class]].tolist())

                remainLabels = np.delete(np.array(remainLabels), idxs_target_class[0:num_sample_of_each_class]).tolist()
            rng.shuffle(partitions[partPointer])


        if args.num_devices == 470:
            partitions_list = [list() for i in range(470)]
            for partPointer in range(num_class):
                index_partitions = partitions[partPointer]
                len_partitions = len(index_partitions)
                rng.shuffle(index_partitions)
                for increase_tmp in range(10):

                    partitions_list[partPointer + increase_tmp*47] = index_partitions[len_partitions*(increase_tmp)//10:len_partitions*(increase_tmp+1)//10]

            partitions = partitions_list

        return partitions, server_idxs

def Generate_device_server_index(args, path_device_idxs):
    utils = Utils()
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

    return server_idxs, device_ids

def Generate_communicate_user_list(args, path_device_idxs):

    iterations_epoch = args.k_img//args.bs

    if args.num_comm_ue == 10:
        ue_list_epoch = np.zeros((args.epoch,iterations_epoch,args.num_comm_ue+1+args.H),dtype='int32')
    if args.num_comm_ue == 30:
        ue_list_epoch = np.zeros((args.epoch,iterations_epoch,args.num_comm_ue+1+args.H*2),dtype='int32')
    if args.num_comm_ue == 47:
        ue_list_epoch = np.zeros((args.epoch,iterations_epoch,args.num_comm_ue+1+args.H*4),dtype='int32')

    if args.user_semi:
        ue_list_epoch = np.zeros((args.epoch,iterations_epoch,args.num_comm_ue),dtype='int32')

    if args.num_comm_ue == 20:
        ue_list_epoch = np.zeros((args.epoch,iterations_epoch,args.num_comm_ue+1+args.H),dtype='int32')

    if args.num_comm_ue <= args.size - 1:

        for e in range(args.epoch):
            for it in range(iterations_epoch):
                if args.user_semi:
                    ue_list = np.arange(0, args.size).tolist()
                    connected_user_list = random.sample(ue_list, args.num_comm_ue)
                    ue_list = connected_user_list
                else:
                    ue_list = np.arange(1, args.num_devices+1).tolist()
                    connected_user_list = random.sample(ue_list, args.num_comm_ue)
                    ue_list = connected_user_list
                    random.shuffle(ue_list)
                    ue_list = [0] + ue_list

                    if args.H and args.dataset == 'emnist':
                        if args.num_comm_ue == 10:
                            ue_list = ue_list + [48]

                        if args.num_comm_ue == 30:
                            ue_list = ue_list + [48,49]
                        if args.num_comm_ue == 47:
                            ue_list = ue_list + [48,49,50,51]

                    if args.H and args.dataset == 'cifar10':
                        if args.num_comm_ue == 10 and args.num_devices == 10:
                            ue_list = ue_list + [11]
                        if args.num_comm_ue == 10 and args.num_devices == 30:
                            ue_list = ue_list + [31]
                        if args.num_comm_ue == 20 and args.num_devices == 20:
                            ue_list = ue_list + [21]
                        if args.num_comm_ue == 20 and args.num_devices == 30:
                            ue_list = ue_list + [31]
                        if args.num_comm_ue == 30:
                            ue_list = ue_list + [31,32]

                    if args.H and args.dataset == 'svhn':
                        if args.num_comm_ue == 10 and args.num_devices == 10:
                            ue_list = ue_list + [11]
                        if args.num_comm_ue == 10 and args.num_devices == 30:
                            ue_list = ue_list + [31]
                        if args.num_comm_ue == 20 and args.num_devices == 20:
                            ue_list = ue_list + [21]
                        if args.num_comm_ue == 20 and args.num_devices == 30:
                            ue_list = ue_list + [31]
                        if args.num_comm_ue == 30:
                            ue_list = ue_list + [31,32]

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

def Generate_Train_data_loader(args, labeled_dataset, unlabeled_dataset, train_sampler):
    Train_data_loader = []
    if args.dataset == 'emnist':
        if args.num_comm_ue == 10:
            server_rank_list = [0,48]
        if args.num_comm_ue == 30:
            server_rank_list = [0,48,49]
        if args.num_comm_ue == 20:
            server_rank_list = [0,21]
        if args.num_comm_ue == 47:
            server_rank_list = [0,48,49,50,51]


    if args.dataset == 'svhn' or args.dataset == 'cifar10':
        if args.num_comm_ue == 10 and args.num_devices == 10:
            server_rank_list = [0,11]

        if args.num_comm_ue == 10 and args.num_devices == 20:
            server_rank_list = [0,21]

        if args.num_comm_ue == 20 and args.num_devices == 20:
            server_rank_list = [0,21]

        if args.num_comm_ue == 10 and args.num_devices == 30:
            server_rank_list = [0,31]

        if args.num_comm_ue == 20 and args.num_devices == 30:
            server_rank_list = [0,31]
        if args.num_comm_ue == 30 and args.num_devices == 30:
            server_rank_list = [0,31,32]


    k = 0
    for rank in range(args.size):
        if rank in set(server_rank_list):
            Data_Loader = Assign_Train_data_loader(args, rank, labeled_dataset, unlabeled_dataset, train_sampler, k=0)
        else:
            k = k+1
            Data_Loader = Assign_Train_data_loader(args, rank, labeled_dataset, unlabeled_dataset, train_sampler, k)


        Train_data_loader.append(Data_Loader)
    return Train_data_loader

def Generate_Train_data_loader_user_side_semi(args, labeled_dataset, unlabeled_dataset, test_dataset):
    Train_data_loader = []

    k = 0
    for rank in range(args.size):
        labeled_trainloader = DataLoader(
            labeled_dataset[rank],
            # sampler=train_sampler(labeled_dataset),
            shuffle = True,
            batch_size=args.bs,
            num_workers=4,
            drop_last=True)

        unlabeled_trainloader = DataLoader(
            unlabeled_dataset[rank],
            # sampler=train_sampler(unlabeled_dataset[rank-1]),
            batch_size=args.bs,
            num_workers=4,
            shuffle = True,
            drop_last = True)

        Train_data_loader.append([labeled_trainloader,unlabeled_trainloader])

        test_loader = DataLoader(test_dataset,
                                 batch_size=args.bs,
                                 shuffle=False,num_workers=4)

    return Train_data_loader, [test_loader]

def Generate_Test_data_loader(args, test_dataset, base_dataset, device_ids):
    Test_data_loader = []
    for rank in range(args.size):
        Data_Loader = Assign_Test_data_loader(args, rank, test_dataset, base_dataset, device_ids)
        Test_data_loader.append(Data_Loader)
    return Test_data_loader

def Assign_Train_data_loader(args, rank, labeled_dataset, unlabeled_dataset, train_sampler, k):

    if k > 0:
        unlabeled_trainloader = DataLoader(
            unlabeled_dataset[k-1],
            # sampler=train_sampler(unlabeled_dataset[rank-1]),
            batch_size=args.bs,
            num_workers=4,
            shuffle = True,
            drop_last = True)
        return unlabeled_trainloader

    else:
        labeled_trainloader = DataLoader(
            labeled_dataset,
            # sampler=train_sampler(labeled_dataset),
            shuffle = True,
            batch_size=args.bs,
            num_workers=4,
            drop_last=True)
        return labeled_trainloader


def Assign_Test_data_loader(args, rank, test_dataset, base_dataset, device_ids):
    Server_test_loader = DataLoader(test_dataset,
                             batch_size=args.bs,
                             shuffle=False,num_workers=4)
    ### generate test loader for users
    if args.dataset == 'svhn':
        targets = base_dataset.labels
    else:
        targets = base_dataset.targets

    if args.dataset == 'cifar10' or args.dataset == 'svhn':
        num_calss = 10
    if args.dataset == 'emnist':
        num_calss = 47

    probability_all = []

    return Server_test_loader

def Save_acc_file(args,save_path, rank, prefix=None, acc_list=None):
    filename = "%s/%s_Rank%s_%s_%s_iid%s_UE%s_%s_comUE%s_%s_bs%s_cp%s.txt" %(save_path, prefix, rank, args.experiment_name, args.dataset, args.iid,
                                                                                        args.size - 1, args.basicLabelRatio,
                                                                                        args.num_comm_ue, args.model, args.bs, args.cp)
    if filename:
        with open(filename, 'w') as f:
            json.dump(acc_list, f)

def get_acc(args,save_path, rank, prefix=None):
    path = "%s/%s_Rank%s_%s_%s_iid%s_UE%s_%s_comUE%s_%s_bs%s_cp%s.txt" %(save_path, prefix, rank, args.experiment_name, args.dataset, args.iid,
                                                                                        args.size - 1, args.basicLabelRatio,
                                                                                        args.num_comm_ue, args.model, args.bs, args.cp)
    data_center = []
    with open(path, 'r') as f:
        for i in f:
            if i[0] == '[':
                i = i[1:]

            if i[-1] == ']':
                i = i[0:-1]

            data_center.append([float(j.rstrip(','))   for j in i.split()])

    data_center = data_center[0]
    data_center = data_center[0:]
    return  data_center

def init_files(args,save_path, rank, prefix=None):
    filename = "%s/%s_Rank%s_%s_%s_iid%s_UE%s_%s_comUE%s_%s_bs%s_cp%s.txt" %(save_path, prefix, rank, args.experiment_name, args.dataset, args.iid,
                                                                                        args.size - 1, args.basicLabelRatio,
                                                                                        args.num_comm_ue, args.model, args.bs, args.cp)
    if os.path.exists(filename):
        os.remove(filename)
    else:
        print('Start recording...')

def Save_model(experiment_name, model, rank, epoch):
    path_checkpoint = "./checkpoint/%s/" %(experiment_name)
    if not os.path.exists(path_checkpoint):
        os.makedirs(path_checkpoint)
    if rank == 0:
        torch.save(model.state_dict(), path_checkpoint+'Rank%s_Epoch_%s_weights.pth' %(rank, epoch))

def Save_model_checkpoint(experiment_name, model, rank, epoch):
    path_checkpoint = "./checkpoint/%s/" %(experiment_name)
    if not os.path.exists(path_checkpoint):
        os.makedirs(path_checkpoint)
    torch.save(copy.deepcopy(model.state_dict()), path_checkpoint+'Rank%s_Epoch_%s_weights.pth' %(rank, epoch))

def Save_model_grad_checkpoint(experiment_folders, experiment_name, model, rank, epoch, tao):
    path_checkpoint = "%s/grad_checkpoint/%s/" %(experiment_folders, experiment_name)
    if not os.path.exists(path_checkpoint):
        os.makedirs(path_checkpoint)
    tmp_grad = get_grads(model)
    torch.save(copy.deepcopy(tmp_grad), path_checkpoint+'Rank%s_Epoch_%s_model_grads_tao_%s.pth' %(rank, epoch, tao))

def Save_train_state(experiment_folders, experiment_name, rank, epoch, values, tao):
    path_checkpoint = "%s/state/%s/" %(experiment_folders, experiment_name)
    if not os.path.exists(path_checkpoint):
        os.makedirs(path_checkpoint)
    np.save(path_checkpoint + 'Rank%s_Epoch_%s_model_stats_tao_%s.npy' %(rank, epoch, tao), values)

def Load_train_state(experiment_folders, experiment_name, rank, epoch, tao):
    path_checkpoint = "%s/state/%s/" %(experiment_folders, experiment_name)
    if not os.path.exists(path_checkpoint):
        os.makedirs(path_checkpoint)
    values = np.load(path_checkpoint + 'Rank%s_Epoch_%s_model_stats_tao_%s.npy' %(rank, epoch, tao), allow_pickle=True)
    return values

def Save_Avg_model_checkpoint(experiment_name, model_weights, rank, epoch, prefix):
    path_checkpoint = "./checkpoint/%s/" %(experiment_name)
    if not os.path.exists(path_checkpoint):
        os.makedirs(path_checkpoint)
    torch.save(model_weights, path_checkpoint+'Avg_%s_Epoch_%s_weights.pth' %(prefix, epoch))

def Load_model_weights(experiment_name, epoch):
    path_checkpoint = './checkpoint/%s/' %(experiment_name)
    pthfile = path_checkpoint+'Rank0_Epoch_%s_weights.pth' %(epoch)
    checkpoint_weights = torch.load(pthfile)
    return checkpoint_weights

def Load_Avg_model_checkpoint(experiment_folders, experiment_name, epoch, prefix):
    path_checkpoint = '%s/checkpoint/%s/' %(experiment_folders, experiment_name)
    pthfile = path_checkpoint + 'Avg_%s_Epoch_%s_weights.pth' %(prefix, epoch)
    checkpoint_weights = torch.load(pthfile)
    return checkpoint_weights



def Assign_ranks_to_threads(args):
    User_list = np.arange(args.num_comm_ue).tolist()
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
        for i in range(args.epoch):
            checkpoint_weights = Load_model_weights(args.experiment_name, i)
            model.load_state_dict(checkpoint_weights)
            acc = test(model, test_loader_list[user_group[n] + 1])
            Users_acc.append(round(acc*100.0,2))
        Save_acc_file(args, user_group[n] + 1, prefix='FedAvg', acc_list=Users_acc)

def Get_SF_train_test_dataloader(device_ids, server_idxs, args):
    data_dir = "./data"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    len_dataset = []
    for id in range(len(device_ids)):
        len_dataset.append(len(device_ids[id]))

    max_len = np.max(len_dataset)
    print('max_len',max_len)

    server_idxs = list(server_idxs)
    print(len(server_idxs))
    if len(server_idxs)<max_len:
        tmp = np.arange(len(server_idxs)).tolist()
        exam_choice_list = [random.choice(tmp) for _ in range(max_len-len(server_idxs))]
        server_idxs += exam_choice_list
    print(len(server_idxs))


    server_sampler = torch.utils.data.sampler.SubsetRandomSampler(server_idxs)
    labeled_trainloader = torch.utils.data.DataLoader(dataset=train_dataset,sampler=server_sampler,num_workers=4,
                                               batch_size=args.bs)
    print(len(server_idxs)//args.bs)

    unlabeled_trainloader_list = []

    train_loader = []
    train_loader.append(labeled_trainloader)
    for id in range(len(device_ids)):
        server_sampler = torch.utils.data.sampler.SubsetRandomSampler(device_ids[id])
        unlabeled_trainloader = torch.utils.data.DataLoader(dataset=train_dataset,sampler=server_sampler,num_workers=4,
                                                   batch_size=args.bs)
        unlabeled_trainloader_list.append(unlabeled_trainloader)
        train_loader.append(unlabeled_trainloader)

    test_loader = DataLoader(test_dataset,
                             batch_size=args.bs,
                             shuffle=False)

    return train_loader, [test_loader], max_len

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

        np2tensor = self.trans(self.dataset[idx][0])
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

def cal_avg_weight_diff(weights_list, avg_weights):
        w = copy.deepcopy(weights_list)
        w2 = copy.deepcopy(avg_weights)

        key = list(w2.keys())

        for key in list(w2.keys()):
            w2[key] = w2[key].reshape(-1).tolist()    # Reshape to 1d tensor and transform to list

        # List for differences: for all devices, get the average of abs((val(device)-val(average))/val(average))
        diff_list = []

        for key in list(w2.keys()):

            tmp2 = []
            for i in range(len(w)):
                tmp = []
                w[i][key] = w[i][key].reshape(-1).tolist()    # Reshape to 1d tensor and transform to list

                for j in range(len(w[i][key])):
                    tmp.append(abs((w[i][key][j]-w2[key][j])/(w2[key][j]+1e-5)))   # Abs((val(device)-val(average))/val(average))


                average = sum(tmp)/len(tmp) # Calculate average
                tmp2.append(average)

            average = sum(tmp2)/len(tmp2)
            diff_list.append(average)

        return sum(diff_list)/len(diff_list)

def cal_avg_weight_diff_v1(weights_list, avg_weights, args):
        w = copy.deepcopy(weights_list)
        w2 = copy.deepcopy(avg_weights)

        del weights_list
        del avg_weights

        key = list(w2.keys())

        for key in list(w2.keys()):
            w2[key] = w2[key].reshape(-1).cpu().tolist()     # Reshape to 1d tensor and transform to list


        diff_list = []
        WD_key_user = []
        for key in list(w2.keys()):
            WD_key = []

            if 'num_batches_tracked' in key or 'running' in key or 'bn' in key:
                pass

            else:
                if args.only_linear:
                    if 'linear' in key or 'fc' in key:
                        calculate = True
                    else:
                        calculate = False
                else:
                    calculate = True

                if calculate == True:
                    tmp2 = []
                    tmp_sum = 0
                    wd_i_sum = 0
                    for i in range(len(w)):
                        tmp = []
                        w[i][key] = w[i][key].reshape(-1).cpu().tolist()

                        if args.method == "weight_diversity" or args.method == "normalization":
                            for j in range(len(w[i][key])):
                                tmp.append(w[i][key][j]-w2[key][j])

                            tmp = np.array(tmp)
                            tmp = torch.from_numpy(tmp).type(torch.FloatTensor)
                            tmp_sum += tmp
                            wd_i_sum += torch.norm(tmp).item()


                    if args.method == "weight_diversity":
                        denominator = torch.norm(tmp_sum)
                    elif args.method == "normalization":
                        denominator = wd_i_sum


                    for i in range(len(w)):
                        tmp = []
                        tmp_avg = []
                        for j in range(len(w[i][key])):
                            tmp.append(w[i][key][j]-w2[key][j])
                            tmp_avg.append(w2[key][j])

                        tmp = np.array(tmp)
                        tmp_avg = np.array(tmp_avg)

                        tmp = torch.from_numpy(tmp).type(torch.FloatTensor)
                        tmp_avg = torch.from_numpy(tmp_avg).type(torch.FloatTensor)


                        if args.method == "weight_diversity" or args.method == "normalization":
                            wd_i = torch.norm(tmp)/denominator
                        elif args.method == 'relative':
                            wd_i = torch.norm(tmp)/torch.norm(tmp_avg)
                        else:
                            wd_i = torch.norm(tmp)#/denominator

                        if args.norm == 'l2':
                            WD_key.append(wd_i.item()**2)
                        else:
                            WD_key.append(wd_i.item())

                    WD_key_user.append(WD_key)

        return WD_key_user

def weight_diff(weights, layer_name):
    w_1 = copy.deepcopy(weights[0])
    w_2 = copy.deepcopy(weights[0])
    w_avg = copy.deepcopy(weights[-1])
    layer_name_1 = layer_name + '.weight'
    layer_name_2 = layer_name + '.bias'
    for i in range(len(weights)-1):
        for key in w_1.keys():

            if key == layer_name_1 or key == layer_name_2:
                weights[i][key] = weights[i][key] - w_avg[key]

    w_1 = copy.deepcopy(weights[0])
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
    return correct / total_num

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


def get_grads(model):
    # wrt data at the current step
    res = []
    for name, parms in model.named_parameters():
        if 'num_batches_tracked' in name or 'running' in name or 'bn' in name:
            pass
        else:
            res.append(parms.grad.view(-1))

    grad_flat = torch.cat(res)

    return grad_flat

def get_params(w, WBN=False):
    # wrt data at the current step
    res = []
    for key in list(w.keys()):
        if 'num_batches_tracked' in key or 'running' in key:
            pass
        else:
            if 'bn' in key and WBN:
                w[key] = w[key].reshape(-1).cpu()
                res.append(w[key])
            else:
                w[key] = w[key].reshape(-1).cpu()
                res.append(w[key])

    weight_flat = torch.cat(res)
    return weight_flat

def get_groups(args):
    if args.dataset == 'emnist':
        if args.num_comm_ue == 10:
            server_list = [0,48]
            group1 = [0] + np.arange(1, 6).tolist()
            group2 = [11] + np.arange(6, 11).tolist()
            group3 = [0, 11]
            groups = [group1, group2, group3]
        if args.num_comm_ue == 30:
            server_list = [0,48,49]
            group1 = [0] + np.arange(1, 11).tolist()
            group2 = [31] + np.arange(11, 21).tolist()
            group3 = [32] + np.arange(21, 31).tolist()
            group4 = [0,31,32]
            groups = [group1, group2, group3, group4]
        if args.num_comm_ue == 47:
            server_list = [0,48,49,50,51]
            group1 = [0] + np.arange(1, 11).tolist()
            group2 = [48] + np.arange(11, 21).tolist()
            group3 = [49] + np.arange(21, 31).tolist()
            group4 = [50] + np.arange(31, 41).tolist()
            group5 = [51] + np.arange(41, 48).tolist()
            group6 = [0, 48, 49, 50, 51]
            groups = [group1, group2, group3, group4, group5, group6]

    if args.dataset == 'cifar10' or args.dataset == 'svhn':
        if args.user_semi:
            server_list = [0,11]
            group1 = np.arange(0, 5).tolist()
            group2 = np.arange(5, 10).tolist()
            group3 = [0, 9]
            groups = [group1, group2, group3]

        if args.num_comm_ue == 10 and args.num_devices == 10:
            server_list = [0,11]
            group1 = [0] + np.arange(1, 6).tolist()
            group2 = [11] + np.arange(6, 11).tolist()
            group3 = [0, 11]
            groups = [group1, group2, group3]

        if args.num_comm_ue == 10 and args.num_devices == 30:
            server_list = [0,31]
            group1 = [0] + np.arange(1, 6).tolist()
            group2 = [11] + np.arange(6, 11).tolist()
            group3 = [0, 11]
            groups = [group1, group2, group3]

        if args.num_comm_ue == 20 and args.num_devices == 20:
            server_list = [0,21]
            group1 = [0] + np.arange(1, 11).tolist()
            group2 = [21] + np.arange(11, 21).tolist()
            group3 = [0,21]
            groups = [group1, group2, group3]

        if args.num_comm_ue == 20 and args.num_devices == 30:
            server_list = [0,31]
            group1 = [0] + np.arange(1, 11).tolist()
            group2 = [21] + np.arange(11, 21).tolist()
            group3 = [0,21]
            groups = [group1, group2, group3]

        if args.num_comm_ue == 30 and args.num_devices == 30:
            server_list = [0,31,32]
            group1 = [0] + np.arange(1, 11).tolist()
            group2 = [31] + np.arange(11, 21).tolist()
            group3 = [32] + np.arange(21, 31).tolist()
            group4 = [0,31,32]
            groups = [group1, group2, group3, group4]

    return groups, server_list


def save_model_of_each_group(args, model, epoch, rank, rank_save, prefix):
    average_model_weights = copy.deepcopy(model.state_dict())
    if epoch%args.epoch_interval == 0 or epoch == args.epoch-1:
        if rank == rank_save:
            Save_Avg_model_checkpoint(args.experiment_name, average_model_weights, rank, epoch, prefix=prefix)

def save_each_group_avg_model(args, average_model_weights, epoch, rank, rank_save, prefix):
    if epoch%args.epoch_interval == 0 or epoch == args.epoch-1:
        if rank == rank_save:
            Save_Avg_model_checkpoint(args.experiment_name, average_model_weights, rank, epoch, prefix=prefix)

def Grouping_Avg(args, model, rank, G, groups, epoch):
    # print('Groupng method >>>>>')
    group_id = Get_group_num(args, groups, rank)
    if epoch == 0:
        print(rank,group_id,groups[group_id],len(groups[group_id]))
    if rank in set(groups[group_id]):
        if epoch == 0:
            print(f'This model rank {rank} in group {group_id} has been saved')
        SyncAllreduce_1(model, rank, size=len(groups[group_id]), group=G[group_id])
        average_group_model_weights = copy.deepcopy(model.state_dict())
        if epoch == 0:
            print(f'This model rank {rank} has been SyncAllreduce_1 in group {group_id}')
        if epoch == 0:
            print(f'This is rank {rank} in group {group_id}')
        save_model_of_each_group(args, model, epoch, rank, rank_save=groups[group_id][0], prefix=f'after_g{group_id+1}')
        if epoch == 0:
            print(f'This model rank {rank} in group {group_id} has been saved')

    if not args.user_semi:
        if rank in set(groups[-1]):
            SyncAllreduce_1(model, rank, size=len(groups[-1]), group=G[-1])
            if epoch == 0:
                print(f'This model rank {rank} has been SyncAllreduce_1 in group {groups[-1]}')

    return average_group_model_weights


def Get_group_num(args, groups, rank):
    groups_user = groups[0:-1]
    for i in range(len(groups_user)):
        if rank in set(groups_user[i]):
            return i

def cal_avg_weight_diff_v2(weights_list, avg_weights):
        w = copy.deepcopy(weights_list)
        w2 = copy.deepcopy(avg_weights)

        del weights_list
        del avg_weights

        key = list(w2.keys())

        for key in list(w2.keys()):
            w2[key] = w2[key].reshape(-1).cpu().tolist()     # Reshape to 1d tensor and transform to list
            for i in range(len(w)):
                tmp = []
                w[i][key] = w[i][key].reshape(-1).cpu().tolist()     # Reshape to 1d tensor and transform to list

        # List for differences: for all devices, get the average of abs((val(device)-val(average))/val(average))
        diff_list = []
        WD_key_user = []
        for key in list(w2.keys()):
            WD_key = []
            # print(key)
            if 'num_batches_tracked' in key or 'running' in key or 'bn' in key:
                pass
            # if 'linear' in key:
            else:
                for i in range(len(w)):
                    tmp = []
                    tmp_avg = []
                    for j in range(len(w[i][key])):
                        tmp.append(w[i][key][j]-w2[key][j])
                        tmp_avg.append(w2[key][j])

                    tmp = np.array(tmp)
                    tmp_avg = np.array(tmp_avg)
                    tmp = torch.from_numpy(tmp).type(torch.FloatTensor)
                    wd_i = torch.norm(tmp)#/denominator

                    WD_key.append(wd_i.item()**2)

                WD_key_user.append(WD_key)

        return WD_key_user

def Get_num_ranks_all_size_num_devices(args):
    # dataset = args.dataset
    if args.datasetid == 0:
        dataset = 'cifar10'
    if args.datasetid == 1:
        dataset = 'svhn'
    if args.datasetid == 2:
        dataset = 'emnist'
    num_comm_ue = args.num_comm_ue
    size = args.size
    H = args.H
    if dataset == 'emnist' and num_comm_ue == 10:
        num_rank = 11 + H
        size_all = 47 + 1 + H
        num_devices = 47
    if dataset == 'emnist' and num_comm_ue == 30:
        num_rank = 31 + 2*H
        size_all = 47 + 1 + 2*H
        num_devices = 47
    if dataset == 'emnist' and num_comm_ue == 47:
        num_rank = 48 + 4*H
        size_all = 47 + 1 + 4*H
        num_devices = 47
    if dataset == 'cifar10' and num_comm_ue == 10:
        if args.user_semi:
            num_rank = 10
            size_all = 100
            num_devices = 100
        else:
            num_rank = 11 + H
            size_all = 10 + 1 + H
            num_devices = 10

    if dataset == 'svhn' or dataset == 'cifar10':
        if num_comm_ue == 10 or num_comm_ue == 20:
            num_rank = num_comm_ue + 1 + H
            size_all = size + H
        if num_comm_ue == 30:
            num_rank = num_comm_ue + 1 + H*2
            size_all = size + H*2
        num_devices = size - 1

    if dataset == 'cifar10' and num_comm_ue == 5:
        if args.user_semi:
            num_rank = num_comm_ue
            size_all = size
            num_devices = size
        else:
            num_rank = num_comm_ue + 1 + H
            size_all = size + H
            num_devices = size - 1



    return num_rank, size_all, num_devices


def Get_torch_init_rank_group(args):
    if args.dataset == 'emnist':
        if args.num_comm_ue == 10:
            group1_size = args.num_rank//2
            group1 = np.arange(0, group1_size).tolist()
            group2 = np.arange(group1_size, size).tolist()
            group3 = [0, args.num_rank - 1]
            groups = [group1, group2, group3]


        if args.num_comm_ue == 20:
            group1_size = 10
            group1 = [0] + np.arange(1, 11).tolist()
            group2 = [21] + np.arange(11, 21).tolist()
            group3 = [0,21]
            groups = [group1, group2, group3]


        if args.num_comm_ue == 47:
            group1_size = 10
            group1 = [0] + np.arange(1, 11).tolist()
            group2 = [48] + np.arange(11, 21).tolist()
            group3 = [49] + np.arange(21, 31).tolist()
            group4 = [50] + np.arange(31, 41).tolist()
            group5 = [51] + np.arange(41, 48).tolist()
            group6 = [0, 48, 49, 50, 51]

            groups = [group1, group2, group3, group4, group5, group6]


    if args.dataset == 'cifar10' or args.dataset == 'svhn':
        if args.user_semi:
            group1 = np.arange(0, 5).tolist()
            group2 = np.arange(5, 10).tolist()
            group3 = [0,9]
            groups = [group1, group2, group3]

        if args.num_comm_ue == 10 :
            group1_size = 10
            group1 = [0] + np.arange(1, 6).tolist()
            group2 = [11] + np.arange(6, 11).tolist()
            group3 = [0,11]
            groups = [group1, group2, group3]


        if args.num_comm_ue == 20 and args.num_devices == 30:
            group1_size = 10
            group1 = [0] + np.arange(1, 11).tolist()
            group2 = [21] + np.arange(11, 21).tolist()
            group3 = [0,21]
            groups = [group1, group2, group3]


        if args.num_comm_ue == 30:
            group1_size = 10
            group1 = [0] + np.arange(1, 11).tolist()
            group2 = [31] + np.arange(11, 21).tolist()
            group3 = [32] + np.arange(21, 31).tolist()
            group4 = [0,31,32]
            groups = [group1, group2, group3, group4]

        return groups
