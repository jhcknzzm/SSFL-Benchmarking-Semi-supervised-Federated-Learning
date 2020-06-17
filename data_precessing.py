import os

import torch
import numpy as np
import random
from tqdm import tqdm
import copy
from torch.utils.data import Dataset, DataLoader
from utils_v1 import Utils
import argparse
from models import *
from scipy import io
from scipy.io import loadmat


class Parser():
    def parse(self):
        parser = argparse.ArgumentParser()
        # Training setting
        parser.add_argument("--dataset", help="dataset (cifar10)",
                            default="cifar10", type=str)
        parser.add_argument("--path_device_idxs", help="path_device_idxs",
                            default='cifar10_post_data/iid/16', type=str)
        parser.add_argument("--num_devices", help="number of devices",
                            default=10, type=int)
        parser.add_argument("--num_data_server", help="max amount of data per device",
                            default=1000, type=int)
        # IIDness setting
        parser.add_argument("--iid", help="data is iid or not",
                            default=0, type=int)
        parser.add_argument("--noniidness", help="percentage of non-iid data per device",
                            default=1, type=float)
        parser.add_argument("--basicLabelRatio", help="percentage of non-iid data",
                            default=1, type=float)
        parser.add_argument("--equal_dist", help="equal data distribution across devices",
                            default=0, type=int)
        parser.add_argument("--class_per_device", help="number of classes per device (non-iid, conflict with noniidness)",
                            default=1, type=int)

        return parser.parse_args()

args = Parser().parse()

utils = Utils()
train_dataset, test_dataset, device_idxs, server_idxs = utils.get_dataset_dist(args)
if args.dataset == 'cifar10':
    num_class = 10
    targets = train_dataset.targets
if args.dataset == 'emnist':
    num_class = 47#62
    targets = train_dataset.targets
if args.dataset == 'svhn':
    num_class = 10
    targets = train_dataset.labels


# num_class = 62#train_dataset.classes
print(num_class)

num_each_class=[]
for i in range(num_class):
    idx_i = np.where(np.array(targets)[server_idxs]==i)
    num_each_class.append(idx_i[0].shape[0])
print(num_each_class)

device_idxs_all = []
for num in range(len(device_idxs)):
    print(len(device_idxs[num]))
    user_targets = np.array(targets)[device_idxs[num]]
    device_idxs_all += device_idxs[num]
    num_each_class=[]
    for i in range(num_class):
        num_class_i = np.where(user_targets==i)
        num_each_class.append(num_class_i[0].shape[0])

    print(num_each_class)
device_idxs_all = list(set(device_idxs_all))
idx_all = np.arange(0,50000).tolist()
print('len of device indx',len(device_idxs_all))
# result4 = list(set(idx_all).difference(set(device_idxs_all)))
# print('result4',result4)


if args.path_device_idxs[-1] != '/':
    args.path_device_idxs += '/'

if not os.path.exists(args.path_device_idxs):
    os.makedirs(args.path_device_idxs)

io.savemat(f"{args.path_device_idxs}device_idxs.mat", {'device_idxs': device_idxs})
io.savemat(f"{args.path_device_idxs}serveridxs_idxs.mat", {'server_idxs': server_idxs})

dictionary1 = {'device_idxs':device_idxs}
np.save(f"{args.path_device_idxs}device_idxs.npy", dictionary1)

dictionary2 = {'server_idxs':server_idxs}
np.save(f"{args.path_device_idxs}server_idxs.npy", dictionary2)
