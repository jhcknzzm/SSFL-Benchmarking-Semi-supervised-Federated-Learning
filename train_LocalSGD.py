import os
import numpy as np
import time
import argparse
import sys

from math import ceil
from random import Random
import time
import random
import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn as nn
import torch.nn.functional as F

from torch.multiprocessing import Process
import torchvision
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import datetime

import LocalSGD as optim
import util_v4 as util
from comm_helpers import SyncAllreduce, SyncAllreduce_1, SyncAllreduce_2
import os
from scipy.io import loadmat
import json
from scipy import io
from dataset.cifar import get_cifar10, get_emnist, get_svhn
from torch.optim.lr_scheduler import LambdaLR
import math



parser = argparse.ArgumentParser(description='CIFAR-10 baseline')
parser.add_argument('--name','-n',
                    default="default",
                    type=str,
                    help='experiment name, used for saving results')
parser.add_argument('--backend',
                    default="nccl",
                    type=str,
                    help='experiment name, used for saving results')
parser.add_argument('--GPU_list',
                    default='0',
                    type=str,
                    help='gpu list')
parser.add_argument('--dataset',
                    default="cifar10",
                    type=str,
                    help='dataset name')
parser.add_argument('--model',
                    default="res_gn",
                    type=str,
                    help='neural network model')
parser.add_argument('--alpha',
                    default=0.2,
                    type=float,
                    help='alpha')
parser.add_argument('--gmf',
                    default=0,
                    type=float,
                    help='global momentum factor')
parser.add_argument('--lr',
                    default=0.1,
                    type=float,
                    help='learning rate')

parser.add_argument('--basicLabelRatio',
                    default=0.4,
                    type=float,
                    help='basicLabelRatio')
parser.add_argument('--bs',
                    default=64,
                    type=int,
                    help='batch size on each worker')
parser.add_argument('--epoch',
                    default=300,
                    type=int,
                    help='total epoch')
parser.add_argument('--cp',
                    default=8,
                    type=int,
                    help='communication period / work per clock')
parser.add_argument('--print_freq',
                    default=100,
                    type=int,
                    help='print info frequency')
parser.add_argument('--rank',
                    default=0,
                    type=int,
                    help='the rank of worker')
parser.add_argument('--size',
                    default=8,
                    type=int,
                    help='number of workers')
parser.add_argument('--seed',
                    default=1,
                    type=int,
                    help='random seed')
parser.add_argument('--num_comm_ue',
                    default=11,
                    type=int,
                    help='communication user number')
parser.add_argument('--iid',
                    default=1,
                    type=int,
                    help='iid')
parser.add_argument('--class_per_device',
                    default=1,
                    type=int,
                    help='class_per_device')

parser.add_argument('--labeled',
                    default=0,
                    type=int,
                    help='labeled all data')
parser.add_argument('--H',
                    default=0,
                    type=int,
                    help='whether use hierarchical method')
parser.add_argument('--save', '-s',
                    action='store_true',
                    help='whether save the training results')
parser.add_argument('--ip_address',
                    default="10.129.2.142",
                    type=str,
                    help='ip_address')
parser.add_argument('--master_port',
                    default="29021",
                    type=str,
                    help='master port')

parser.add_argument('--experiment_name',
                    default="Major1_setting1",
                    type=str,
                    help='name of this experiment')
parser.add_argument('--k-img', default=65536, type=int,  ### 65536
                    help='number of examples')
parser.add_argument('--num_data_server', default=1000, type=int,
                    help='number of samples in server')
parser.add_argument('--num-data-server', default=1000, type=int,
                    help='number of labeled examples in server')

parser.add_argument('--num-devices', default=10, type=int,
                    help='num of devices')


args = parser.parse_args()

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1,lr_weight=1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        num_cycles = 7.0/16.0*(1024*1024 - num_warmup_steps)/(1024*200 - num_warmup_steps)
        return max(0.00000, math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

######### Assign Ranks to different GPUs
GRU_list = [i for i in args.GPU_list]
if args.H:
    increase_tmp = args.size//len(GRU_list)
else:
    increase_tmp = (args.size+1)//len(GRU_list)

ranks_list = np.arange(0,args.size).tolist()

rank_group = []
for rank_id in range(len(GRU_list)):
    if rank_id == len(GRU_list)-1:
        ranks = ranks_list[rank_id*increase_tmp:]
    else:
        ranks = ranks_list[rank_id*increase_tmp:(rank_id+1)*increase_tmp]
    rank_group.append(ranks)

for group_id in range(len(GRU_list)):
    if args.rank in set(rank_group[group_id]):
        os.environ["CUDA_VISIBLE_DEVICES"] = GRU_list[group_id]
        break



device = 'cuda' if torch.cuda.is_available() else 'cpu'

DATASET_GETTERS = {'cifar10': get_cifar10, 'emnist': get_emnist, 'svhn':get_svhn}

### generate the index of the server dataset and the device dataset
if args.iid:
    path_device_idxs = f'{args.dataset}_post_data/iid/{args.size - 1 - args.H}_{args.num_data_server}'
else:
    path_device_idxs = f'{args.dataset}_post_data/noniid/{args.size - 1 - args.H}_{args.num_data_server}_{args.class_per_device}_{args.basicLabelRatio}'

if args.dataset == 'emnist':
    if args.iid:
        path_device_idxs = f'{args.dataset}_post_data/iid/{47}_{args.num_data_server}'
    else:
        path_device_idxs = f'{args.dataset}_post_data/noniid/{47}_{args.num_data_server}_{args.class_per_device}_{args.basicLabelRatio}'


device_ids = np.load(path_device_idxs + '/device_idxs' + '.npy', allow_pickle=True).item()
server_idxs = np.load(path_device_idxs + '/server_idxs' + '.npy', allow_pickle=True).item()
device_ids = device_ids['device_idxs']
server_idxs = server_idxs['server_idxs']

if args.num_comm_ue < args.size - 1 - args.H:
    ue_list_epoches = np.load(path_device_idxs + '/ue_list_epoch' + '.npy', allow_pickle=True).item()
    ue_list_epoches = ue_list_epoches['ue_list_epoch']
else:
    ue_list_epoches = []


print('get dataset')

labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
    './data', args.k_img, args.k_img * len(device_ids), device_ids, server_idxs)
print('get dataset, done')
train_sampler = RandomSampler

labeled_trainloader = DataLoader(
    labeled_dataset,
    sampler=train_sampler(labeled_dataset),
    batch_size=args.bs,
    num_workers=0,
    drop_last=True)

unlabeled_trainloader_list = []

for id in range(len(unlabeled_dataset)):
    unlabeled_trainloader = DataLoader(
        unlabeled_dataset[id],
        sampler=train_sampler(unlabeled_dataset[id]),
        batch_size=args.bs,
        num_workers=0,
        drop_last=True)
    unlabeled_trainloader_list.append(unlabeled_trainloader)

test_loader = DataLoader(test_dataset,
                         batch_size=args.bs,
                         shuffle=False)


print(args)

def run(rank, size, G):
    # initiate experiments folder
    save_path = './results_v0/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    folder_name = save_path+args.name+'/'
    if rank == 0 and os.path.isdir(folder_name)==False and args.save:
        os.makedirs(folder_name)
    dist.barrier()

    # seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # load datasets
    if args.H:
        if args.dataset == 'emnist':
            labeled_set = [0,48,49,50,51]
            if rank in set(labeled_set):
                train_loader = labeled_trainloader
            else:
                train_loader = unlabeled_trainloader_list[rank - 1]
        else:
            if rank == 0 or rank == args.size -1:
                train_loader = labeled_trainloader
            else:
                train_loader = unlabeled_trainloader_list[rank - 1]
    else:
        if rank == 0:
            train_loader = labeled_trainloader
        else:
            train_loader = unlabeled_trainloader_list[rank - 1]

    # define neural nets model, criterion, and optimizer
    model = util.select_model(args.model, args).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          alpha=args.alpha,
                          gmf=args.gmf,
                          size=size,
                          momentum=0.9,
                          nesterov = True,
                          weight_decay=1e-4)

    args.iteration = args.k_img // args.bs
    total_steps = 1024 * args.iteration
    # total_steps = args.epoch * args.iteration
    warmup_epoch = 5
    if args.dataset == 'emnist':
        warmup_epoch = 0
        total_steps = args.epoch * args.iteration

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_epoch * args.iteration, total_steps,lr_weight=1)

    batch_meter = util.Meter(ptag='Time')
    comm_meter = util.Meter(ptag='Time')

    best_test_accuracy = 0
    req = None
    acc_list = []
    print('Now train the model')

    for epoch in range(args.epoch):
        if rank == 0:
            begin_time = time.time()

        train(rank, model, criterion, optimizer,scheduler, batch_meter, comm_meter,
              train_loader, epoch, device, ue_list_epoches, G)
        ### test the server model
        if rank == 0:
            test_acc = evaluate(model, test_loader)
            acc_list.append(round(test_acc, 2))
            print('test acc',epoch, test_acc,time.time() - begin_time)
            if args.H:
                filename = f"./results_v0/{args.experiment_name}_{args.dataset}_iid{args.iid}_UE{args.size - 1}_{args.basicLabelRatio}_{args.model}_bs{args.bs}_H1_cp{args.cp}.txt"
            else:
                filename = f"./results_v0/{args.experiment_name}_{args.dataset}_iid{args.iid}_UE{args.size - 1 - args.H}_{args.basicLabelRatio}_comUE{args.num_comm_ue}_{args.model}_bs{args.bs}_H0_cp{args.cp}.txt"
            if filename:
                with open(filename, 'w') as f:
                    json.dump(acc_list, f)

        path_checkpoint = f"./checkpoint/{args.experiment_name}/"
        if not os.path.exists(path_checkpoint):
            os.makedirs(path_checkpoint)
        torch.save({'epoch': epoch,'model_state_dict': model.state_dict()}, path_checkpoint+f'{rank}_weights.pth')


def evaluate(model, test_loader):
    model.eval()
    top1 = util.AverageMeter()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.cuda(non_blocking = True)
            target = target.cuda(non_blocking = True)
            outputs = model(data)
            acc1 = util.comp_accuracy(outputs, target)
            top1.update(acc1[0].item(), data.size(0))

    return top1.avg

def train(rank, model, criterion, optimizer,scheduler, batch_meter, comm_meter,
          loader, epoch, device, ue_list_epoches, G):

    model.train()

    top1 = util.Meter(ptag='Prec@1')

    iter_time = time.time()

    if args.H:
        if args.dataset == 'emnist':
            group1 = [0] + np.arange(1, 11).tolist()
            group2 = [48] + np.arange(11, 21).tolist()
            group3 = [49] + np.arange(21, 31).tolist()
            group4 = [50] + np.arange(31, 41).tolist()
            group5 = [51] + np.arange(41, 48).tolist()
            group6 = [0, 48, 49, 50, 51]
        else:
            group6 = [0,args.size -1]

    for batch_idx, (data) in enumerate(loader):
        training = 0
        if args.num_comm_ue < args.size - 1 - args.H:
            ue_list = ue_list_epoches[epoch][batch_idx]
            ue_list_set = set(ue_list)
            if rank in ue_list_set:
                training = 1
            else:
                training = 0
        else:
            training = 1
        if training:
            if args.H:
                if rank in set(group6):
                    inputs_x, targets_x = data
                    inputs_x = inputs_x.to(device)
                    targets_x = targets_x.to(device)
                    output = model(inputs_x)
                    loss = criterion(output, targets_x)
                else:
                    (inputs_u_w, inputs_u_s), _ = data

                    inputs = torch.cat((inputs_u_w, inputs_u_s)).to(device)
                    logits = model(inputs)
                    logits_u_w, logits_u_s = logits.chunk(2)
                    del logits

                    pseudo_label = torch.softmax(logits_u_w.detach_(), dim=-1)
                    max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                    mask = max_probs.ge(0.95).float()

                    loss = (F.cross_entropy(logits_u_s, targets_u,
                                          reduction='none') * mask).mean()
            else:
                if rank == 0:
                    inputs_x, targets_x = data
                    inputs_x = inputs_x.to(device)
                    targets_x = targets_x.to(device)
                    output = model(inputs_x)
                    loss = criterion(output, targets_x)
                else:
                    (inputs_u_w, inputs_u_s), _ = data

                    inputs = torch.cat((inputs_u_w, inputs_u_s)).to(device)
                    logits = model(inputs)
                    logits_u_w, logits_u_s = logits.chunk(2)
                    del logits

                    pseudo_label = torch.softmax(logits_u_w.detach_(), dim=-1)
                    max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                    mask = max_probs.ge(0.95).float()

                    loss = (F.cross_entropy(logits_u_s, targets_u,
                                          reduction='none') * mask).mean()


            # backward pass
            accum_steps = 1
            loss = loss / accum_steps
            loss.backward()
            if batch_idx % accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        torch.cuda.synchronize()
        comm_start = time.time()

        accum_steps = 1
        if args.H:
            if args.dataset == 'emnist':
                if batch_idx != 0 and batch_idx % args.cp*accum_steps == 0:

                    if rank in set(group1):
                        SyncAllreduce_1(model, rank, size=len(group1), group=G[0])
                    elif rank in set(group2):
                        SyncAllreduce_1(model, rank, size=len(group2), group=G[1])
                    elif rank in set(group3):
                        SyncAllreduce_1(model, rank, size=len(group3), group=G[2])
                    elif rank in set(group4):
                        SyncAllreduce_1(model, rank, size=len(group4), group=G[3])
                    elif rank in set(group5):
                        SyncAllreduce_1(model, rank, size=len(group5), group=G[4])
                    if rank in set(group6):
                        SyncAllreduce_1(model, rank, size=len(group6), group=G[5])



            else:
                if batch_idx != 0 and batch_idx % args.cp*accum_steps == 0:
                    if rank < args.size//2:
                        #### Group 1 avgerage and communicate
                        SyncAllreduce_1(model, rank, size=args.size//2, group=G[0])
                    else:
                        #### Group 2 avgerage and communicate
                        SyncAllreduce_1(model, rank, size=args.size - args.size//2, group=G[1])
                    if rank == 0 or rank == args.size - 1:
                        #### Server model 1 and server 2 avgerage and communicate
                        SyncAllreduce_1(model, rank, size=2, group=G[2])



        else:
            if batch_idx != 0 and batch_idx % args.cp*accum_steps == 0:
                if args.num_comm_ue < args.size - 1:
                    ue_list = ue_list_epoches[epoch][batch_idx]
                    SyncAllreduce_2(model, rank, size, ue_list)
                else:
                    SyncAllreduce(model, rank, size)


        if not (epoch == 0 and batch_idx == 0):
            torch.cuda.synchronize()
            comm_meter.update(time.time() - comm_start)
            batch_meter.update(time.time() - iter_time)


        torch.cuda.synchronize()
        iter_time = time.time()


def init_processes(rank,size,fn, ip_address, master_port, H):
    os.environ['MASTER_ADDR'] = ip_address # a18 169.229.49.58 # a23 169.229.49.63
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group('gloo', rank=rank, world_size=size)
    torch.cuda.manual_seed(1)
    if H:
        group1_size = size//2
        group1 = np.arange(0, group1_size).tolist()
        group2 = np.arange(group1_size, size).tolist()

        G1 = torch.distributed.new_group(ranks=group1)
        G2 = torch.distributed.new_group(ranks=group2)
        G3 = torch.distributed.new_group(ranks=[0, size - 1])
        G = [G1,G2,G3]
    else:
        G = []

    if args.dataset == 'emnist' and H:
        group1_size = 10
        group1 = [0] + np.arange(1, 11).tolist()
        group2 = [48] + np.arange(11, 21).tolist()
        group3 = [49] + np.arange(21, 31).tolist()
        group4 = [50] + np.arange(31, 41).tolist()
        group5 = [51] + np.arange(41, 48).tolist()

        G1 = torch.distributed.new_group(ranks=group1)
        G2 = torch.distributed.new_group(ranks=group2)
        G3 = torch.distributed.new_group(ranks=group3)
        G4 = torch.distributed.new_group(ranks=group4)
        G5 = torch.distributed.new_group(ranks=group5)
        G6 = torch.distributed.new_group(ranks=[0, 48, 49, 50, 51])
        G = [G1,G2,G3,G4,G5,G6]


    fn(rank, size, G)

if __name__ == "__main__":
    rank = args.rank
    size = args.size
    master_port = args.master_port
    print(rank)
    init_processes(rank, size, run, args.ip_address, master_port, args.H)
