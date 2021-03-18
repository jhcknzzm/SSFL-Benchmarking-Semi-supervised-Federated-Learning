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
import torch.optim as optim_tr
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
from comm_helpers import SyncAllreduce, SyncAllreduce_1, SyncAllreduce_2, SyncAllGather
from scipy.io import loadmat
import json
from scipy import io
from dataset.cifar import get_cifar10, get_cifar100, get_emnist, get_svhn, get_cifar10_semi#, get_emnist, get_svhn
from torch.optim.lr_scheduler import LambdaLR
import utils_v2 as util_1
import math
from utils_v2 import Utils
import copy
import torch.distributed as dist
from utils_v2 import *

"""
The reference sample:
parameters:
    size = 10+1
    batch_size = 64
    cp = 16
    basicLabelRatio = 0.0
    model = 'res_gn'
    iid = 0
    num_comm_ue = 5
    k_img = 65536
    epoches = 300
    warmup_epoch = 5
    num_data_server = 1000
    experiment_name = 'UE10_comUE5_LabeledPerson'
    GPU_list = '01234'
    rank = 0
    master_port = 12345
    ip_address = '10.129.2.142'

comm:
    python train_LocalSGD_pers_v2.py --dataset {dataset} --model {model}  \
                            --lr {lr} --bs {batch_size} --cp {cp} --alpha 0.6 --gmf 0.7 --basicLabelRatio {basicLabelRatio} --master_port {master_port}\
                            --name revised_results_e300 --ip_address {ip_address} --num_comm_ue {num_comm_ue} --num_data_server {num_data_server}\
                            --iid {iid} --rank {rank} --size {size} --backend gloo --warmup_epoch {warmup_epoch} --GPU_list {GPU_list} --labeled {labeled}\
                            --class_per_device {1} --num-devices {size - 1} --epoch {epoches} --experiment_name {experiment_name}
"""
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
                    default=0.16,
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
                    default=10,
                    type=int,
                    help='communication user number')
parser.add_argument('--iid',
                    default=0,
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
parser.add_argument('--warmup_epoch',
                    default=0,
                    type=int,
                    help='warmup epoch')
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
parser.add_argument('--k-img', default=65536, type=int,
                    help='number of examples')
parser.add_argument('--num_data_server', default=1000, type=int,
                    help='number of samples in server')
parser.add_argument('--num-data-server', default=1000, type=int,
                    help='number of labeled examples in server')
parser.add_argument('--num-devices', default=10, type=int,
                    help='num of devices')
parser.add_argument('--fast',
                    default=0,
                    type=int,
                    help='use scheduler fast model or not')
parser.add_argument('--H',
                    default=0,
                    type=int,
                    help='grouping or not')
parser.add_argument('--epoch_resume',
                    default=0,
                    type=int,
                    help='epoch checkpoints')
parser.add_argument('--epoch_interval',
                    default=10,
                    type=int,
                    help='epoch_interval')
parser.add_argument('--num_rank',
                    default=10,
                    type=int,
                    help='num_rank')

parser.add_argument('--eval_grad',
                    default=1,
                    type=int,
                    help='eval_grad or training')

parser.add_argument('--experiment_folder', default='.', type=str,
                    help='the path of the experiment')

parser.add_argument('--tao',
                    default=0.95,
                    type=float,
                    help='tao for cal. mask')

parser.add_argument('--ue_loss', default='CRL', type=str,
                    help='user loss')

parser.add_argument('--user_semi',
                    default=1,
                    type=int,
                    help='user side semi')


args = parser.parse_args()
def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1,fast=True):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        if fast:
            num_cycles = 7.0/16.0*(1024*1024 - num_warmup_steps)/(1024*200 - num_warmup_steps)
            return max(0.00001, math.cos(math.pi * num_cycles * no_progress))
        else:
            num_cycles = 7.0/16.0
            return max(0.000001, math.cos(math.pi * num_cycles * no_progress))
    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def Get_Model(args):
    model = util.select_model(args.model, args).cuda()
    return model

def Get_Criterion(args):
    criterion = nn.CrossEntropyLoss().cuda()
    return criterion

def Get_Optimizer(args, model, size=1, lr=0.03):
    optimizer = optim.SGD(model.parameters(),
                          lr=lr,
                          alpha=args.alpha,
                          gmf=args.gmf,
                          size=size,
                          momentum=0.9,
                          nesterov = True,
                          weight_decay=1e-4)
    return optimizer

def Get_Scheduler(args, optimizer, warmup_epoch=5, base_epoch=1024, fast=True):
    args.iteration = args.k_img // args.bs

    total_steps = base_epoch * args.iteration
    if args.dataset == 'emnist' or args.dataset == 'svhn':
        total_steps = args.epoch * args.iteration
    if args.user_semi:
        total_steps = args.epoch * args.iteration

    # total_steps = args.epoch * args.iteration
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_epoch * args.iteration, total_steps, fast=fast)
    return scheduler

### generate the index of the server dataset and the device dataset
def Get_TrainLoader(args):
    ##### generate the path to save data indexes of the users and the server
    if args.iid:
        path_device_idxs = '%s_post_data/iid/%s_%s_%s_H_%s_UserSemi_%s/' %(args.dataset, args.num_devices, args.num_data_server, args.num_comm_ue, args.H, args.user_semi)
    else:
        path_device_idxs = '%s_post_data/noniid/%s_%s_%s_%s_%s_H_%s_UserSemi_%s/' %(args.dataset, args.num_devices, args.num_data_server, args.class_per_device, args.basicLabelRatio, args.num_comm_ue, args.H, args.user_semi)

    if not os.path.exists(path_device_idxs):
        try:
            os.makedirs(path_device_idxs)
        except OSError:
            pass

        util_1.Generate_device_server_index(args, path_device_idxs) #### generate and save data index of the users and the server
        ##### if the number of users communicate with the server smaller than the total number of users,
        ##### we generate a user list to decide which user will communicate with the server, and server it in the path of 'path_device_idxs'
        util_1.Generate_communicate_user_list(args, path_device_idxs)

    import time
    time.sleep(5)
    if not args.user_semi:
        DATASET_GETTERS = {'cifar10': get_cifar10, 'emnist': get_emnist, 'svhn': get_svhn}
    else:
        DATASET_GETTERS = {'cifar10': get_cifar10_semi}

    server_idxs, device_ids = util_1.Load_device_server_index(args, path_device_idxs)
    labeled_dataset, unlabeled_dataset, test_dataset, base_dataset = DATASET_GETTERS[args.dataset](
        './data', args.k_img, args.k_img * len(device_ids), device_ids, server_idxs)
    print('get dataset, done')

    if args.user_semi:
        Train_loader_list, Test_loader_list = Generate_Train_data_loader_user_side_semi(args, labeled_dataset, unlabeled_dataset, test_dataset)
        max_len = 0
    else:
        if args.ue_loss == 'SF':
            Train_loader_list, Test_loader_list, max_len = util_1.Get_SF_train_test_dataloader(device_ids, server_idxs, args)
            print('max_len',max_len)
        else:
            Train_loader_list = util_1.Generate_Train_data_loader(args, labeled_dataset, unlabeled_dataset, RandomSampler)
            Test_loader_list = util_1.Generate_Test_data_loader(args, test_dataset, base_dataset, device_ids)
            max_len = 0
    return Train_loader_list, Test_loader_list, path_device_idxs, max_len

def run(rank, size, G):
    # initiate experiments folder
    save_path = f'./results_v0/{args.experiment_name}/'
    if rank == 0:
        if not os.path.exists(save_path):
            try:
                os.makedirs(save_path)
            except OSError:
                pass

        folder_name = save_path+args.name+'/'
        if rank == 0 and os.path.isdir(folder_name)==False and args.save:
            os.makedirs(folder_name)
    else:
        time.sleep(5)

    dist.barrier()
    # seed for reproducibility
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True

    train_loader_list, test_loader_list, path_device_idxs, max_len = Get_TrainLoader(args) ### load datasets
    ue_list_epoches = util_1.Load_communicate_user_list(args, path_device_idxs) ### load communicate user list

    # define neural nets model, criterion, and optimizer
    model = Get_Model(args)
    criterion = Get_Criterion(args)
    optimizer = Get_Optimizer(args, model, size = size, lr = args.lr)
    if args.fast == 0:
        fast = False
    else:
        fast = True

    scheduler = Get_Scheduler(args, optimizer, warmup_epoch=args.warmup_epoch, fast=fast)

    batch_meter = util.Meter(ptag='Time')
    comm_meter = util.Meter(ptag='Time')

    print('Now train the model')


    Fed_training = True


    user_weight_diff_array = np.zeros((args.size, args.epoch, args.iteration+1))
    if Fed_training:

        if args.epoch_resume == 0:
            start_epoch = 0
            #### At the first epoch, we delete the past files
            if not args.eval_grad and rank == 0:
                util_1.init_files(args, save_path, rank, prefix='Test_Acc')
            Fed_acc_list = []

        else:
            start_epoch = args.epoch_resume+1
            if rank == 0:
                Fed_acc_list = util_1.get_acc(args, save_path, rank, prefix='Test_Acc')

        if args.eval_grad: #### at this time, we only want to cal. grad. dont want to change DataLoader
            args.iteration = 1

        if args.ue_loss == 'SF':
            args.iteration = max_len // args.bs

        for epoch in range(start_epoch, args.epoch):

            begin_time = time.time()
            if args.epoch_resume > 0 and epoch == args.epoch_resume+1:
                print('Loading saved averaged model ... epoch=', epoch,  args.epoch_resume)
                checkpoint_weights = util_1.Load_Avg_model_checkpoint(args.experiment_folder, args.experiment_name, epoch, prefix='after')
                model.load_state_dict(checkpoint_weights, strict=False)


            if not args.eval_grad or epoch%args.epoch_interval == 0:
                user_id, WD_list, user_weight_diff_array = train(rank, model, criterion, optimizer, scheduler, batch_meter, comm_meter,
                      train_loader_list, test_loader_list, epoch, device, ue_list_epoches, G, user_weight_diff_array)
                      # get and save the local fine-tuning acc
                if rank == 0:

                    test_acc = evaluate(model, test_loader_list[0])
                    test_acc = round(test_acc, 2)
                    print('test acc', epoch, test_acc, time.time() - begin_time)

                    if not args.eval_grad:
                        Fed_acc_list.append(test_acc)
                        util_1.Save_acc_file(args, save_path, rank, prefix='Test_Acc', acc_list=Fed_acc_list)

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

def train(rank, model, criterion, optimizer, scheduler, batch_meter, comm_meter,
          train_loader_list, test_loader_list, epoch, device, ue_list_epoches, G, user_weight_diff_array):
    average_model_weights = copy.deepcopy(model.state_dict())
    average_group_model_weights = copy.deepcopy(model.state_dict())

    model.train()
    WD_list = []
    top1 = util.Meter(ptag='Prec@1')
    iter_time = time.time()
    accum_steps = 1
    iteration = 0

    while iteration < args.iteration:

        ue_list = ue_list_epoches[epoch][iteration] ### Get the users (a list) that are involved in the computation

        user_id = ue_list[rank]

        groups, server_list = get_groups(args)

        if args.user_semi:
            loader = zip(train_loader_list[user_id][0], train_loader_list[user_id][1])
            test_loader = test_loader_list[0]

            if args.eval_grad and epoch%args.epoch_interval == 0:
                group_id = Get_group_num(args, groups, rank)
                checkpoint_weights = util_1.Load_Avg_model_checkpoint(args.experiment_folder, args.experiment_name, epoch, prefix=f'after_g{group_id+1}')
                model.load_state_dict(checkpoint_weights, strict=False)

        else:
            if args.H:

                if user_id in set(server_list):
                    loader = train_loader_list[0]
                    test_loader = test_loader_list[0]
                else:
                    loader = train_loader_list[user_id]
                    test_loader = test_loader_list[0]

                if args.eval_grad and epoch%args.epoch_interval == 0:
                    group_id = Get_group_num(args, groups, rank)
                    checkpoint_weights = util_1.Load_Avg_model_checkpoint(args.experiment_folder, args.experiment_name, epoch, prefix=f'after_g{group_id+1}')
                    model.load_state_dict(checkpoint_weights, strict=False)



            else:
                loader = train_loader_list[user_id]
                test_loader = test_loader_list[0]

                if args.eval_grad and epoch%args.epoch_interval == 0:
                    group_id = Get_group_num(args, groups, rank)
                    checkpoint_weights = util_1.Load_Avg_model_checkpoint(args.experiment_folder, args.experiment_name, epoch, prefix=f'before')
                    model.load_state_dict(checkpoint_weights, strict=False)

        while 1:
            break_flag = False
            train_loss = 0
            loss_steps = 0
            train_mask = 0

            for batch_idx, (data) in enumerate(loader):
                if args.user_semi:
                    data_x, data_u = data
                    inputs_x, targets_x = data_x
                    (inputs_u_w, inputs_u_s), _ = data_u

                    batch_size = inputs_x.shape[0]
                    inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s)).to(device)

                    targets_x = targets_x.to(device)
                    logits = model(inputs)
                    logits_x = logits[:batch_size]

                    logits_u_w = logits[batch_size:]
                    del logits

                    Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

                    pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
                    max_probs, targets_u = torch.max(pseudo_label, dim=-1)

                    if not args.eval_grad:
                        mask = max_probs.ge(0.95).float()
                    else:
                        mask = max_probs.ge(args.tao).float()
                        train_mask += max_probs.ge(0.95).float().sum().item()

                    Lu = (F.cross_entropy(logits_u_w, targets_u,
                                          reduction='none') * mask).mean()

                    loss = Lx + Lu
                else:

                    if user_id in set(server_list):
                        inputs_x, targets_x = data
                        inputs_x = inputs_x.to(device)
                        targets_x = targets_x.to(device)
                        output = model(inputs_x)
                        loss = criterion(output, targets_x)
                    else:
                        if args.labeled:
                            (inputs_u_w, inputs_u_s), target_labels = data

                            inputs_x = inputs_u_w.to(device)
                            targets_x = target_labels.to(device)
                            output = model(inputs_x)
                            loss = criterion(output, targets_x)
                        else:
                            if args.ue_loss == 'CRL':
                                (inputs_u_w, inputs_u_s), _ = data

                                inputs = torch.cat((inputs_u_w, inputs_u_s)).to(device)
                                logits = model(inputs)
                                logits_u_w, logits_u_s = logits.chunk(2)
                                del logits

                                pseudo_label = torch.softmax(logits_u_w.detach_(), dim=-1)
                                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                                if not args.eval_grad:
                                    mask = max_probs.ge(0.95).float()
                                else:

                                    mask = max_probs.ge(args.tao).float()
                                    train_mask += max_probs.ge(0.95).float().sum().item()

                                loss = (F.cross_entropy(logits_u_s, targets_u,
                                                      reduction='none') * mask).mean()
                                train_loss += loss.item()
                                loss_steps += 1

                            if args.ue_loss == 'SF':

                                inputs_x, targets_x = data

                                inputs = inputs_x.to(device)
                                model.eval()

                                with torch.no_grad():
                                    logits = model(inputs)


                                pseudo_label = torch.softmax(logits.detach_(), dim=-1)
                                max_probs, targets_u = torch.max(pseudo_label, dim=-1)

                                if not args.eval_grad:
                                    mask = max_probs.ge(0.95).float()
                                else:
                                    mask = max_probs.ge(args.tao).float()
                                    train_mask += max_probs.ge(0.95).float().sum().item()

                                model.train()
                                output = model(inputs)
                                loss = (F.cross_entropy(output, targets_u,
                                                      reduction='none') * mask).mean()

                loss.backward()

                if not args.eval_grad:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                if not args.eval_grad:
                    if iteration != 0 and iteration % args.cp*accum_steps == 0 :

                        if epoch%args.epoch_interval == 0 or epoch == args.epoch-1:
                            util_1.Save_model_checkpoint(args.experiment_name, model, rank, epoch)
                            group_id = Get_group_num(args, groups, rank)
                            save_each_group_avg_model(args, average_group_model_weights, epoch, rank, rank_save=groups[group_id][0], prefix=f'before_g{group_id+1}')
                            if rank == 0:
                                util_1.Save_Avg_model_checkpoint(args.experiment_name, average_model_weights, rank, epoch, prefix='before')

                        if args.user_semi:
                            if args.H:
                                ue_list = ue_list[0:args.num_comm_ue]
                                group1_size = len(ue_list)//2
                                group1 = np.array(ue_list)[np.arange(0, group1_size).tolist()].tolist()
                                group2 = np.array(ue_list)[np.arange(group1_size, len(ue_list)).tolist()].tolist()

                                if rank < len(ue_list)//2:
                                    #### Group 1 avgerage and communicate
                                    SyncAllreduce_1(model, rank, size=len(group1), group=G[0])
                                else:
                                    #### Group 2 avgerage and communicate
                                    SyncAllreduce_1(model, rank, size=len(group2), group=G[1])
                                if rank == 0 or rank == args.num_rank-1:
                                    SyncAllreduce_1(model, rank, size=len(groups[-1]), group=G[-1])
                            else:
                                SyncAllreduce(model, rank, args.num_rank)

                        else:
                            if args.H:
                                # print('Groupng method >>>>>')
                                average_group_model_weights = Grouping_Avg(args, model, rank, G, groups, epoch)
                            else:
                                SyncAllreduce(model, rank, args.num_rank)


                        average_model_weights = copy.deepcopy(model.state_dict())
                        if epoch%args.epoch_interval == 0 or epoch == args.epoch-1:
                            if rank == 0:
                                util_1.Save_Avg_model_checkpoint(args.experiment_name, average_model_weights, rank, epoch, prefix='after')

                        iteration += 1
                        break_flag = True

                        break
                iteration += 1

            if args.eval_grad:
                print(f"save grad. of the whole DataLoader of UE {user_id}")
                Save_model_grad_checkpoint(args.experiment_folder, args.experiment_name, model, rank, epoch, args.tao)
                ### save train_loss train_mask of this epoch
                values = {'train_loss': train_loss, 'train_mask':train_mask, 'len_loader':len(loader)}
                print(epoch,rank,values)
                Save_train_state(args.experiment_folder, args.experiment_name, rank, epoch, values, args.tao)
                break
            if break_flag:
                break

    return  user_id, WD_list, user_weight_diff_array

def init_processes(args, rank, size, fn, ip_address, master_port):
    os.environ['MASTER_ADDR'] = ip_address
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group('gloo', rank=rank, world_size=size)

    if args.H:
        groups = Get_torch_init_rank_group(args)
        G = []
        for g in range(len(groups)):
            group = groups[g]
            G_tmp = torch.distributed.new_group(ranks=group)
            G.append(G_tmp)
    else:
        G = torch.distributed.new_group(ranks=np.arange(0, size).tolist())


    torch.cuda.manual_seed(1)
    fn(rank, size, G)

if __name__ == "__main__":
    rank = args.rank
    world_size = args.num_rank
    master_port = args.master_port

    print(rank)
    print(args)
    ######### Assign Ranks to different GPUs
    GRU_list = [i for i in args.GPU_list]
    increase_tmp = (world_size+1)//len(GRU_list)
    ranks_list = np.arange(0, world_size).tolist()

    if args.num_rank == 52:
        ranks_list = np.arange(0,48).tolist()

    if args.num_rank == 33:
        ranks_list = np.arange(0,30).tolist()

    if args.num_rank == 21 or args.num_rank == 22:
        ranks_list = np.arange(0,18).tolist()


    rank_group = []
    for rank_id in range(len(GRU_list)):
        if rank_id == len(GRU_list)-1:
            ranks = ranks_list[rank_id*increase_tmp:]
        else:
            ranks = ranks_list[rank_id*increase_tmp:(rank_id+1)*increase_tmp]
        rank_group.append(ranks)

    for group_id in range(len(GRU_list)):
        if args.num_rank == 52:
            if args.rank >= 48 and args.rank < 52:
                os.environ["CUDA_VISIBLE_DEVICES"] = GRU_list[ args.rank-48 ]
            else:
                if args.rank in set(rank_group[group_id]):
                    os.environ["CUDA_VISIBLE_DEVICES"] = GRU_list[group_id]

        elif args.num_rank == 21:
            if args.rank >= 18 and args.rank < 21:
                os.environ["CUDA_VISIBLE_DEVICES"] = GRU_list[ args.rank-18 ]
            else:
                if args.rank in set(rank_group[group_id]):
                    os.environ["CUDA_VISIBLE_DEVICES"] = GRU_list[group_id]

        elif args.num_rank == 22:
            if args.rank >= 18 and args.rank < 22:
                os.environ["CUDA_VISIBLE_DEVICES"] = GRU_list[ args.rank-18 ]

            else:
                if args.rank in set(rank_group[group_id]):
                    os.environ["CUDA_VISIBLE_DEVICES"] = GRU_list[group_id]

        elif args.num_rank == 33:
            if args.rank >= 30 and args.rank < 33:
                os.environ["CUDA_VISIBLE_DEVICES"] = GRU_list[ args.rank-30 ]
            else:
                if args.rank in set(rank_group[group_id]):
                    os.environ["CUDA_VISIBLE_DEVICES"] = GRU_list[group_id]
        else:
            if args.rank in set(rank_group[group_id]):
                os.environ["CUDA_VISIBLE_DEVICES"] = GRU_list[group_id]
                break

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    init_processes(args, rank, world_size, run, args.ip_address, master_port)
