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
import datetime
from scipy.io import loadmat
import json
from scipy import io
import utils_v2 as util_1
import math
import copy
import torch.distributed as dist

from utils_v2 import *


parser = argparse.ArgumentParser(description='Calculating gradient diversity')
parser.add_argument('--experiment_name', default='EMNIST_size47_comUE10_H1_R0.4_SSFL', type=str,
                    help='path to checkpoint')
parser.add_argument('--experiment_folder', default='.', type=str,
                    help='the path of the experiment')

parser.add_argument('--vector_type', default='grad', type=str,
                    help='vector_type = grad or weight_variation')

parser.add_argument('--l2',
                    default=1,
                    type=int,
                    help='l2 norm or l1 norm')

parser.add_argument('--only_user',
                    default=0,
                    type=int,
                    help='only user or not')

parser.add_argument('--ord',
                    default=2,
                    type=int,
                    help='norm ord = 1 or 2')

args = parser.parse_args()

def get_groups(args):
    if 'EMNIST' in args.experiment_name:
        if 'comUE10' in args.experiment_name:
            server_list = [0,48]
            group1 = [0] + np.arange(1, 6).tolist()
            group2 = [11] + np.arange(6, 11).tolist()
            group3 = [0, 11]
            groups = [group1, group2, group3]

        if 'comUE30' in args.experiment_name:
            server_list = [0,48,49]
            group1 = [0] + np.arange(1, 11).tolist()
            group2 = [31] + np.arange(11, 21).tolist()
            group3 = [32] + np.arange(21, 31).tolist()
            group4 = [0,31,32]
            groups = [group1, group2, group3, group4]

        if 'comUE47' in args.experiment_name:
            server_list = [0,48,49,50,51]
            group1 = [0] + np.arange(1, 11).tolist()
            group2 = [48] + np.arange(11, 21).tolist()
            group3 = [49] + np.arange(21, 31).tolist()
            group4 = [50] + np.arange(31, 41).tolist()
            group5 = [51] + np.arange(41, 48).tolist()
            group6 = [0, 48, 49, 50, 51]
            groups = [group1, group2, group3, group4, group5, group6]

    if 'SVHN' in args.experiment_name:
        if 'commUE10' in args.experiment_name:
            server_list = [0,31]
            group1 = [0] + np.arange(1, 6).tolist()
            group2 = [11] + np.arange(6, 11).tolist()
            group3 = [0, 11]
            groups = [group1, group2, group3]

        if 'commUE20' in args.experiment_name:
            server_list = [0,31]
            group1 = [0] + np.arange(1, 11).tolist()
            group2 = [21] + np.arange(11, 21).tolist()
            group3 = [0,21]
            groups = [group1, group2, group3]

        if 'commUE30' in args.experiment_name:
            server_list = [0,48,49,50,51]
            group1 = [0] + np.arange(1, 11).tolist()
            group2 = [31] + np.arange(11, 21).tolist()
            group3 = [32] + np.arange(21, 31).tolist()
            group4 = [0, 31, 32]
            groups = [group1, group2, group3, group4]


    return groups, server_list

def Load_model_grad_checkpoint(experiment_folder='.', experiment_name=None, rank=0, epoch=10, tao=0.95):
    try:
        path_checkpoint = '%s/grad_checkpoint/%s/' %(experiment_folder, experiment_name)
        pthfile = path_checkpoint+'Rank%s_Epoch_%s_model_grads_tao_%s.pth' %(rank, epoch, tao)
        checkpoint_grads = torch.load(pthfile, map_location=lambda storage, loc: storage)
    except:
        path_checkpoint = '%s/checkpoint/%s/' %(experiment_folder, experiment_name)
        pthfile = path_checkpoint+'Rank%s_Epoch_%s_model_grads_tao_%s.pth' %(rank, epoch, tao)
        checkpoint_grads = torch.load(pthfile, map_location=lambda storage, loc: storage)

    return checkpoint_grads

def Load_Avgmodel_weights(experiment_folder, experiment_name, epoch):
    path_checkpoint = '%s/checkpoint/%s/' %(experiment_folder, experiment_name)
    pthfile = path_checkpoint+'Avg_before_Epoch_%s_weights.pth' %(epoch)
    checkpoint_weights = torch.load(pthfile, map_location=lambda storage, loc: storage)
    try:
        checkpoint_weights = checkpoint_weights['state_dict']
    except:
        pass
    return checkpoint_weights

def Load_Avgmodel_weights_with_middle(experiment_folder, experiment_name, epoch, middle):
    path_checkpoint = '%s/checkpoint/%s/' %(experiment_folder, experiment_name)
    pthfile = path_checkpoint+'Avg_%s_Epoch_%s_weights.pth' %(middle, epoch)
    checkpoint_weights = torch.load(pthfile, map_location=lambda storage, loc: storage)
    try:
        checkpoint_weights = checkpoint_weights['state_dict']
    except:
        pass
    return checkpoint_weights

def Load_model_weight_checkpoint(experiment_folder='.', experiment_name=None, rank=0, epoch=10):
    path_checkpoint = '%s/checkpoint/%s/' %(experiment_folder, experiment_name)
    pthfile = path_checkpoint+'Rank%s_Epoch_%s_weights.pth' %(rank, epoch)
    checkpoint_weights = torch.load(pthfile, map_location=lambda storage, loc: storage)
    try:
        checkpoint_weights = checkpoint_weights['state_dict']
    except:
        pass

    return checkpoint_weights

if 'EMNIST' in args.experiment_name:
    E_list = [0,10,20,30,40,50,60,70,80,90]


if 'cifar' in args.experiment_name:
    E_list = [0,30,60,90,120,150,180,210,240,270]

if 'SVHN' in args.experiment_name:
    E_list = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]

if 'H1' in args.experiment_name:
    Grouping_method = True
    if 'comUE10' in args.experiment_name or 'commUE10' in args.experiment_name:
        num_rank = 12
    if 'comUE20' in args.experiment_name or 'commUE20' in args.experiment_name:
        num_rank = 22
    if 'comUE30' in args.experiment_name or 'commUE30' in args.experiment_name:
        num_rank = 33
    if 'comUE47' in args.experiment_name or 'commUE47' in args.experiment_name:
        num_rank = 52

else:
    Grouping_method = False
    if 'comUE10' in args.experiment_name or 'commUE10' in args.experiment_name:
        num_rank = 11
    if 'comUE20' in args.experiment_name or 'commUE20' in args.experiment_name:
        num_rank = 21
    if 'comUE30' in args.experiment_name or 'commUE30' in args.experiment_name:
        num_rank = 31
    if 'comUE47' in args.experiment_name or 'commUE47' in args.experiment_name:
        num_rank = 48



if not Grouping_method:
    groups, server_list = get_groups(args)
    W_e_list_norm_mean_list = []
    state_train_loss_epoch_list = []
    state_train_masks_epoch_list = []
    for e in E_list:
        W_e_list = []
        state_train_loss_list = []
        state_train_masks_list = []

        for rank in range(num_rank):
            if rank in set(groups[-1]) and args.only_user:
                pass
            else:
                if args.vector_type == 'grad':
                    W_rank_e = Load_model_grad_checkpoint(args.experiment_folder, args.experiment_name, rank=rank, epoch=e)

                elif args.vector_type == 'weight_variation':
                    W = Load_model_weight_checkpoint(args.experiment_folder, args.experiment_name, rank=rank, epoch=e)
                    W_i = get_params(W, WBN=False)
                    W_avg = Load_Avgmodel_weights(args.experiment_folder, args.experiment_name, epoch=e)
                    W_avg = get_params(W_avg, WBN=False)
                    W_rank_e = W_i - W_avg


                W_rank_e_norm = torch.norm(W_rank_e)
                print(f'epoch={e},rank={rank},grad_norm={W_rank_e_norm}')
                W_e_list.append(W_rank_e.numpy())
                try:
                    state_values = Load_train_state(args.experiment_folder, args.experiment_name, rank=rank, epoch=e)

                    state_train_loss = state_values.item()['train_loss']
                    state_train_masks = state_values.item()['train_mask']
                except:
                    state_train_loss = 0
                    state_train_masks = 0

                state_train_loss_list.append(state_train_loss)
                state_train_masks_list.append(state_train_masks)

        W_e_list = np.array(W_e_list)
        W_e_list_mean = np.mean(W_e_list, axis=0)


        state_train_loss_epoch_list.append(round(np.mean(state_train_loss_list),2))
        state_train_masks_epoch_list.append(np.mean(state_train_masks_list))

        W_e_list_norm_list = []
        for i in range(len(W_e_list)):
            W_Wavg_norm = 0

            W_Wavg_norm =  np.linalg.norm(W_e_list[i] -W_e_list_mean)
            W_e_list_norm = np.linalg.norm(W_e_list[i],ord=args.ord)/np.linalg.norm(W_e_list_mean,ord=args.ord)
            if args.l2:
                W_e_list_norm = W_e_list_norm**2

            print(f'epoch={e}, i={i}, norm of (g-E[g]) = {W_Wavg_norm}')
            W_e_list_norm_list.append(W_e_list_norm)

        W_e_list_norm_mean = np.mean(W_e_list_norm_list)
        W_e_list_norm_mean_list.append(W_e_list_norm_mean)

        print(f'epoch={e}, grad_var={W_e_list_norm_mean}')
    print('Grouping_method = False',W_e_list_norm_mean_list)
    print(state_train_loss_epoch_list)
    print(state_train_masks_epoch_list)
    print('############')

    save_var_path = f'./diversity_ord_{args.ord}/'
    if args.vector_type == 'grad':
        save_var_path = save_var_path + 'grad_FedAvg_post/'
    elif args.vector_type == 'weight_variation':
        save_var_path = save_var_path + 'weight_variation_FedAvg_post/'
    save_var_path = save_var_path + f'{args.experiment_name}/'
    if not os.path.exists(save_var_path):
        os.makedirs(save_var_path)
    np.save(save_var_path+f'{args.vector_type}_diversity_list_l2_{args.l2}_only_user_{args.only_user}.npy',W_e_list_norm_mean_list)


else:
    groups, server_list = get_groups(args)
    W_e_list_norm_mean_list = []
    state_train_loss_epoch_list = []
    state_train_masks_epoch_list = []

    num_group = len(groups[0:-1])
    for e in E_list:
        W_e_list = [[] for i in range(num_group)]

        state_train_loss_list = []
        state_train_masks_list = []

        for rank in range(num_rank):
            if rank in set(groups[-1]) and args.only_user:
                pass
            else:
                group_id = Get_group_num(args, groups, rank)
                if args.vector_type == 'grad':
                    W_rank_e = Load_model_grad_checkpoint(args.experiment_folder, args.experiment_name, rank=rank, epoch=e)

                elif args.vector_type == 'weight_variation':
                    W = Load_model_weight_checkpoint(args.experiment_folder, args.experiment_name, rank=rank, epoch=e)
                    W_i = get_params(W, WBN=False)
                    W_avg = Load_Avgmodel_weights_with_middle(args.experiment_folder, args.experiment_name, epoch=e, middle=f'before_g{group_id+1}')
                    W_avg = get_params(W_avg, WBN=False)
                    W_rank_e = W_i - W_avg

                W_rank_e_norm = torch.norm(W_rank_e)
                print(f'epoch={e},rank={rank},grad_norm={W_rank_e_norm}')

                W_e_list[group_id].append(W_rank_e.numpy())

                try:
                    state_values = Load_train_state(args.experiment_folder, args.experiment_name, rank=rank, epoch=e)

                    state_train_loss = state_values.item()['train_loss']
                    state_train_masks = state_values.item()['train_mask']
                except:
                    state_train_loss = 0
                    state_train_masks = 0

                state_train_loss_list.append(state_train_loss)
                state_train_masks_list.append(state_train_masks)

        W_e_var_list = []
        W_e_list_norm_list = []
        for i in range(num_group):
            W_e = np.array(W_e_list[i])
            W_e_mean = np.mean(W_e, axis=0)

            for j in range(len(W_e_list[i])):
                W_Wavg_norm =  np.linalg.norm(W_e[j]-W_e_mean)
                W_e_list_norm = np.linalg.norm(W_e[j],ord=args.ord)/np.linalg.norm(W_e_mean,ord=args.ord)
                if args.l2:
                    W_e_list_norm = W_e_list_norm**2

                print(f'epoch={e}, j={j}, norm of (g-E[g]) = {W_Wavg_norm}')
                W_e_list_norm_list.append(W_e_list_norm)

        W_e_list_norm_mean = np.mean(W_e_list_norm_list)
        W_e_list_norm_mean_list.append(W_e_list_norm_mean)

        state_train_loss_epoch_list.append(round(np.mean(state_train_loss_list),2))
        state_train_masks_epoch_list.append(np.mean(state_train_masks_list))

        print(f'epoch={e}, grad_var={W_e_list_norm_mean}')
    print('Grouping_method = True',W_e_list_norm_mean_list)
    print(state_train_loss_epoch_list)
    print(state_train_masks_epoch_list)
    print('############')

    save_var_path = f'./diversity_ord_{args.ord}/'
    if args.vector_type == 'grad':
        save_var_path = save_var_path + 'grad_Group_post/'
    elif args.vector_type == 'weight_variation':
        save_var_path = save_var_path + 'weight_variation_Group_post/'

    save_var_path = save_var_path + f'{args.experiment_name}/'
    if not os.path.exists(save_var_path):
        os.makedirs(save_var_path)
    np.save(save_var_path+f'{args.vector_type}_diversity_list_l2_{args.l2}_only_user_{args.only_user}.npy',W_e_list_norm_mean_list)
