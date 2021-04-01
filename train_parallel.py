#!/usr/bin/python
#!/usr/bin/python3

import threading
import time
import os
import numpy as np
import random

import gpustat
import logging

import itertools
import torch
import torch.optim as optim
import argparse
import sys
from scipy import io
import datetime
from utils_v2 import Get_num_ranks_all_size_num_devices

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

parser = argparse.ArgumentParser(description='SSFL training')

parser.add_argument('--GPU_list',
                    default='01',
                    type=str,
                    help='gpu list')
parser.add_argument('--datasetid',
                    default=0,
                    type=int,
                    help='dataset')
parser.add_argument('--basicLabelRatio',
                    default=0.4,
                    type=float,
                    help='basicLabelRatio')
parser.add_argument('--labeled',
                    default=0,
                    type=int,
                    help='supervised or not')
parser.add_argument('--num_comm_ue',
                    default=10,
                    type=int,
                    help='supervised or not')
parser.add_argument('--H',
                    default=1,
                    type=int,
                    help='Group or not')
parser.add_argument('--cp',
                    default=16,
                    type=int,
                    help='cp')

parser.add_argument('--eval_grad',
                    default=1,
                    type=int,
                    help='eval_grad or training')

parser.add_argument('--experiment_folder', default='.', type=str,
                    help='the path of the experiment')

parser.add_argument('--experiment_name', default=None, type=str,
                    help='experiment_name')

parser.add_argument('--tao',
                    default=0.95,
                    type=float,
                    help='tao for cal. mask')


parser.add_argument('--model', default='res_gn', type=str,
                    help='model')

parser.add_argument('--ue_loss', default='CRL', type=str,
                    help='user loss for training')

parser.add_argument('--user_semi',
                    default=0,
                    type=int,
                    help='user side semi')

parser.add_argument('--size',
                    default=11,
                    type=int,
                    help='user number + one server')
parser.add_argument('--epoch',
                    default=300,
                    type=int,
                    help='training epoch')

parser.add_argument('--batch_size',
                    default=64,
                    type=int,
                    help='batch_size for training')

parser.add_argument('--k_img',
                    default=65536,
                    type=int,
                    help='k_img')

parser.add_argument('--fast',
                    default=1,
                    type=int,
                    help='use fast model for lr scheduler or not')

parser.add_argument('--Ns',
                    default=1000,
                    type=int,
                    help='number of labeled data in server')

args = parser.parse_args()

FORMAT = '[%(asctime)-15s %(filename)s:%(lineno)s] %(message)s'
FORMAT_MINIMAL = '%(message)s'

logger = logging.getLogger('runner')
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)


exitFlag = 0
GPU_MEMORY_THRESHOLD = 24000 # MB?

def get_free_gpu_indices():
    '''
        Return an available GPU index.
    '''
    while True:
        stats = gpustat.GPUStatCollection.new_query()
        return_list = []
        for i, stat in enumerate(stats.gpus):
            memory_used = stat['memory.used']
            if memory_used < GPU_MEMORY_THRESHOLD:
                return i

        logger.info("Waiting on GPUs")
        time.sleep(5)


class DispatchThread(threading.Thread):
    def __init__(self, threadID, name, counter, bash_command_list):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.bash_command_list = bash_command_list

    def run(self):
#         logger.info("Starting " + self.name)
        threads = []
        for i, bash_command in enumerate(self.bash_command_list):

            cuda_device = get_free_gpu_indices()
            thread1 = ChildThread(1, f"{i}th + {bash_command}", 1, cuda_device, bash_command)
            thread1.start()
            import time
            time.sleep(5)
            threads.append(thread1)

        # join all.
        for t in threads:
            t.join()
        logger.info("Exiting " + self.name)



class ChildThread(threading.Thread):
    def __init__(self, threadID, name, counter, cuda_device, bash_command):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.cuda_device = cuda_device
        self.bash_command = bash_command

    def run(self):
        bash_command = self.bash_command

        # ACTIVATE
        os.system(bash_command)
        import time
        import random
        time.sleep(random.random() % 5)

        logger.info("Finishing " + self.name)

if args.datasetid == 0:
    dataset = 'cifar10'
if args.datasetid == 1:
    dataset = 'svhn'
if args.datasetid == 2:
    dataset = 'emnist'

"""
##########
Assume the number of UEs is K
***************************************************************************************************************************************
parameters         Value/meaning
size:              size = K + 1 (server);
cp:                cp in {2, 4, 8, 16} is frequency of communication; cp = 2 means UEs and server communicates every 2 iterations;
basicLabelRatio:   basicLabelRatio in {0.1, 0.2, 0.4, ..., 1.0},  is the degree of data dispersion for each UE,
                   basicLabelRatio = 0.0 means UE has the same amount of samples in each class; basicLabelRatio = 1.0 means all samples owned
                   by UE belong to the same class;
model:             model in {'res', 'res_gn'}; model = 'res' means we use ResNet18 + BN; model = 'res_gn' means we use ResNet18 + GN;
iid:               iid in {0, 1}; iid = 1 is the IID case; iid = 0 is the Non-IID case;
num_comm_ue:       num_comm_ue in {1, 2, ..., K}; communication user number per iteration;
k_img:             Total data volume after data augmentation for each UE and server;
H:                 H in {0, 1}; use grouping-based method or not; H = 1 means we use grouping-based method;
GPU_list:          GPU_list is a string; GPU_list = '01' means we use GPU0 and GPU1 for training;
num_data_server:   num_data_server in {1000, 4000}, number of labeled samples in server
master_port:       a random string; MASTER_PORT
ip_address:        a string; MASTER_ADDR

For examples:
size = 5 + 1
batch_size = 64
cp = 4
basicLabelRatio = 0.4
model = 'res_gn'
iid = 0
num_comm_ue = 2
k_img = 65536
epoches = 300
H = 1
num_data_server = 1000
***************************************************************************************************************************************
"""


size = args.size
batch_size = args.batch_size
cp_list = [args.cp]

basicLabelRatio = args.basicLabelRatio
model_list = [args.model]

if basicLabelRatio == 0.0:
    iid = 1
else:
    iid = 0

num_comm_ue = args.num_comm_ue
k_img = args.k_img
epoches = args.epoch
warmup_epoch = 5
num_data_server = args.Ns

labeled = args.labeled
fast = args.fast
H = args.H
epoch_interval = epoches//10

GPU_list = args.GPU_list

import socket
myname = socket.getfqdn(socket.gethostname(  ))
myaddr = socket.gethostbyname(myname)
print('The ip address:',myaddr)
ip_address = myaddr

class_per_device = 1
Start_Epoch = [0]
num_rank, size_all, num_devices = Get_num_ranks_all_size_num_devices(args)

now_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

if args.experiment_name is None:
    experiment_name = f'{dataset}_size_all_{size_all}_UE{num_devices}_comUE{num_comm_ue}_cp{cp_list[0]}_Model{model_list[0]}_H{H}_labeled_{labeled}_Ns_{num_data_server}_eval_grad_{args.eval_grad}_Time_{now_time}'
else:
    experiment_name = args.experiment_name

### submitte models to uers' device for training
for model in model_list:
    for cp in cp_list:
        for epoch_resume in Start_Epoch:
            master_port = random.sample(range(10000,30000),1)
            master_port = str(master_port[0])
            BASH_COMMAND_LIST = []
            for rank in range(num_rank):
                lr = 0.03*(10.0+1.0)*batch_size/128.0
                if args.model == 'res9':
                    lr = 0.003*num_comm_ue*batch_size/128.0
                if dataset == 'emnist' or args.ue_loss == 'SF':
                    lr = 0.03
                    warmup_epoch = 0
                if args.user_semi and args.model != 'res9':
                    lr = 0.01
                    warmup_epoch = 0

                comm = f"setsid python train_LocalSGD.py --dataset {dataset} --model {model} --eval_grad {args.eval_grad} --epoch_resume {epoch_resume} --epoch_interval {epoch_interval}\
                                        --lr {lr} --bs {batch_size} --cp {cp} --alpha 0.6 --gmf 0.7 --basicLabelRatio {basicLabelRatio} --master_port {master_port}\
                                        --name revised_results_e300 --ip_address {ip_address} --num_comm_ue {num_comm_ue} --num_data_server {num_data_server} --k-img {k_img}\
                                        --iid {iid} --rank {rank} --size {size_all} --backend gloo --warmup_epoch {warmup_epoch} --GPU_list {GPU_list} --labeled {labeled}\
                                        --class_per_device {1} --num-devices {num_devices} --num_rank {num_rank} --epoch {epoches} --experiment_name {experiment_name} --fast {fast} --H {H} \
                                        --experiment_folder {args.experiment_folder} --tao {args.tao} --ue_loss {args.ue_loss} --user_semi {args.user_semi}"


                BASH_COMMAND_LIST.append(comm)


            dispatch_thread = DispatchThread(2, "Thread-2", 4, BASH_COMMAND_LIST)
            # # Start new Threads
            dispatch_thread.start()
            dispatch_thread.join()

            import time
            time.sleep(5)
