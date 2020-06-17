#!/usr/bin/python
#!/usr/bin/python3

# This script assume exclusive usage of the GPUs.
# If you have limited usage of GPUs, you can limit the range of gpu indices you are using.
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

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

parser = argparse.ArgumentParser(description='CIFAR-10 baseline')
parser.add_argument('--experiment_num',
                    default=0,
                    type=int,
                    help='serial number of the experiment')
parser.add_argument('--ip_address',
                    default="169.229.49.63",
                    type=str,
                    help='ip_address')
parser.add_argument('--GPU_list',
                    default='0123',
                    type=str,
                    help='gpu list')
parser.add_argument('--datasetid',
                    default=0,
                    type=int,
                    help='dataset')

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
        time.sleep(10)


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
            time.sleep(10)
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

if args.datasetid==0:
    dataset = 'cifar10'
if args.datasetid==1:
    dataset = 'svhn'
if args.datasetid==2:
    dataset = 'emnist'

####### run setting.py to generate Setting files for different experiments
if dataset == 'cifar10':
    if not os.path.exists('./Setting/cifar/'):
        comm = "python setting_cifar.py"
        os.system(comm)

if dataset == 'emnist':
    if not os.path.exists('./Setting/emnist/'):
        comm = "python setting_emnist.py"
        os.system(comm)

if dataset == 'svhn':
    if not os.path.exists('./Setting/svhn/'):
        comm = "python setting_svhn.py"
        os.system(comm)
###### generate device_idxs server_idxs for different experiments
print(dataset,dataset=="cifar10")

if dataset == "cifar10":
    size_list = [10, 20, 30]
    # size_list = [30]
    iid_list = [0]
    basicLabelRatio_list = [0.2, 0.4, 0.6, 0.8, 1.0]
    # basicLabelRatio_list = [0.4]

    num_data_server_list = [1000]
    class_per_device = 1

if dataset == 'emnist':
    size_list = [47]
    iid_list = [0,1]
    basicLabelRatio_list = [0.4]

    num_data_server_list = [4700]#[3100,6200]
    class_per_device = 1

if dataset == 'svhn':
    size_list = [10, 20, 30]
    iid_list = [0,1]
    basicLabelRatio_list = [0.2, 0.4, 0.6, 0.8, 1.0]

    num_data_server_list = [1000]
    class_per_device = 1
# Data pre-processing
for num_data_server in num_data_server_list:
    for iid in iid_list:
        for size in size_list:
            if iid:
                path_device_idxs = f'{dataset}_post_data/iid/{size}_{num_data_server}'
                print(path_device_idxs)
                if not os.path.exists(path_device_idxs):
                    comm = f"python data_precessing.py --dataset {dataset} \
                            --num_devices {size} \
                            --num_data_server {num_data_server} \
                            --iid {iid} \
                            --path_device_idxs {path_device_idxs}"
                    os.system(comm)
            else:
                for basicLabelRatio in basicLabelRatio_list:
                    path_device_idxs = f'{dataset}_post_data/noniid/{size}_{num_data_server}_{class_per_device}_{basicLabelRatio}'
                    print(path_device_idxs)
                    if not os.path.exists(path_device_idxs):
                        comm = f"python data_precessing.py --dataset {dataset} \
                                --num_devices {size} \
                                --num_data_server {num_data_server} \
                                --iid {iid} \
                                --path_device_idxs {path_device_idxs}\
                                --basicLabelRatio {basicLabelRatio}\
                                --class_per_device {class_per_device}"
                        os.system(comm)



BASH_COMMAND_LIST = []

##################
### load setting.npy file to get the value of parameters
if dataset == 'cifar10':
    path_setting = './Setting/cifar/'
    setting_list = ['Exper1_setting1', 'Exper1_setting2', 'Exper1_setting3', 'Exper1_setting4','Exper1_setting5', 'Exper1_setting6',
                    'Exper2_setting1', 'Exper2_setting2', 'Exper2_setting3', 'Exper2_setting4',
                    'Exper3_setting1', 'Exper3_setting2', 'Exper3_setting3',
                    'Exper4_setting1', 'Exper4_setting2', 'Exper4_setting3', 'Exper4_setting4',
                    'Exper5_setting1',
                    'Exper6_setting1']

if dataset == 'emnist':
    path_setting = './Setting/emnist/'
    setting_list = ['Exper1_setting1', 'Exper1_setting2', 'Exper1_setting3', 'Exper1_setting4','Exper1_setting5', 'Exper1_setting6',
                    'Exper2_setting1', 'Exper2_setting2', 'Exper2_setting3', 'Exper2_setting4',
                    'Exper3_setting1', 'Exper3_setting2', 'Exper3_setting3',
                    'Exper4_setting1', 'Exper4_setting2', 'Exper4_setting3', 'Exper4_setting4',
                    'Exper5_setting1',
                    'Exper6_setting1', 'Exper6_setting2']

if dataset == 'svhn':
    path_setting = './Setting/svhn/'
    setting_list = ['Exper1_setting1', 'Exper1_setting2', 'Exper1_setting3',
                    'Exper2_setting1']
setting = np.load(path_setting + setting_list[args.experiment_num] + '.npy', allow_pickle=True).item()

"""
##########
Assume the number of UEs is K
***************************************************************************************************************************************
parameters         Value/meaning
size:              size = K + 1 (server);
cp:                cp in {2, 4, 8, 16} is frequency of communication; cp = 2 means UEs and server communicates every 2 iterations;
basicLabelRatio:   basicLabelRatio in {0.1, 0.2, 0.3, 0.4, ..., 0.9, 1.0},  is the degree of data dispersion for each UE,
                   basicLabelRatio = 0.0 means UE has the same amount of samples in each class; basicLabelRatio = 1.0 means all samples owned
                   by UE belong to the same class;
model:             model in {'res', 'res_gn'}; model = 'res' means we use ResNet18 + BN; model = 'res_gn' means we use ResNet18 + GN;
iid:               iid in {0, 1}; iid = 1 is the IID case; iid = 0 is the Non-IID case;
num_comm_ue:       num_comm_ue in {1, 2, ..., K}; a communication user number per iteration;
k_img:             the number of training samples used in one epoch;
H:                 H in {0, 1}; use grouping-based model average method or not; H = 1 means we use grouping-based method;
GPU_list:          GPU_list is a string; GPU_list = '01' means we use GPU0 and GPU1 for training;
num_data_server:   num_data_server in {1000, 4000}, number of labeled samples in server
master_port:       a random string; MASTER_PORT
ip_address:        a string; MASTER_ADDR

For examples:
size = 10 + 1
batch_size = 64
cp = 4
basicLabelRatio = 0.4
model = 'res_gn'
iid = 0
num_comm_ue = 10
k_img = 65536
epoches = 300
H = 0
num_data_server = 1000
***************************************************************************************************************************************
"""
size = setting['size']
batch_size = setting['batch_size']
cp_list = setting['cp']
basicLabelRatio = setting['basicLabelRatio']
model_list = setting['model']
iid = setting['iid']
num_comm_ue = setting['num_comm_ue']
k_img = setting['k_img']
epoches = setting['epoches']
H = setting['H']
num_data_server = setting['num_data_server']
experiment_name = setting_list[args.experiment_num]
labeled = 0
######### GPUs
GPU_list = args.GPU_list
ip_address = args.ip_address



ue_list_epoch = [[]]*epoches
iterations_epoch = k_img//batch_size
ue_list_epoch = np.zeros((epoches,iterations_epoch,num_comm_ue+1),dtype='int32')

### If the number of communication user < K; We randomly generate a subset (ue_list) of users
if num_comm_ue < size - 1:
    for e in range(epoches):
        for it in range(iterations_epoch):
            ue_list = np.arange(1, size).tolist()
            random.shuffle(ue_list)
            ue_list = [0] + random.sample(ue_list,num_comm_ue)
            ue_list.sort()
            ue_list = np.array(ue_list, dtype='int32')
            ue_list_epoch[e,it,0:len(ue_list)] = ue_list

    ue_list_epoch = np.array(ue_list_epoch, dtype='int32')
    dictionary1 = {'ue_list_epoch':ue_list_epoch.tolist()}
    path_ue_list_epoch = f'{dataset}_post_data/noniid/{size - 1}_{num_data_server}_{class_per_device}_{basicLabelRatio}/'
    if iid:
        path_ue_list_epoch = f'{dataset}_post_data/iid/{size - 1}_{num_data_server}/'
    np.save(path_ue_list_epoch + "ue_list_epoch.npy", dictionary1)

### run train_LocalSGD.py to train the server model and user model
for model in model_list:
    for cp in cp_list:
        master_port = random.sample(range(10000,30000),1)
        master_port = str(master_port[0])
        BASH_COMMAND_LIST = []
        for rank in range(size + H):
            if H:
                if rank < (size+H)//2:
                    lr_increase = float((size+H)//2)
                    lr = 0.03*lr_increase*batch_size/128.0
                else:
                    lr_increase = float( (size+H) - (size+H)//2 )
                    lr = 0.03*lr_increase*batch_size/128.0
            else:
                lr = 0.03*(10.0+1.0)*batch_size/128.0

            if dataset == 'emnist':
                lr = 0.03

            comm = f"setsid python train_LocalSGD.py --dataset {dataset} --model {model}  \
                                    --lr {lr} --bs {batch_size} --cp {cp} --alpha 0.6 --gmf 0.7 --basicLabelRatio {basicLabelRatio} --master_port {master_port}\
                                    --name revised_results_e300 --ip_address {ip_address} --num_comm_ue {num_comm_ue} --num_data_server {num_data_server}\
                                    --iid {iid} --rank {rank} --size {size + H} --backend gloo --H {H} --GPU_list {GPU_list} --labeled {labeled}\
                                    --class_per_device {1} --num-devices {size - 1} --epoch {epoches} --experiment_name {experiment_name}"
            BASH_COMMAND_LIST.append(comm)

        dispatch_thread = DispatchThread(2, "Thread-2", 4, BASH_COMMAND_LIST)
        # # Start new Threads
        dispatch_thread.start()
        dispatch_thread.join()

        import time
        time.sleep(5)
