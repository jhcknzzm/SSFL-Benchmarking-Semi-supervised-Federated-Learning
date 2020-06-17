"""
##########
Assume the number of UEs is K
***************************************************************************************************************************************
size:              size = K + 1 (server);
cp:                cp in {2, 4, 8, 16} is frequency of communication; cp = 2 means UEs ans server communicates every 2 iterations;
basicLabelRatio:   basicLabelRatio in {0.1, 0.2, 0.3, 0.4, ..., 0.9, 1.0},  is the degree of data dispersion for each UE,
                   basicLabelRatio = 0.0 means UE has the same amount of samples in each class; basicLabelRatio = 1.0 samples owned
                   by UE all belong to the same class;
model:             model in {'res', 'res_gn'}; model = 'res' means we use ResNet18 + BN; model = 'res_gn' means we use ResNet18 + GN;
iid:               iid in {0, 1}; iid = 1 is the IID case; iid = 0 is the Non-IID case;
num_comm_ue:       num_comm_ue in {1, 2, ..., K}; a communication user number per iteration;
k_img:             the number of training samples used in one epoch;
H:                 H in {0, 1}; use grouping-based model average method or not; H = 1 means we use grouping-based method;
GPU_list:          GPU_list is a string; GPU_list = '01' means we use GPU0 and GPU1 for training;
num_data_server:   num_data_server in {4700}, number of labeled samples in server
***************************************************************************************************************************************

"""

import numpy as np
import scipy.io as scio

import os

path_setting = './Setting/emnist/'
if not os.path.exists(path_setting):
    os.makedirs(path_setting)

basesize = 47
basenum_data_server = 47*100


"""
Exper 1:
(1) 47 users, R = 0.4, Communication period = 16;
Server data number N_s = 4700, Number of participating clients C_k = 47.
"""
size = basesize + 1
batch_size = 64
basicLabelRatio = 0.4
iid = 0
num_comm_ue = 47
k_img = 65536
epoches = 100
H = 0
cp = [16]
model = ['EMNIST_model']
num_data_server = basenum_data_server

dictionary1 = {'size':size, 'batch_size':batch_size, 'cp':cp,
              'basicLabelRatio':basicLabelRatio, 'model':model, 'iid':iid,
              'num_comm_ue':num_comm_ue, 'k_img':k_img, 'epoches':epoches,
              'H':H, 'num_data_server':num_data_server}
np.save(path_setting+'Exper1_setting1.npy', dictionary1)

"""
Exper 1:
(2) 47 users, R = 0.4, Communication period = 16;
Server data number N_s = 4700, Number of participating clients C_k = 30.
"""
size = basesize + 1
batch_size = 64
basicLabelRatio = 0.4
iid = 0
num_comm_ue = 30
k_img = 65536
epoches = 100
H = 0
cp = [16]
model = ['EMNIST_model']
num_data_server = basenum_data_server

dictionary2 = {'size':size, 'batch_size':batch_size, 'cp':cp,
              'basicLabelRatio':basicLabelRatio, 'model':model, 'iid':iid,
              'num_comm_ue':num_comm_ue, 'k_img':k_img, 'epoches':epoches,
              'H':H, 'num_data_server':num_data_server}
np.save(path_setting+'Exper1_setting2.npy', dictionary2)


"""
Exper 1:
(2) 47 users, R = 0.4, Communication period = 16;
Server data number N_s = 4700, Number of participating clients C_k = 10.
"""
size = basesize + 1
batch_size = 64
basicLabelRatio = 0.4
iid = 0
num_comm_ue = 10
k_img = 65536
epoches = 100
H = 0
cp = [16]
model = ['EMNIST_model']
num_data_server = basenum_data_server

dictionary3 = {'size':size, 'batch_size':batch_size, 'cp':cp,
              'basicLabelRatio':basicLabelRatio, 'model':model, 'iid':iid,
              'num_comm_ue':num_comm_ue, 'k_img':k_img, 'epoches':epoches,
              'H':H, 'num_data_server':num_data_server}
np.save(path_setting+'Exper1_setting3.npy', dictionary3)



"""
Exper 2:
(2) 47 users, R = 0.4, Communication period = 16;
Server data number N_s = 4700, Number of participating clients C_k = 47.
"""
size = basesize + 1
batch_size = 64
basicLabelRatio = 0.4
iid = 0
num_comm_ue = 47
k_img = 65536
epoches = 100
H = 1
cp = [16]
model = ['EMNIST_model']
num_data_server = basenum_data_server

dictionary1 = {'size':size, 'batch_size':batch_size, 'cp':cp,
              'basicLabelRatio':basicLabelRatio, 'model':model, 'iid':iid,
              'num_comm_ue':num_comm_ue, 'k_img':k_img, 'epoches':epoches,
              'H':H, 'num_data_server':num_data_server}
np.save(path_setting+'Exper2_setting1.npy', dictionary1)
