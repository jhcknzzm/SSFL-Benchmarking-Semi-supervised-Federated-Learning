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
iid:               iid in {0, 1}; iid = 1 is the  ; iid = 0 is the Non- ;
num_comm_ue:       num_comm_ue in {1, 2, ..., K}; a communication user number per iteration;
k_img:             the number of training samples used in one epoch;
H:                 H in {0, 1}; use grouping-based model average method or not; H = 1 means we use grouping-based method;
GPU_list:          GPU_list is a string; GPU_list = '01' means we use GPU0 and GPU1 for training;
num_data_server:   num_data_server in {1000, 4000}, number of labeled samples in server
***************************************************************************************************************************************

"""

import numpy as np
import scipy.io as scio

import os

path_setting = './Setting/svhn/'
if not os.path.exists(path_setting):
    os.makedirs(path_setting)

"""
Exper 1:
(1) 10 users, each one only has the accessto one class data R = 1.0, Communication period = 16;
Server data number N_s = 1000, Number of participating clients C_k = 10; ResNet18 with group normalization
is used for training.
"""
size = 10 + 1
batch_size = 64
basicLabelRatio = 1.0
iid = 0
num_comm_ue = 10
k_img = 65536
epoches = 40
H = 0
cp = [16]
model = ['res_gn']
num_data_server = 1000

dictionary1 = {'size':size, 'batch_size':batch_size, 'cp':cp,
              'basicLabelRatio':basicLabelRatio, 'model':model, 'iid':iid,
              'num_comm_ue':num_comm_ue, 'k_img':k_img, 'epoches':epoches,
              'H':H, 'num_data_server':num_data_server}
np.save(path_setting+'Exper1_setting1.npy', dictionary1)

"""
Exper 1:
(2) 10 users,   R = 0.0, Communication period = 16;
Server data number N_s = 1000, Number of participating clients C_k = 10; ResNet18 with group normalization
is used for training.
"""
size = 10 + 1
batch_size = 64
basicLabelRatio = 0.0
iid = 1
num_comm_ue = 10
k_img = 65536
epoches = 40
H = 0
cp = [16]
model = ['res_gn']
num_data_server = 1000

dictionary2 = {'size':size, 'batch_size':batch_size, 'cp':cp,
              'basicLabelRatio':basicLabelRatio, 'model':model, 'iid':iid,
              'num_comm_ue':num_comm_ue, 'k_img':k_img, 'epoches':epoches,
              'H':H, 'num_data_server':num_data_server}
np.save(path_setting+'Exper1_setting2.npy', dictionary2)


"""
Exper 1:
(3) 10 users,   R = 0.2, Communication period = 16;
Server data number N_s = 1000, Number of participating clients C_k = 10; ResNet18 with group normalization
is used for training.
"""
size = 10 + 1
batch_size = 64
basicLabelRatio = 0.2
iid = 0
num_comm_ue = 10
k_img = 65536
epoches = 40
H = 0
cp = [16]
model = ['res_gn']
num_data_server = 1000

dictionary3 = {'size':size, 'batch_size':batch_size, 'cp':cp,
              'basicLabelRatio':basicLabelRatio, 'model':model, 'iid':iid,
              'num_comm_ue':num_comm_ue, 'k_img':k_img, 'epoches':epoches,
              'H':H, 'num_data_server':num_data_server}
np.save(path_setting+'Exper1_setting3.npy', dictionary3)


"""
Exper 1:
(4) 10 users,   R = 0.4, Communication period = 16;
Server data number N_s = 1000, Number of participating clients C_k = 10; ResNet18 with group normalization
is used for training.
"""
size = 10 + 1
batch_size = 64
basicLabelRatio = 0.4
iid = 0
num_comm_ue = 10
k_img = 65536
epoches = 40
H = 0
cp = [16]
model = ['res_gn']
num_data_server = 1000

dictionary4 = {'size':size, 'batch_size':batch_size, 'cp':cp,
              'basicLabelRatio':basicLabelRatio, 'model':model, 'iid':iid,
              'num_comm_ue':num_comm_ue, 'k_img':k_img, 'epoches':epoches,
              'H':H, 'num_data_server':num_data_server}
np.save(path_setting+'Exper1_setting4.npy', dictionary4)



"""
Exper 1:
(5) 10 users,   R = 0.6, Communication period = 16;
Server data number N_s = 1000, Number of participating clients C_k = 10; ResNet18 with group normalization
is used for training.
"""
size = 10 + 1
batch_size = 64
basicLabelRatio = 0.6
iid = 0
num_comm_ue = 10
k_img = 65536
epoches = 40
H = 0
cp = [16]
model = ['res_gn']
num_data_server = 1000

dictionary5 = {'size':size, 'batch_size':batch_size, 'cp':cp,
              'basicLabelRatio':basicLabelRatio, 'model':model, 'iid':iid,
              'num_comm_ue':num_comm_ue, 'k_img':k_img, 'epoches':epoches,
              'H':H, 'num_data_server':num_data_server,}
np.save(path_setting+'Exper1_setting5.npy', dictionary5)


"""
Exper 1:
(6) 10 users,   R = 0.8, Communication period = 16;
Server data number N_s = 1000, Number of participating clients C_k = 10; ResNet18 with group normalization
is used for training.
"""
size = 10 + 1
batch_size = 64
basicLabelRatio = 0.8
iid = 0
num_comm_ue = 10
k_img = 65536
epoches = 40
H = 0
cp = [16]
model = ['res_gn']
num_data_server = 1000

dictionary6 = {'size':size, 'batch_size':batch_size, 'cp':cp,
              'basicLabelRatio':basicLabelRatio, 'model':model, 'iid':iid,
              'num_comm_ue':num_comm_ue, 'k_img':k_img, 'epoches':epoches,
              'H':H, 'num_data_server':num_data_server,}
np.save(path_setting+'Exper1_setting6.npy', dictionary5)


"""
Exper 2:
(1) 10 users,   R = 0.4, Communication period = 2;
Server data number N_s = 1000, Number of participating clients C_k = 10; ResNet18 with group normalization
is used for training.
"""
size = 10 + 1
batch_size = 64
basicLabelRatio = 0.4
iid = 0
num_comm_ue = 10
k_img = 65536
epoches = 40
H = 0
cp = [2]
model = ['res_gn']
num_data_server = 1000

dictionary1 = {'size':size, 'batch_size':batch_size, 'cp':cp,
              'basicLabelRatio':basicLabelRatio, 'model':model, 'iid':iid,
              'num_comm_ue':num_comm_ue, 'k_img':k_img, 'epoches':epoches,
              'H':H, 'num_data_server':num_data_server,}
np.save(path_setting+'Exper2_setting1.npy', dictionary1)

"""
Exper 2:
(2) 10 users,   R = 0.4, Communication period = 4;
Server data number N_s = 1000, Number of participating clients C_k = 10; ResNet18 with group normalization
is used for training.
"""
size = 10 + 1
batch_size = 64
basicLabelRatio = 0.4
iid = 0
num_comm_ue = 10
k_img = 65536
epoches = 40
H = 0
cp = [4]
model = ['res_gn']
num_data_server = 1000

dictionary2 = {'size':size, 'batch_size':batch_size, 'cp':cp,
              'basicLabelRatio':basicLabelRatio, 'model':model, 'iid':iid,
              'num_comm_ue':num_comm_ue, 'k_img':k_img, 'epoches':epoches,
              'H':H, 'num_data_server':num_data_server,}
np.save(path_setting+'Exper2_setting2.npy', dictionary2)

"""
Exper 2:
(3) 10 users,   R = 0.4, Communication period = 8;
Server data number N_s = 1000, Number of participating clients C_k = 10; ResNet18 with group normalization
is used for training.
"""
size = 10 + 1
batch_size = 64
basicLabelRatio = 0.4
iid = 0
num_comm_ue = 10
k_img = 65536
epoches = 40
H = 0
cp = [8]
model = ['res_gn']
num_data_server = 1000

dictionary3 = {'size':size, 'batch_size':batch_size, 'cp':cp,
              'basicLabelRatio':basicLabelRatio, 'model':model, 'iid':iid,
              'num_comm_ue':num_comm_ue, 'k_img':k_img, 'epoches':epoches,
              'H':H, 'num_data_server':num_data_server,}
np.save(path_setting+'Exper2_setting3.npy', dictionary3)

"""
Exper 2:
(4) 10 users,   R = 0.4, Communication period = 32;
Server data number N_s = 1000, Number of participating clients C_k = 10; ResNet18 with group normalization
is used for training.
"""
size = 10 + 1
batch_size = 64
basicLabelRatio = 0.4
iid = 0
num_comm_ue = 10
k_img = 65536
epoches = 40
H = 0
cp = [32]
model = ['res_gn']
num_data_server = 1000

dictionary4 = {'size':size, 'batch_size':batch_size, 'cp':cp,
              'basicLabelRatio':basicLabelRatio, 'model':model, 'iid':iid,
              'num_comm_ue':num_comm_ue, 'k_img':k_img, 'epoches':epoches,
              'H':H, 'num_data_server':num_data_server,}
np.save(path_setting+'Exper2_setting4.npy', dictionary4)

"""
Exper 3:
(1) 10 users,   R = 0.4, Communication period = 16;
Server data number N_s = 2000, Number of participating clients C_k = 10; ResNet18 with group normalization
is used for training.
"""
size = 10 + 1
batch_size = 64
basicLabelRatio = 0.4
iid = 0
num_comm_ue = 10
k_img = 65536
epoches = 40
H = 0
cp = [16]
model = ['res_gn']
num_data_server = 2000

dictionary1 = {'size':size, 'batch_size':batch_size, 'cp':cp,
              'basicLabelRatio':basicLabelRatio, 'model':model, 'iid':iid,
              'num_comm_ue':num_comm_ue, 'k_img':k_img, 'epoches':epoches,
              'H':H, 'num_data_server':num_data_server,}
np.save(path_setting+'Exper3_setting1.npy', dictionary1)

"""
Exper 3:
(2) 10 users,   R = 0.4, Communication period = 16;
Server data number N_s = 3000, Number of participating clients C_k = 10; ResNet18 with group normalization
is used for training.
"""
size = 10 + 1
batch_size = 64
basicLabelRatio = 0.4
iid = 0
num_comm_ue = 10
k_img = 65536
epoches = 40
H = 0
cp = [16]
model = ['res_gn']
num_data_server = 3000

dictionary2 = {'size':size, 'batch_size':batch_size, 'cp':cp,
              'basicLabelRatio':basicLabelRatio, 'model':model, 'iid':iid,
              'num_comm_ue':num_comm_ue, 'k_img':k_img, 'epoches':epoches,
              'H':H, 'num_data_server':num_data_server,}
np.save(path_setting+'Exper3_setting2.npy', dictionary2)

"""
Exper 3:
(3) 10 users,   R = 0.4, Communication period = 16;
Server data number N_s = 4000, Number of participating clients C_k = 10; ResNet18 with group normalization
is used for training.
"""
size = 10 + 1
batch_size = 64
basicLabelRatio = 0.4
iid = 0
num_comm_ue = 10
k_img = 65536
epoches = 40
H = 0
cp = [16]
model = ['res_gn']
num_data_server = 4000

dictionary3 = {'size':size, 'batch_size':batch_size, 'cp':cp,
              'basicLabelRatio':basicLabelRatio, 'model':model, 'iid':iid,
              'num_comm_ue':num_comm_ue, 'k_img':k_img, 'epoches':epoches,
              'H':H, 'num_data_server':num_data_server,}
np.save(path_setting+'Exper3_setting3.npy', dictionary3)

"""
Exper 4:
(1) 20 users,   R = 0.4, Communication period = 16;
Server data number N_s = 1000, Number of participating clients C_k = 10; ResNet18 with group normalization
is used for training.
"""
size = 20 + 1
batch_size = 64
basicLabelRatio = 0.4
iid = 0
num_comm_ue = 10
k_img = 65536
epoches = 40
H = 0
cp = [16]
model = ['res_gn']
num_data_server = 1000

dictionary1 = {'size':size, 'batch_size':batch_size, 'cp':cp,
              'basicLabelRatio':basicLabelRatio, 'model':model, 'iid':iid,
              'num_comm_ue':num_comm_ue, 'k_img':k_img, 'epoches':epoches,
              'H':H, 'num_data_server':num_data_server,}
np.save(path_setting+'Exper4_setting1.npy', dictionary1)

"""
Exper 4:
(2) 20 users,   R = 0.4, Communication period = 16;
Server data number N_s = 1000, Number of participating clients C_k = 20; ResNet18 with group normalization
is used for training.
"""
size = 20 + 1
batch_size = 64
basicLabelRatio = 0.4
iid = 0
num_comm_ue = 20
k_img = 65536
epoches = 40
H = 0
cp = [16]
model = ['res_gn']
num_data_server = 1000

dictionary2 = {'size':size, 'batch_size':batch_size, 'cp':cp,
              'basicLabelRatio':basicLabelRatio, 'model':model, 'iid':iid,
              'num_comm_ue':num_comm_ue, 'k_img':k_img, 'epoches':epoches,
              'H':H, 'num_data_server':num_data_server,}
np.save(path_setting+'Exper4_setting2.npy', dictionary2)

"""
Exper 4:
(4) 30 users,   R = 0.4, Communication period = 16;
Server data number N_s = 1000, Number of participating clients C_k = 10; ResNet18 with group normalization
is used for training.
"""
size = 30 + 1
batch_size = 64
basicLabelRatio = 0.4
iid = 0
num_comm_ue = 10
k_img = 65536
epoches = 40
H = 0
cp = [16]
model = ['res_gn']
num_data_server = 1000

dictionary3 = {'size':size, 'batch_size':batch_size, 'cp':cp,
              'basicLabelRatio':basicLabelRatio, 'model':model, 'iid':iid,
              'num_comm_ue':num_comm_ue, 'k_img':k_img, 'epoches':epoches,
              'H':H, 'num_data_server':num_data_server,}
np.save(path_setting+'Exper4_setting3.npy', dictionary3)

"""
Exper 4:
(4) 30 users,   R = 0.4, Communication period = 16;
Server data number N_s = 1000, Number of participating clients C_k = 30; ResNet18 with group normalization
is used for training.
"""
size = 30 + 1
batch_size = 64
basicLabelRatio = 0.4
iid = 0
num_comm_ue = 30
k_img = 65536
epoches = 40
H = 0
cp = [16]
model = ['res_gn']
num_data_server = 1000

dictionary4 = {'size':size, 'batch_size':batch_size, 'cp':cp,
              'basicLabelRatio':basicLabelRatio, 'model':model, 'iid':iid,
              'num_comm_ue':num_comm_ue, 'k_img':k_img, 'epoches':epoches,
              'H':H, 'num_data_server':num_data_server,}
np.save(path_setting+'Exper4_setting4.npy', dictionary4)


"""
Exper 5:
(1) 20 users,   R = 0.4, Communication period = 16;
Server data number N_s = 1000, Number of participating clients C_k = 20; ResNet18 with group normalization
is used for training, grouping-based model average H = 1.
"""
size = 20 + 1
batch_size = 64
basicLabelRatio = 0.4
iid = 0
num_comm_ue = 20
k_img = 65536
epoches = 40
H = 1
cp = [16]
model = ['res_gn']
num_data_server = 1000

dictionary1 = {'size':size, 'batch_size':batch_size, 'cp':cp,
              'basicLabelRatio':basicLabelRatio, 'model':model, 'iid':iid,
              'num_comm_ue':num_comm_ue, 'k_img':k_img, 'epoches':epoches,
              'H':H, 'num_data_server':num_data_server,}
np.save(path_setting+'Exper5_setting1.npy', dictionary1)

"""
Exper 6:
(1) 10 users,   R = 0.4, Communication period = 16;
Server data number N_s = 1000, Number of participating clients C_k = 10; ResNet18 with batch normalization
is used for training. Train 120 epochs.
"""
size = 10 + 1
batch_size = 64
basicLabelRatio = 0.4
iid = 0
num_comm_ue = 10
k_img = 65536
epoches = 120
H = 0
cp = [16]
model = ['res_gn']
num_data_server = 1000

dictionary1 = {'size':size, 'batch_size':batch_size, 'cp':cp,
              'basicLabelRatio':basicLabelRatio, 'model':model, 'iid':iid,
              'num_comm_ue':num_comm_ue, 'k_img':k_img, 'epoches':epoches,
              'H':H, 'num_data_server':num_data_server,}
np.save(path_setting+'Exper6_setting1.npy', dictionary1)

"""
Exper 6:
(2) 30 users,   R = 0.4, Communication period = 16;
Server data number N_s = 1000, Number of participating clients C_k = 10; ResNet18 with batch normalization
is used for training. Train 120 epochs.
"""
size = 30 + 1
batch_size = 64
basicLabelRatio = 0.4
iid = 0
num_comm_ue = 30
k_img = 65536
epoches = 120
H = 0
cp = [16]
model = ['res_gn']
num_data_server = 1000

dictionary2 = {'size':size, 'batch_size':batch_size, 'cp':cp,
              'basicLabelRatio':basicLabelRatio, 'model':model, 'iid':iid,
              'num_comm_ue':num_comm_ue, 'k_img':k_img, 'epoches':epoches,
              'H':H, 'num_data_server':num_data_server,}
np.save(path_setting+'Exper6_setting2.npy', dictionary1)
