# SSFL-Benchmarking-Semi-supervised-Federated-Learning
Benchmarking Semi-supervised Federated Learning
## Introduction
This repository includes all necessary programs to implement the Semi-supervised Federated Learning of the following paper. We appreciate it if you would please cite the following paper if you found the repository useful for your work:
* Z. Zhang, Z. Yao, Y. Yang, Y. Yan, J. E. Gonzalez, and M. W. Mahoney, “Benchmarking semi-supervised federated learning,” in NeurIPS 2020, under review.
## Usage
Please first clone the this library to your local system:

`git clone https://github.com/jhcknzzm/SSFL-Benchmarking-Semi-supervised-Federated-Learning.git` 

After cloning, please use Anaconda to install all the dependencies:

`conda env create -f environment.yml` 

To run the main scripte "train_parallel.py", one should first use `ifconfig -a` to find the ip address of the machine.
Then, you can train a Semi-supervised Federated Learning experiment using the following command:

```
python train_parallel.py [--experiment_num]  [--GPU_list] [--ip_address] [--datasetid]`
optional arguments:
--experiment_num      experiment number, each experiment number corresponds to a specific experiment setting, you can modify these settings by modifying setting_cifar/svhn/emnist.py, e.g., --experiment_num 0   
--GPU_list            GPUs used for training, e.g., --GPU_list 0123456789   
--ip_address          the ip address of the machine, e.g., --ip_address 128.32.162.169
--datasetid           the id of the datasets, datasetid = 0/1/2 means the Cifar-10/SVHN/EMNIST dataset is used in the experiment. 
```
Following is the default experiment setting of Cifar-10:
Experiment num|  K |  Ck |  T |  Ns | R | Batch size | Epochs | Average method | Neural network model
--- | --- | --- | --- | --- | --- | --- | --- | --- | ---
0 |  10 | 10  | 16  | 1000  | 1.0 | 64 | 300 | FedAvg | ResNet-18 with group normalization
1 |  10 | 10  | 16  | 1000  | 0.0 | 64 | 300 | FedAvg | ResNet-18 with group normalization
2 |  10 | 10  | 16  | 1000  | 0.2 | 64 | 300 |  FedAvg | ResNet-18 with group normalization
... | ... | ...   | ...   | ...   | ...  | ...  | ...  |  ...  | ... 

* K:    User number
* Ck:   The number of communication users
* T:    Communication period
* Ns:   The number of labeled samples in server
* R:    The non-iidness
