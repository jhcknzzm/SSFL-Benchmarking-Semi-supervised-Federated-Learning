# SSFL-Semi-supervised-Federated-Learning: Improving Semi-supervised Federated Learning by Reducing the Gradient Diversity of Models
Improving Semi-supervised Federated Learning by Reducing the Gradient Diversity of Models
## Introduction
This repository includes all necessary programs to implement Semi-supervised Federated Learning of [the following paper](https://arxiv.org/abs/2008.11364). The code runs on Python 3.7.6 with PyTorch 1.0.0 and torchvision 0.2.2. We appreciate it if you would please cite the following paper if you found the repository useful for your work:


```
@article{SSFL,
  title={Improving Semi-supervised Federated Learning by Reducing the Gradient Diversity of Models},
  author={Zhang, Zhengming and Yang, Yaoqing and Yao, Zhewei and Yan, Yujun and Gonzalez, Joseph E and Ramchandran, Kannan and Mahoney, Michael W},
  journal={arXiv preprint arXiv:2105.14636},
  year={2021}
}
```


## Usage
Please first clone the this library to your local system:

```
git clone https://github.com/jhcknzzm/SSFL-Benchmarking-Semi-supervised-Federated-Learning.git
```

After cloning, please use Anaconda to install all the dependencies:

```
conda env create -f environment.yml
```

To run the main scripte "train_parallel.py", one needs to determine number of available GPUs.
For example, the number of GPUs on your machine is 4, you need to specify GPU_list as 0123 in run_cifar10.sh, run_svhn.sh and run_emnist.sh.

Then, you can train a Semi-supervised Federated Learning experiment using the following command:

```
python train_parallel.py  [--GPU_list] [--datasetid] [--size] [--basicLabelRatio] [--labeled] [--num_comm_ue] [--H] [--cp] [--eval_grad] [--experiment_folder] [--experiment_name] [--tao] [--model] --[ue_loss] [--user_semi]
                          [--epoch] [--batch_size] [--fast] [--Ns]
optional arguments:
--GPU_list:          GPUs used for training, e.g., --GPU_list 0123456789
--datasetid:         the id of the datasets (default: 0), datasetid = 0/1/2 means the Cifar-10/SVHN/EMNIST dataset is used in the experiment.
--size:              size = K (users) + 1 (server);
--cp:                cp in {2, 4, 8, 16} is frequency of communication; cp = 2 means UEs and server communicates every 2 iterations;
--basicLabelRatio:   basicLabelRatio in {0.1, 0.2, 0.4, ..., 1.0},  is the degree of data dispersion for each UE,
                     basicLabelRatio = 0.0 means UE has the same amount of samples in each class; basicLabelRatio = 1.0 means all samples owned
                     by UE belong to the same class;
--model:             model in {'res', 'res_gn', 'EMNIST_model'}; model = 'res' means we use ResNet18 + BN; model = 'res_gn' means we use ResNet18 + GN, EMNIST_model are used to                      train SSFL models on EMNIST dataset;
--num_comm_ue:       num_comm_ue in {1, 2, ..., K}; communication user number per iteration;
--H:                 H in {0, 1}; use grouping-based method or not; H = 1 means we use grouping-based method; H = 0 means we use FedAvg method;
--Ns:                num_data_server in {1000, 4000}, number of labeled samples in server;
--labeled:           labeled in {0, 1}, labeled=1 means supervised FL, labeled=0 means semi-supervised FL;
--cp:                cp in {2,4,8,16,32,64} is communication period between the users and the server
--eval_grad:         eval_grad in  {0, 1}, eval_grad=1 means that we load the model stored during training to calculate the gradient;
--experiment_folder: storage directory of experiment checkpoints;
--experiment_name:   the name of current experiment;
--tao:               hyperparameters used to calculate CRL;
--model:             neural network model for training;
--ue_loss:           ue_loss=CRL means we use CRL as the loss for local training; ue_loss=SF means we use self-training method for local training;
--epoch:             training epoches of SSFL;
--batch_size:        batch size used for training;
--fast:              hyperparameters used for learning rate update;
--Ns:                the number of labeled data in the server.
```
For example, you can also run the following script to reproduce the results of SSFL on Cifar10 in the setting of K=C=10, Ns=1000, model=res_gn with the non-iidness 0.4 and the grouping-based method.
```
python train_parallel.py --GPU_list 01234 --H 1 --num_comm_ue 10 --size 11 --epoch 300 --eval_grad 0 --model res_gn --basicLabelRatio 0.4 --experiment_name Cifar10_res_gn_H1_comUE10_R0.4_SSFL
```

In the above experiment, the default model is ResNet18. One can use the following command to change the model:
```
python train_parallel.py --GPU_list 01234 --H 1 --num_comm_ue 10  --model res9 --size 11 --epoch 300 --eval_grad 0 --model res_gn --basicLabelRatio 0.4 --experiment_name Cifar10_res_gn_H1_comUE10_R0.4_SSFL
```
One can also perform the experiments on ResNet9 to compare with another paper on federated semi-supervised learning. See Table 3 in section 4.3 in our paper for more details.
```
nohup bash run_cifar10_res9.sh
```


The results will be saved in the folder results_v0, and the checkpoints will be save in "/checkpoints/Cifar10_res_gn_H1_comUE10_R0.4_SSFL"

When all the checkpoints are saved, you can run the following script to calculate gradient diversity to reproduce the results on gradient diversity of our paper:
```
nohup bash Grad_Diff.sh
```
When all gradient diversities are calculated, you can run the following script to plot the results of gradient diversity.
```
python Plot_grad_diversity.py
```

You can also run the following scripts to reproduce the results reported in [the paper](https://arxiv.org/abs/2008.11364).

```
nohup bash Run_Exper.sh
```

