nohup python train_parallel.py --GPU_list 01234567 --H 1 --num_comm_ue 10 --size 48 --epoch 100 --eval_grad 0 --model EMNIST_model --Ns 1000 --datasetid 2 --fast 0
nohup python train_parallel.py --GPU_list 01234567 --H 1 --num_comm_ue 10 --size 48 --epoch 100 --eval_grad 0 --model EMNIST_model --Ns 2000 --datasetid 2 --fast 0
nohup python train_parallel.py --GPU_list 01234567 --H 1 --num_comm_ue 10 --size 48 --epoch 100 --eval_grad 0 --model EMNIST_model --Ns 3000 --datasetid 2 --fast 0
nohup python train_parallel.py --GPU_list 01234567 --H 1 --num_comm_ue 10 --size 48 --epoch 100 --eval_grad 0 --model EMNIST_model --Ns 4000 --datasetid 2 --fast 0
nohup python train_parallel.py --GPU_list 01234567 --H 1 --num_comm_ue 10 --size 48 --epoch 100 --eval_grad 0 --model EMNIST_model --Ns 5000 --datasetid 2 --fast 0
nohup python train_parallel.py --GPU_list 01234567 --H 1 --num_comm_ue 10 --size 48 --epoch 100 --eval_grad 0 --model EMNIST_model --cp 2 --datasetid 2 --fast 0 --Ns 4700
nohup python train_parallel.py --GPU_list 01234567 --H 1 --num_comm_ue 10 --size 48 --epoch 100 --eval_grad 0 --model EMNIST_model --cp 4 --datasetid 2 --fast 0 --Ns 4700
nohup python train_parallel.py --GPU_list 01234567 --H 1 --num_comm_ue 10 --size 48 --epoch 100 --eval_grad 0 --model EMNIST_model --cp 8 --datasetid 2 --fast 0 --Ns 4700
nohup python train_parallel.py --GPU_list 01234567 --H 1 --num_comm_ue 10 --size 48 --epoch 100 --eval_grad 0 --model EMNIST_model --cp 32 --datasetid 2 --fast 0 --Ns 4700
nohup python train_parallel.py --GPU_list 01234567 --H 1 --num_comm_ue 10 --size 48 --epoch 100 --eval_grad 0 --model EMNIST_model --cp 64 --datasetid 2 --fast 0 --Ns 4700
nohup python train_parallel.py --GPU_list 01234567 --H 1 --num_comm_ue 10 --size 48 --epoch 100 --eval_grad 0 --model EMNIST_model --basicLabelRatio 0.0 --datasetid 2 --fast 0 --Ns 4700
nohup python train_parallel.py --GPU_list 01234567 --H 1 --num_comm_ue 10 --size 48 --epoch 100 --eval_grad 0 --model EMNIST_model --basicLabelRatio 0.2 --datasetid 2 --fast 0 --Ns 4700
nohup python train_parallel.py --GPU_list 01234567 --H 1 --num_comm_ue 10 --size 48 --epoch 100 --eval_grad 0 --model EMNIST_model --basicLabelRatio 0.6 --datasetid 2 --fast 0 --Ns 4700
nohup python train_parallel.py --GPU_list 01234567 --H 1 --num_comm_ue 10 --size 48 --epoch 100 --eval_grad 0 --model EMNIST_model --basicLabelRatio 0.8 --datasetid 2 --fast 0 --Ns 4700
nohup python train_parallel.py --GPU_list 01234567 --H 1 --num_comm_ue 10 --size 48 --epoch 100 --eval_grad 0 --model EMNIST_model --basicLabelRatio 1.0 --datasetid 2 --fast 0 --Ns 4700
nohup python train_parallel.py --GPU_list 01234567 --H 1 --num_comm_ue 10 --size 48 --epoch 100 --eval_grad 0 --model EMNIST_model --datasetid 2 --Ns 4700 --fast 0 --experiment_name EMNIST_size47_comUE10_H1_R0.4_SSFL
nohup python train_parallel.py --GPU_list 01234567 --H 1 --num_comm_ue 30 --size 48 --epoch 100 --eval_grad 0 --model EMNIST_model --datasetid 2 --Ns 4700 --fast 0 --experiment_name EMNIST_size47_comUE20_H1_R0.4_SSFL
nohup python train_parallel.py --GPU_list 01234567 --H 1 --num_comm_ue 47 --size 48 --epoch 100 --eval_grad 0 --model EMNIST_model --datasetid 2 --Ns 4700 --fast 0 --experiment_name EMNIST_size47_comUE47_H1_R0.4_SSFL
nohup python train_parallel.py --GPU_list 01234567 --H 0 --num_comm_ue 10 --size 48 --epoch 100 --eval_grad 0 --model EMNIST_model --datasetid 2 --Ns 4700 --fast 0 --experiment_name EMNIST_size47_comUE10_H0_R0.4_SSFL
nohup python train_parallel.py --GPU_list 01234567 --H 0 --num_comm_ue 30 --size 48 --epoch 100 --eval_grad 0 --model EMNIST_model --datasetid 2 --Ns 4700 --fast 0 --experiment_name EMNIST_size47_comUE30_H0_R0.4_SSFL
nohup python train_parallel.py --GPU_list 01234567 --H 0 --num_comm_ue 47 --size 48 --epoch 100 --eval_grad 0 --model EMNIST_model --datasetid 2 --Ns 4700 --fast 0 --experiment_name EMNIST_size47_comUE47_H0_R0.4_SSFL
nohup python train_parallel.py --GPU_list 01234567 --H 1 --num_comm_ue 10 --size 48 --epoch 100 --eval_grad 1 --model EMNIST_model --datasetid 2 --Ns 4700 --fast 0 --experiment_name EMNIST_size47_comUE10_H1_R0.4_SSFL
nohup python train_parallel.py --GPU_list 01234567 --H 1 --num_comm_ue 30 --size 48 --epoch 100 --eval_grad 1 --model EMNIST_model --datasetid 2 --Ns 4700 --fast 0 --experiment_name EMNIST_size47_comUE20_H1_R0.4_SSFL
nohup python train_parallel.py --GPU_list 01234567 --H 1 --num_comm_ue 47 --size 48 --epoch 100 --eval_grad 1 --model EMNIST_model --datasetid 2 --Ns 4700 --fast 0 --experiment_name EMNIST_size47_comUE47_H1_R0.4_SSFL
nohup python train_parallel.py --GPU_list 01234567 --H 0 --num_comm_ue 10 --size 48 --epoch 100 --eval_grad 1 --model EMNIST_model --datasetid 2 --Ns 4700 --fast 0 --experiment_name EMNIST_size47_comUE10_H0_R0.4_SSFL
nohup python train_parallel.py --GPU_list 01234567 --H 0 --num_comm_ue 30 --size 48 --epoch 100 --eval_grad 1 --model EMNIST_model --datasetid 2 --Ns 4700 --fast 0 --experiment_name EMNIST_size47_comUE30_H0_R0.4_SSFL
nohup python train_parallel.py --GPU_list 01234567 --H 0 --num_comm_ue 47 --size 48 --epoch 100 --eval_grad 1 --model EMNIST_model --datasetid 2 --Ns 4700 --fast 0 --experiment_name EMNIST_size47_comUE47_H0_R0.4_SSFL
