nohup python train_parallel.py --GPU_list 01234 --H 0 --num_comm_ue 10 --size 11 --epoch 300 --eval_grad 0 --model res --experiment_name Cifar10_res_H0_comUE10_R0.4_SSFL
nohup python train_parallel.py --GPU_list 01234 --H 0 --num_comm_ue 10 --size 11 --epoch 300 --eval_grad 0 --model res --experiment_name Cifar10_res_H0_comUE10_R0.4_SFL
nohup python train_parallel.py --GPU_list 01234 --H 0 --num_comm_ue 10 --size 11 --epoch 300 --eval_grad 0 --model res --experiment_name Cifar10_res_H0_comUE10_R0.0_SFL --basicLabelRatio 0.0
nohup python train_parallel.py --GPU_list 01234 --H 0 --num_comm_ue 10 --size 11 --epoch 300 --eval_grad 0 --model res_gn --experiment_name Cifar10_res_gn_H0_comUE10_R0.4_SSFL
nohup python train_parallel.py --GPU_list 01234 --H 1 --num_comm_ue 10 --size 11 --epoch 300 --eval_grad 0 --model res_gn --experiment_name Cifar10_res_gn_H1_comUE10_R0.4_SSFL
nohup python train_parallel.py --GPU_list 01234 --H 0 --num_comm_ue 10 --size 11 --epoch 300 --eval_grad 0 --model res --experiment_name Cifar10_res_H0_comUE10_R0.0_SSFL --basicLabelRatio 0.0
nohup python train_parallel.py --GPU_list 01234 --H 0 --num_comm_ue 10 --size 11 --epoch 300 --eval_grad 0 --model res_gn --experiment_name Cifar10_res_gn_H0_comUE10_R0.0_SSFL --basicLabelRatio 0.0
nohup python train_parallel.py --GPU_list 01234 --H 1 --num_comm_ue 10 --size 11 --epoch 300 --eval_grad 0 --model res_gn --Ns 2000 --experiment_name Cifar10_res_gn_H0_comUE10_R0.0_SSFL_Ns_2000_eval_grad_0
nohup python train_parallel.py --GPU_list 01234 --H 1 --num_comm_ue 10 --size 11 --epoch 300 --eval_grad 0 --model res_gn --Ns 3000 --experiment_name Cifar10_res_gn_H0_comUE10_R0.0_SSFL_Ns_3000_eval_grad_0
nohup python train_parallel.py --GPU_list 01234 --H 1 --num_comm_ue 10 --size 11 --epoch 300 --eval_grad 0 --model res_gn --Ns 4000 --experiment_name Cifar10_res_gn_H0_comUE10_R0.0_SSFL_Ns_4000_eval_grad_0
nohup python train_parallel.py --GPU_list 01234 --H 1 --num_comm_ue 10 --size 11 --epoch 300 --eval_grad 0 --model res_gn --Ns 5000 --experiment_name Cifar10_res_gn_H0_comUE10_R0.0_SSFL_Ns_5000_eval_grad_0
nohup python train_parallel.py --GPU_list 01234 --H 1 --num_comm_ue 10 --size 11 --epoch 300 --eval_grad 0 --model res_gn --cp 2 --experiment_name Cifar10_res_gn_H0_comUE10_R0.0_SSFL_Ns_2000_eval_grad_0_cp2
nohup python train_parallel.py --GPU_list 01234 --H 1 --num_comm_ue 10 --size 11 --epoch 300 --eval_grad 0 --model res_gn --cp 4 --experiment_name Cifar10_res_gn_H0_comUE10_R0.0_SSFL_Ns_2000_eval_grad_0_cp4
nohup python train_parallel.py --GPU_list 01234 --H 1 --num_comm_ue 10 --size 11 --epoch 300 --eval_grad 0 --model res_gn --cp 8 --experiment_name Cifar10_res_gn_H0_comUE10_R0.0_SSFL_Ns_2000_eval_grad_0_cp8
nohup python train_parallel.py --GPU_list 01234 --H 1 --num_comm_ue 10 --size 11 --epoch 300 --eval_grad 0 --model res_gn --cp 32 --experiment_name Cifar10_res_gn_H0_comUE10_R0.0_SSFL_Ns_2000_eval_grad_0_cp32
nohup python train_parallel.py --GPU_list 01234 --H 1 --num_comm_ue 10 --size 11 --epoch 300 --eval_grad 0 --model res_gn --cp 64 --experiment_name Cifar10_res_gn_H0_comUE10_R0.0_SSFL_Ns_2000_eval_grad_0_cp64
nohup python train_parallel.py --GPU_list 01234 --H 1 --num_comm_ue 10 --size 11 --epoch 300 --eval_grad 0 --model res_gn --basicLabelRatio 0.0 --experiment_name Cifar10_res_gn_H1_comUE10_R0.0_SSFL
nohup python train_parallel.py --GPU_list 01234 --H 1 --num_comm_ue 10 --size 11 --epoch 300 --eval_grad 0 --model res_gn --basicLabelRatio 0.2 --experiment_name Cifar10_res_gn_H1_comUE10_R0.2_SSFL
nohup python train_parallel.py --GPU_list 01234 --H 1 --num_comm_ue 10 --size 11 --epoch 300 --eval_grad 0 --model res_gn --basicLabelRatio 0.6 --experiment_name Cifar10_res_gn_H1_comUE10_R0.6_SSFL
nohup python train_parallel.py --GPU_list 01234 --H 1 --num_comm_ue 10 --size 11 --epoch 300 --eval_grad 0 --model res_gn --basicLabelRatio 0.8 --experiment_name Cifar10_res_gn_H1_comUE10_R0.8_SSFL
nohup python train_parallel.py --GPU_list 01234 --H 1 --num_comm_ue 10 --size 11 --epoch 300 --eval_grad 0 --model res_gn --basicLabelRatio 1.0 --experiment_name Cifar10_res_gn_H1_comUE10_R1.0_SSFL
nohup python train_parallel.py --GPU_list 01234 --H 1 --num_comm_ue 10 --size 21 --epoch 300 --eval_grad 0 --model res_gn
nohup python train_parallel.py --GPU_list 01234 --H 1 --num_comm_ue 20 --size 21 --epoch 300 --eval_grad 0 --model res_gn
nohup python train_parallel.py --GPU_list 01234 --H 1 --num_comm_ue 10 --size 31 --epoch 300 --eval_grad 0 --model res_gn
nohup python train_parallel.py --GPU_list 01234 --H 1 --num_comm_ue 30 --size 31 --epoch 300 --eval_grad 0 --model res_gn
nohup python train_parallel.py --GPU_list 01234 --H 0 --num_comm_ue 10 --size 11 --epoch 300 --eval_grad 1 --model res --experiment_name Cifar10_res_H0_comUE10_R0.4_SSFL
nohup python train_parallel.py --GPU_list 01234 --H 0 --num_comm_ue 10 --size 11 --epoch 300 --eval_grad 1 --model res_gn --experiment_name Cifar10_res_gn_H0_comUE10_R0.4_SSFL
nohup python train_parallel.py --GPU_list 01234 --H 1 --num_comm_ue 10 --size 11 --epoch 300 --eval_grad 1 --model res_gn --experiment_name Cifar10_res_gn_H1_comUE10_R0.4_SSFL
nohup python train_parallel.py --GPU_list 01234 --H 0 --num_comm_ue 10 --size 11 --epoch 300 --eval_grad 1 --model res --experiment_name Cifar10_res_H0_comUE10_R0.0_SSFL --basicLabelRatio 0.0
nohup python train_parallel.py --GPU_list 01234 --H 0 --num_comm_ue 10 --size 11 --epoch 300 --eval_grad 1 --model res_gn --experiment_name Cifar10_res_gn_H0_comUE10_R0.0_SSFL --basicLabelRatio 0.0
nohup python train_parallel.py --GPU_list 01234 --H 1 --num_comm_ue 10 --size 11 --epoch 300 --eval_grad 1 --model res_gn --basicLabelRatio 0.0  --experiment_name Cifar10_res_gn_H1_comUE10_R0.0_SSFL
