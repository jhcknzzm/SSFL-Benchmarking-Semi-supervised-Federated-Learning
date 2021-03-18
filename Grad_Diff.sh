
declare -a experiment_names=("Cifar10_res_H0_comUE10_R0.4_SSFL" "Cifar10_res_gn_H0_comUE10_R0.4_SSFL" "Cifar10_res_gn_H1_comUE10_R0.4_SSFL"\
                             "Cifar10_res_H0_comUE10_R0.0_SSFL" "Cifar10_res_gn_H0_comUE10_R0.0_SSFL" "Cifar10_res_gn_H1_comUE10_R0.0_SSFL"\
                             "Cifar10_res_H0_comUE10_R0.4_SFL" "Cifar10_res_H0_comUE10_R0.0_SFL" \
                             "EMNIST_size47_comUE10_H1_R0.4_SSFL"  "EMNIST_size47_comUE30_H1_R0.4_SSFL" "EMNIST_size47_comUE47_H1_R0.4_SSFL"\
                             "EMNIST_size47_comUE10_H0_R0.4_SSFL"  "EMNIST_size47_comUE30_H0_R0.4_SSFL" "EMNIST_size47_comUE47_H0_R0.4_SSFL"\
                             )

declare -a tao=("0.95")
declare -a l2=("1" "0")
declare -a method_list=("diversity1")
declare -a vector_type_list=("weight_variation" "grad")
declare -a only_user_list=("0" "1")
declare -a ord_list=("2" "1")

for i in ${!experiment_names[@]};
do
  for j in ${!tao[@]};
  do
      for method in ${method_list[@]};
      do

          for k in ${!l2[@]};
          do

              for vector_type in ${vector_type_list[@]};
              do

                  for p in ${!only_user_list[@]};
                  do
                          for q in ${!ord_list[@]};
                          do

                                  python Grad_Diff.py --experiment_name ${experiment_names[$i]} --only_user ${only_user_list[$p]} --ord ${ord_list[$q]} --vector_type $vector_type --experiment_folder ${experiment_folders[$i]} --l2 ${l2[$k]} --tao ${tao[$j]} --method $method 1>./logs/"${experiment_names[$i]}"_"${tao[$j]}"_"${l2[$k]}"_"$method"_"$vector_type"_"only_user"_"${only_user_list[$p]}"_"Norm_ord"_"${order[$q]}".log 2>./logs/"${experiment_names[$i]}"_"${tao[$j]}"_"${l2[$k]}"_"$method"_"$vector_type"_"only_user"_"${only_user_list[$p]}"_"Norm_ord"_"${order[$q]}".err &

                                  /bin/sleep 5
                          done

                    done

             done

          done
      done
  done

done
