import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.ticker as ticker

age_list = ['10','20','30','40','50','60','70','80','90','100']

stat_point = 0
end_point = 10
interval =  1
age_list = age_list[stat_point:end_point:interval]

color = ['#696969','coral','steelblue','maroon','deeppink','limegreen','firebrick','khaki','yellowgreen','navy']

hatch_list = ["/","X", "\\", "." , "+", "*", "o", "O", "x", "-"]

name_list=age_list

dir_root_list = ['./diversity_ord_1/', './diversity_ord_2/']'

methods_list = ['diversity']

#### If you want to plot the results of EMNIST:
Name_list = ["weight_variation_FedAvg_post/EMNIST_size47_comUE10_H0_R0.4_SSFL",
             "weight_variation_Group_post/EMNIST_size47_comUE10_H1_R0.4_SSFL",
            "weight_variation_FedAvg_post/EMNIST_size47_comUE30_H0_R0.4_SSFL",
            "weight_variation_Group_post/EMNIST_size47_comUE30_H1_R0.4_SSFL",
            "weight_variation_FedAvg_post/EMNIST_size47_comUE47_H0_R0.4_SSFL",
            "weight_variation_Group_post/EMNIST_size47_comUE47_H1_R0.4_SSFL"]
#### If you want to plot the results of Cifar10
# Name_list = ["weight_variation_FedAvg_post/Cifar10_res_H0_comUE10_R0.4_SSFL",
#              "weight_variation_Group_post/Cifar10_res_gn_H1_comUE10_R0.4_SSFL",
#             "weight_variation_FedAvg_post/Cifar10_res_gn_H0_comUE10_R0.4_SSFL",
#             "weight_variation_FedAvg_post/Cifar10_res_H0_comUE10_R0.0_SSFL",
#             "weight_variation_Group_post/Cifar10_res_gn_H1_comUE10_R0.0_SSFL",
#             "weight_variation_FedAvg_post/Cifar10_res_gn_H0_comUE10_R0.0_SSFL",]

l2_list = ['1','0']
all_user_list = ['1','0']
type_list = ['grad', 'weight_variation']

for type in type_list:
    for dir_root in dir_root_list:
        for l2_value in l2_list:
            Var_list = []
            for only_user in all_user_list:
                for name in Name_list:
                    dir = dir_root + f'{name}/'
                    WD0 = np.load(dir+f'{type}_diversity_list_l2_{l2_value}_only_user_{only_user}.npy')

                    WD0 = np.array(WD0)

                    Var = np.zeros((10,))
                    Var[0:len(WD0[0:])] = WD0[0:]
                    Var_list.append(Var[stat_point:end_point:interval])

                x = list(range(len(name_list)))
                width=0.3/1.5
                index=np.arange(len(name_list))+1


                plt.bar(index,Var_list[0],width,color='k',tick_label = name_list, hatch=hatch_list[0],alpha=0.6)
                plt.bar(index+width,Var_list[1],width,color='#d95f0e',hatch=hatch_list[1])

                Legend_name = ['FedAvg','Grouping-based']

                font_size = 29
                plt.yscale('log')
                plt.yticks(fontproperties = 'Times New Roman', size = font_size-10)
                plt.xticks(fontproperties = 'Times New Roman', size = font_size-10)
                plt.ylabel('Gradient diversity', fontdict={'family' : 'Times New Roman', 'size'   : font_size})
                plt.xlabel('epoch', fontdict={'family' : 'Times New Roman', 'size'   : font_size})
                if l2_value == '1':
                    plt.ylim([1.0,49.0])
                else:
                    if only_user == '1':
                        plt.ylim([1.0000,7.20025])
                    else:
                        plt.ylim([1.00001,7.20025])
                plt.grid(True, linestyle = "-.", linewidth = "0.15")

                if l2_value == '1':
                    plt.legend(Legend_name,labelspacing=0.2, loc=4,fontsize=18.2, ncol=1)
                else:
                    plt.legend(Legend_name,labelspacing=0.2, loc=4,fontsize=18.2, ncol=1)
                plt.tight_layout()
                if 'ord_1' in dir_root:
                    Norm_ord_L = 1
                else:
                    Norm_ord_L = 2
                if only_user == '0':
                    plt.savefig(f'{type}_diversity_EMNIST_C10_L2_{l2_value}_all_ord{Norm_ord_L}.pdf')
                else:
                    plt.savefig(f'{type}_diversity_EMNIST_C10_L2_{l2_value}_only_user_ord{Norm_ord_L}.pdf')
                plt.show()

                #### C=30
                plt.bar(index,Var_list[2],width,color='peru',tick_label = name_list, hatch=hatch_list[0],alpha=0.6)
                plt.bar(index+width,Var_list[3],width,color='#2c7fb8',hatch=hatch_list[1])

                Legend_name = ['FedAvg','Grouping-based']

                font_size = 29
                plt.yscale('log')
                plt.yticks(fontproperties = 'Times New Roman', size = font_size-10)
                plt.xticks(fontproperties = 'Times New Roman', size = font_size-10)
                plt.ylabel('Gradient diversity', fontdict={'family' : 'Times New Roman', 'size'   : font_size})
                plt.xlabel('epoch', fontdict={'family' : 'Times New Roman', 'size'   : font_size})
                if l2_value == '1':
                    plt.ylim([1.0,49.0])
                else:
                    if only_user == '1':
                        plt.ylim([1.0000,7.20025])
                    else:
                        plt.ylim([1.00001,7.20025])
                plt.grid(True, linestyle = "-.", linewidth = "0.15")
                if l2_value == '1':
                    plt.legend(Legend_name,labelspacing=0.2, loc=4,fontsize=18.2, ncol=1)
                else:
                    plt.legend(Legend_name,labelspacing=0.2, loc=4,fontsize=18.2, ncol=1)
                plt.tight_layout()
                if 'ord_1' in dir_root:
                    Norm_ord_L = 1
                else:
                    Norm_ord_L = 2
                if only_user == '0':
                    plt.savefig(f'{type}_diversity_EMNIST_C30_L2_{l2_value}_all_ord{Norm_ord_L}.pdf')
                else:
                    plt.savefig(f'{type}_diversity_EMNIST_C30_L2_{l2_value}_only_user_ord{Norm_ord_L}.pdf')
                plt.show()

                ###### C=47
                plt.bar(index,Var_list[4],width,color='#756bb1',tick_label = name_list, hatch=hatch_list[0],alpha=0.6)
                plt.bar(index+width,Var_list[5],width,color='#c51b8a',hatch=hatch_list[1])

                Legend_name = ['FedAvg','Grouping-based']

                font_size = 29
                plt.yscale('log')
                plt.yticks(fontproperties = 'Times New Roman', size = font_size-10)
                plt.xticks(fontproperties = 'Times New Roman', size = font_size-10)
                plt.ylabel('Gradient diversity', fontdict={'family' : 'Times New Roman', 'size'   : font_size})
                plt.xlabel('epoch', fontdict={'family' : 'Times New Roman', 'size'   : font_size})
                if l2_value == '1':
                    plt.ylim([1.0,49.0])
                else:
                    if only_user == '1':
                        plt.ylim([1.0000,7.20025])
                    else:
                        plt.ylim([1.00001,7.20025])
                plt.grid(True, linestyle = "-.", linewidth = "0.15")
                if l2_value == '1':
                    plt.legend(Legend_name,labelspacing=0.2, loc=4,fontsize=18.2, ncol=1)
                else:
                    plt.legend(Legend_name,labelspacing=0.2, loc=4,fontsize=18.2, ncol=1)
                plt.tight_layout()
                if 'ord_1' in dir_root:
                    Norm_ord_L = 1
                else:
                    Norm_ord_L = 2
                if only_user == '0':
                    plt.savefig(f'{type}_diversity_EMNIST_C47_L2_{l2_value}_all_ord{Norm_ord_L}.pdf')
                else:
                    plt.savefig(f'{type}_diversity_EMNIST_C47_L2_{l2_value}_only_user_ord{Norm_ord_L}.pdf')

                plt.show()
