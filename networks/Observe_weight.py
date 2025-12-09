


import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


weight_path = r"/media/root/3339482d-9d23-44ee-99a0-85e517217d15/CKY/Bladder_project/Remodeling_final/RESULT_20220704_NonCross/ResNet_classify_fold1/models/epoch1464 Val_auc0.908 Test_auc0.963 Extra_auc0.861.pkl"

model = torch.load(weight_path)

# print("model:{}".format(model))
# print("model:{}".format(model["fc"]))

fc_weight = model["fc.weight"]

"""data"""
fc_weight_value = fc_weight[0, :]
x = pd.DataFrame(np.array(fc_weight_value.cpu()))
print("max_x:{}".format(max(abs(fc_weight_value))))
print("min_x:{}".format(min(abs(fc_weight_value))))

"""Plotting Histogram"""
# plt.figure(figsize=(20, 8), dpi=80)
# plt.hist(x, bins=100, rwidth=0.9, density=True)
# plt.title("Histogram of weight values for 2048 features", fontsize=30)
# plt.xlabel("Weight values", fontsize=25)
# plt.ylabel("Frequency", fontsize=25)
# plt.xticks(fontsize=25)
# plt.yticks(fontsize=25)
# plt.savefig("/root/hist_of_weight.png")
# plt.show()

"""Plotting waterfall"""
# # y_sort = np.sort(np.array(fc_weight_value.cpu()))
# y = np.array(fc_weight_value.cpu())
# # print("y_sort:{}".format(y_sort))
#
#
# # colorlist = ["#FFA500" for i in range(len(y_sort))]
# colorlist = ["#FFA500" for i in range(len(y))]
# print("len_colorlist:{}".format(colorlist))
#
# fig = plt.figure(figsize=(20, 15), dpi=80)
# # fig = plt.figure(1)
# axs = fig.add_subplot(111)
# x_axis = [i for i in range(0, len(x))]
# # axs.bar(x=x_axis, height=y_sort, color=colorlist)
# axs.bar(x=x_axis, height=y, color=colorlist)
# axs.set_facecolor("white")
# plt.title("Weight values for 2048 features", fontsize=30)
# plt.xlabel("Order", fontsize=25)
# plt.ylabel("Weight values", fontsize=25)
# plt.xticks(fontsize=25)
# plt.yticks(fontsize=25)
# plt.savefig("/root/waterfall_of_weight.png")
# plt.show()



