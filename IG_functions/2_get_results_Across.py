import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# scratch code 
# a dummy version of code for finding the IG postion of the embedding layer, and prompt layer(during prompt tuning)

file_path = '/home/ch7858/vptSelf/A_test_as_finetune_before_pruning/vtab-cifar(num_classes=100)/sup_vitb16_224/lr0.00025_wd0.001/run1/txt_save_folder'
model_type = 'end2end' # set as default (no need to change)
file_path_2 = '/home/ch7858/vptSelf/A_test_as_before_pruning/vtab-cifar(num_classes=100)/sup_vitb16_224/lr2.5_wd0.001/run1/txt_save_folder'
model_type_2 = 'prompt'

def plot_histogram(data, type_name, mark_name='default'):
    
    counter = Counter(data)
    x = list(counter.keys())
    y = list(counter.values())

    # 绘制柱状图
    plt.figure(dpi=300)
    # plt.bar(x, y)
    # plot the prompt value red
    if 'prompt' in mark_name:  # 添加标记为highlight的条件分支
        color_list = ['r' if value > 197 else 'b' for value in x]
        plt.bar(x, y, color=color_list)
    else:
        plt.bar(x, y)
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title(f'Histogram of {type_name}')
    # plt.show()
    plt.savefig(f'./histogram/histogram_{type_name}.png')

# convert each line of text into a numpy array
data = []
with open(file_path + f'/1_dataL2Norm_{model_type}.txt', 'r') as f:
    lines = f.readlines()
# 遍历每行内容，并转换为整数
for line in lines:
    line = line.strip()
    values = line.split(' ')
    # number = line.numpy()
    # print(line)
    # print(number)
    # convert string values to float and append to list
    data.append([float(x) for x in values])

# convert data list into a numpy array
data_L2Norm = np.array(data)

# convert each line of text into a numpy array
data = []
with open(file_path + f'/2_dataMaxValue_{model_type}.txt', 'r') as f:
    lines = f.readlines()
# 遍历每行内容，并转换为整数
for line in lines:
    line = line.strip()
    values = line.split(' ')
    # number = line.numpy()
    # print(line)
    # print(number)
    # convert string values to float and append to list
    data.append([float(x) for x in values])

# convert data list into a numpy array
data_MaxValue = np.array(data)

# convert each line of text into a numpy array
data = []
with open(file_path + f'/3_data_targets_{model_type}.txt', 'r') as f:
    lines = f.readlines()
# 遍历每行内容，并转换为整数
for line in lines:
    line = line.strip()
    values = line.split(' ')
    # number = line.numpy()
    # print(line)
    # print(number)
    # convert string values to float and append to list
    data.append([float(x) for x in values])

# convert data list into a numpy array
data_DataTargets = np.array(data)

# convert each line of text into a numpy array
data = []
with open(file_path + f'/4_data_outputs_{model_type}.txt', 'r') as f:
    lines = f.readlines()
# 遍历每行内容，并转换为整数
for line in lines:
    line = line.strip()
    values = line.split(' ')
    # number = line.numpy()
    # print(line)
    # print(number)
    # convert string values to float and append to list
    data.append([float(x) for x in values])

# convert data list into a numpy array
data_DataLogits = np.array(data)

### for prompt ###

# convert each line of text into a numpy array
data = []
with open(file_path_2 + f'/1_dataL2Norm_{model_type_2}.txt', 'r') as f:
    lines = f.readlines()
# 遍历每行内容，并转换为整数
for line in lines:
    line = line.strip()
    values = line.split(' ')
    # number = line.numpy()
    # print(line)
    # print(number)
    # convert string values to float and append to list
    data.append([float(x) for x in values])

# convert data list into a numpy array
data_L2Norm_2 = np.array(data)

# convert each line of text into a numpy array
data = []
with open(file_path_2 + f'/2_dataMaxValue_{model_type_2}.txt', 'r') as f:
    lines = f.readlines()
# 遍历每行内容，并转换为整数
for line in lines:
    line = line.strip()
    values = line.split(' ')
    # number = line.numpy()
    # print(line)
    # print(number)
    # convert string values to float and append to list
    data.append([float(x) for x in values])

# convert data list into a numpy array
data_MaxValue_2 = np.array(data)

# convert each line of text into a numpy array
data = []
with open(file_path_2 + f'/3_data_targets_{model_type_2}.txt', 'r') as f:
    lines = f.readlines()
# 遍历每行内容，并转换为整数
for line in lines:
    line = line.strip()
    values = line.split(' ')
    # number = line.numpy()
    # print(line)
    # print(number)
    # convert string values to float and append to list
    data.append([float(x) for x in values])

# convert data list into a numpy array
data_DataTargets_2 = np.array(data)

# convert each line of text into a numpy array
data = []
with open(file_path_2 + f'/4_data_outputs_{model_type_2}.txt', 'r') as f:
    lines = f.readlines()
# 遍历每行内容，并转换为整数
for line in lines:
    line = line.strip()
    values = line.split(' ')
    # number = line.numpy()
    # print(line)
    # print(number)
    # convert string values to float and append to list
    data.append([float(x) for x in values])

# convert data list into a numpy array
data_DataLogits_2 = np.array(data)

assert data_L2Norm.shape == data_MaxValue.shape == data_DataTargets.shape == data_DataLogits.shape 
assert data_L2Norm_2.shape == data_MaxValue_2.shape == data_DataTargets_2.shape == data_DataLogits_2.shape

print(data_L2Norm.shape)
print(data_L2Norm_2.shape)

data_L2Norm_line1 = []
data_MaxValue_line1 = []
data_DataTargets_line1 = []
data_DataLogits_line1 = []

data_L2Norm_2_line1 = []
data_MaxValue_2_line1 = []
data_DataTargets_2_line1 = []
data_DataLogits_2_line1 = []


for i in range(data_L2Norm.shape[0]):
    for j in range(data_L2Norm.shape[1]):
        data_L2Norm_line1.append(data_L2Norm[i][j])
        data_MaxValue_line1.append(data_MaxValue[i][j])
        data_DataTargets_line1.append(data_DataTargets[i][j])
        data_DataLogits_line1.append(data_DataLogits[i][j])

for i in range(data_L2Norm_2.shape[0]):
    for j in range(data_L2Norm_2.shape[1]):
        data_L2Norm_2_line1.append(data_L2Norm_2[i][j])
        data_MaxValue_2_line1.append(data_MaxValue_2[i][j])
        data_DataTargets_2_line1.append(data_DataTargets_2[i][j])
        data_DataLogits_2_line1.append(data_DataLogits_2[i][j])


data_EW_PR = [] # end2end wrong prompt right
data_ER_PW = [] # end2end right prompt wrong
data_EW_PW = [] # end2end wrong prompt wrong
data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12 = [], [], [], [], [], [], [], [], [], [], [], []

# print(data_DataTargets_line1==data_DataTargets_2_line1)
assert data_DataTargets_line1==data_DataTargets_2_line1

count_1 = 0
count_2 = 0

for i in range(len(data_DataTargets_line1)):
    # end2end get wrong answer while prompt get right answer
    if data_DataLogits_line1[i] != data_DataTargets_line1[i] and data_DataLogits_2_line1[i] == data_DataTargets_line1[i]:
        data_EW_PR.append(f"EL2{data_L2Norm_line1[i]}_EMV{data_MaxValue_line1[i]}_PL2{data_L2Norm_2_line1[i]}_PMV{data_MaxValue_2_line1[i]}")
        data1.append(data_L2Norm_line1[i])
        data2.append(data_MaxValue_line1[i])
        data3.append(data_L2Norm_2_line1[i])
        data4.append(data_MaxValue_2_line1[i])
        
    
    # end2end get right answer while prompt get wrong answer
    elif data_DataLogits_line1[i] == data_DataTargets_line1[i] and data_DataLogits_2_line1[i] != data_DataTargets_line1[i]:
        data_ER_PW.append(f"EL2{data_L2Norm_line1[i]}_EMV{data_MaxValue_line1[i]}_PL2{data_L2Norm_2_line1[i]}_PMV{data_MaxValue_2_line1[i]}")
        data5.append(data_L2Norm_line1[i])
        data6.append(data_MaxValue_line1[i])
        data7.append(data_L2Norm_2_line1[i])
        data8.append(data_MaxValue_2_line1[i])
        
    # both get wrong answer
    elif data_DataLogits_line1[i] != data_DataTargets_line1[i] and data_DataLogits_2_line1[i] != data_DataTargets_line1[i]:
        # print('1', data_DataLogits_line1[i], data_DataTargets_line1[i])
        # print('2', data_DataLogits_2_line1[i], data_DataTargets_line1[i])
        data_EW_PW.append(f"EL2{data_L2Norm_line1[i]}_EMV{data_MaxValue_line1[i]}_PL2{data_L2Norm_2_line1[i]}_PMV{data_MaxValue_2_line1[i]}")
        data9.append(data_L2Norm_line1[i])
        data10.append(data_MaxValue_line1[i])
        data11.append(data_L2Norm_2_line1[i])
        data12.append(data_MaxValue_2_line1[i])
        
    else:
        pass
    
for i in range(len(data_DataTargets_line1)):
    if data_DataLogits_line1[i] != data_DataTargets_line1[i]:
        count_1 += 1
    
    elif data_DataLogits_2_line1[i] != data_DataTargets_line1[i]:
        count_2 += 1
# for i in range(data_L2Norm.shape[0]):
#     for j in range(data_L2Norm.shape[1]):
#         # end2end get wrong answer while prompt get right answer
#         if int(data_DataTargets[i][j]) != int(data_DataLogits[i][j]) and int(data_DataTargets_2[i][j]) == int(data_DataLogits_2[i][j]):
#             # haven't been tested!!!
#             data_EW_PR.append(f"EL2{data_L2Norm[i][j]}_EMV{data_MaxValue[i][j]}_PL2{data_L2Norm_2[i][j]}_PMV{data_MaxValue_2[i][j]}")
            
#         # end2end get right answer while prompt get wrong answer
#         elif int(data_DataTargets[i][j]) == int(data_DataLogits[i][j]) and int(data_DataTargets_2[i][j]) != int(data_DataLogits_2[i][j]):
#             data_ER_PW.append(f"EL2{data_L2Norm[i][j]}_EMV{data_MaxValue[i][j]}_PL2{data_L2Norm_2[i][j]}_PMV{data_MaxValue_2[i][j]}")
        
#         # both get wrong answer
#         elif int(data_DataTargets[i][j]) != int(data_DataLogits[i][j]) and int(data_DataTargets_2[i][j]) != int(data_DataLogits_2[i][j]):
#             data_EW_PW.append(f"EL2{data_L2Norm[i][j]}_EMV{data_MaxValue[i][j]}_PL2{data_L2Norm_2[i][j]}_PMV{data_MaxValue_2[i][j]}")
#         else:
#             pass
# print(data_EW_PR)
# print(data_ER_PW)
# print(data_EW_PW)

plot_histogram(data1, '1_EW_PR-EL2', 'default')
plot_histogram(data2, '1_EW_PR-EMV', 'default')
plot_histogram(data3, '1_EW_PR-PL2', 'prompt')
plot_histogram(data4, '1_EW_PR-PMV', 'prompt')
plot_histogram(data5, '2_ER_PW-EL2', 'default')
plot_histogram(data6, '2_ER_PW-EMV', 'default')
plot_histogram(data7, '2_ER_PW-PL2', 'prompt')
plot_histogram(data8, '2_ER_PW-PMV', 'prompt')
plot_histogram(data9, '3_EW_PW-EL2', 'default')
plot_histogram(data10, '3_EW_PW-EMV', 'default')
plot_histogram(data11, '3_EW_PW-PL2', 'prompt')
plot_histogram(data12, '3_EW_PW-PMV', 'prompt')


print('count_1:', count_1)
print('count_2:', count_2)
print('data_EW_PR:', len(data_EW_PR))
print('data_ER_PW:', len(data_ER_PW))
print('data_EW_PW:', len(data_EW_PW))