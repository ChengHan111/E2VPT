import os
import numpy as np

# scratch code 
# a dummy version of code for finding the IG postion of the embedding layer, and prompt layer(during prompt tuning)

file_path = '/home/ch7858/vptSelf/A_test_as_finetune_before_pruning/vtab-cifar(num_classes=100)/sup_vitb16_224/lr0.00025_wd0.01/run1/txt_save_folder'
model_type = 'end2end' # 'end2end'

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

assert data_L2Norm.shape == data_MaxValue.shape == data_DataTargets.shape == data_DataLogits.shape

data = []
for i in range(data_L2Norm.shape[0]):
    for j in range(data_L2Norm.shape[1]):
        if int(data_L2Norm[i][j]) > 197 or int(data_MaxValue[i][j]) > 197:
        # data.append(int(data_L2Norm[i][j]))
            data.append(f"{data_L2Norm[i][j]}_{data_MaxValue[i][j]}_{data_DataTargets[i][j]}_{data_DataLogits[i][j]}")
# print(len(data))
if len(data) == 0:
    print('no IG position is larger than 197')
else:
    print(data)