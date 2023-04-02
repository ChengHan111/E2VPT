import os
import numpy as np

# scratch code 
# a dummy version of code for finding the IG postion of the embedding layer, and prompt layer(during prompt tuning)

file_path = '/home/ch7858/vptSelf/A_test_as_finetune_before_pruning/vtab-cifar(num_classes=100)/sup_vitb16_224/lr0.00025_wd0.01/run1/txt_save_folder'
model_type = 'end2end' # set as default (no need to change)
file_path_2 = 
model_type_2 = 'prompt'

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

assert data_L2Norm.shape == data_MaxValue.shape == data_DataTargets.shape == data_DataLogits.shape == data_L2Norm_2.shape == data_MaxValue_2.shape == data_DataTargets_2.shape == data_DataLogits_2.shape

data_EW_PR = [] # end2end wrong prompt right
data_ER_PW = [] # end2end right prompt wrong
data_EW_PW = [] # end2end wrong prompt wrong

for i in range(data_L2Norm.shape[0]):
    for j in range(data_L2Norm.shape[1]):
        # end2end get wrong answer while prompt get right answer
        if int(data_DataTargets[i][j]) != int(data_DataLogits[i][j]) and int(data_DataTargets_2[i][j]) == int(data_DataLogits_2[i][j]):
            # haven't been tested!!!
            data_EW_PR.append(f"EL2{data_L2Norm[i][j]}_EMV{data_MaxValue[i][j]}_PL2{data_L2Norm_2[i][j]}_PMV{data_MaxValue_2[i][j]}")
            
        # end2end get right answer while prompt get wrong answer
        elif int(data_DataTargets[i][j]) == int(data_DataLogits[i][j]) and int(data_DataTargets_2[i][j]) != int(data_DataLogits_2[i][j]):
            data_ER_PW.append(f"EL2{data_L2Norm[i][j]}_EMV{data_MaxValue[i][j]}_PL2{data_L2Norm_2[i][j]}_PMV{data_MaxValue_2[i][j]}")
        
        # both get wrong answer
        elif int(data_DataTargets[i][j]) != int(data_DataLogits[i][j]) and int(data_DataTargets_2[i][j]) != int(data_DataLogits_2[i][j]):
            data_EW_PW.append(f"EL2{data_L2Norm[i][j]}_EMV{data_MaxValue[i][j]}_PL2{data_L2Norm_2[i][j]}_PMV{data_MaxValue_2[i][j]}")
        else:
            pass

print('data_EW_PR:', len(data_EW_PR))
print('data_ER_PW:', len(data_ER_PW))
print('data_EW_PW:', len(data_EW_PW))