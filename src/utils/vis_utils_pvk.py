import datetime
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json

from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import confusion_matrix
# plt.rcParams["axes.grid"] = False

import warnings
warnings.filterwarnings("ignore")
LOG_NAME = "logs.txt"


def remove_trailing(eval_dict):
    min_num = min([len(v) for k, v in eval_dict.items() if "top5" not in k])
    new_dict ={}
    for k, v in eval_dict.items():
        if "top5" not in k:
            new_dict[k] = v[:min_num]
    return new_dict


def get_meta(job_root, job_path, model_type, model_name, dataset_type='vtab'):
    # get lr, wd, feature-type, dataset
    # print(job_root, job_path, model_type)
    j_data = job_path.split("/run")[0].split(
        job_root + "/" + model_type)[-1].split("/")
    if dataset_type == 'vtab':
        job_name = job_root.split("/output_finalfinal/")[1] # output_finalfinal ft_pt_finalfinal
    elif dataset_type == 'fgvc':
        job_name = job_root.split("/output_fgvc_finalfinal/")[1]
    elif dataset_type == 'vtab_rewind':
        job_name = job_root.split("/output_rewind/")[1]
    elif dataset_type == 'fgvc_rewind':
        job_name = job_root.split("/output_fgvc_rewind/")[1]
    # elif job_name == 'vtab_finetune':
    else:
        job_name = job_root.split("/_finalfinal/")[1] #_finalfinal
    
    # print(job_name)
    job_name_split = job_name.split("_")
    if dataset_type != 'vtab_finetune' and dataset_type != 'vtab':
        P_value, VK_value, Shared, Init = job_name_split[1], job_name_split[2], job_name_split[4], job_name_split[6]
    else:
        P_value, VK_value, Shared, Init = None, None, None, None
    
    j_data = job_path.split("/run")[0].split(str(model_name) + "/")
    j_data_lrwd = j_data[1]
    # data_name, feat_type, opt_params = j_data[1], j_data[2], j_data[3]
    lr = float(j_data_lrwd.split("_")[0].split("lr")[-1])
    wd = float(j_data_lrwd.split("_")[1].split("wd")[-1])
    # return data_name, feat_type, lr, wd
    if dataset_type == 'vtab':
        data_name = job_root.split(f"_{P_value}")[0].split("/output_finalfinal/")[1] #output_finalfinal
        # data_name = job_root.split("/ft_pt_finalfinal/")[1]
    elif dataset_type == 'fgvc':
        data_name = job_root.split(f"_{P_value}")[0].split("/output_fgvc_finalfinal/")[1]
    elif dataset_type == 'vtab_rewind':
        data_name = job_root.split(f"_{P_value}")[0].split("/output_rewind/")[1]
    elif dataset_type == 'fgvc_rewind':
        data_name = job_root.split(f"_{P_value}")[0].split("/output_fgvc_rewind/")[1]
    else:
        data_name = job_root.split("/_finalfinal/")[1]  #_finalfinal
          
        
    return data_name, model_name, P_value, VK_value, Shared, lr, wd, Init


def update_eval(line, eval_dict, data_name):        
    if "top1" in line and "top" in line.split(": top1:")[-1]:
        metric = "top"     
    else:
        metric = "rocauc"
    top1 = float(line.split(": top1:")[-1].split(metric)[0])
    # print(top1)
    eval_type = line.split(" Classification results with ")[-1].split(": top1")[0] 
    eval_type = "".join(eval_type.split("_" + data_name))
    
    # self added to rm '"'
    eval_type = eval_type.replace('"', "")
    if 'test' in eval_type:
        eval_type = 'test'
    elif 'val' in eval_type:
        eval_type = 'val'
    eval_dict[eval_type + "_top1"].append(top1)


def get_nmi(job_path):
    with open(job_path) as f:
        lines = f.readlines()
    nmi_dict = defaultdict(list)
    num_jobs = 0
    log_temp = []
    for l in lines:  #, leave=False):
        if "Rank of current process:" in l:
            num_jobs += 1
        if num_jobs == 2:
            break
        if "Clutering nmi" in l:
            n = l.split("Clutering nmi: ")[-1].split(",")[0]
            a_n = l.split("adjusted nmi: ")[-1].split(",")[0]
            v = l.split("v: ")[-1].split(",")[0]
            nmi_dict["nmi"].append(float(n))
            nmi_dict["a_nmi"].append(float(a_n))
            nmi_dict["v_nmi"].append(float(v))
    return nmi_dict


def get_mean_accuracy(job_path, data_name):
    val_data = torch.load(
        job_path.replace("logs.txt", f"val_{data_name}_logits.pth"))
    test_data = torch.load(
        job_path.replace("logs.txt", f"val_{data_name}_logits.pth"))
    v_matrix = confusion_matrix(
        val_data['targets'],
        np.argmax(val_data['joint_logits'], 1)
    )
    t_matrix = confusion_matrix(
        test_data['targets'],
        np.argmax(test_data['joint_logits'], 1)
    )
    return np.mean(v_matrix.diagonal()/v_matrix.sum(axis=1) ) * 100, np.mean(t_matrix.diagonal()/t_matrix.sum(axis=1) ) * 100


def get_training_data(job_path, model_type, job_root, MODEL_NAME, dataset_type):
    # data_name, feat_type, lr, wd = get_meta(job_root, job_path, model_type, MODEL_NAME)
    data_name, feat_type, P_value, VK_value, Shared, lr, wd, Init = get_meta(job_root, job_path, model_type, MODEL_NAME, dataset_type)
    # print(data_name, feat_type, P_value, VK_value, Shared, lr, wd, Init)
    
    with open(job_path) as f:
        lines = f.readlines()

    # get training loss per epoch, 
    # cls results for both val and test
    train_loss = []
    test_loss = []
    val_loss = []
    eval_dict = defaultdict(list)
#     best_epoch = -1
    num_jobs = 0
    total_params = -1
    gradiented_params = -1
    batch_size = None
    for line in lines:  #, leave=False):
        if "{'BATCH_SIZE'" in line and batch_size is None:
            batch_size = int(line.split("'BATCH_SIZE': ")[-1].split(",")[0])
            
        if "Total Parameters: " in line:
            total_params = int(line.split("Total Parameters: ")[-1].split("\t")[0])
            gradiented_params = int(line.split("Gradient Parameters: ")[-1].split("\n")[0])
            
            # for rewind approach, consider subtraction on coresponding parameters.
            if dataset_type == 'vtab_rewind' or dataset_type == 'fgvc_rewind':

                cls_token_mask = int(job_path.split('_mt')[1].split('_mtr')[0])
                cls_token_pieces_mask = int(job_path.split('_mtr')[1].split('/run')[0])
                # print('!!!!',cls_token_mask, cls_token_pieces_mask)
                if dataset_type == 'vtab_rewind':
                    root_path = job_path.split('/output_rewind')[0]# print('cls_token_mask', cls_token_mask)
                    # print('cls_token_pieces_mask', cls_token_pieces_mask)
                    # default as onVK = False # if you wanna change, make changes here
                    # ONVK_0 BS64_LB0
                    mask_tokens_path = root_path + '/output_before_pruning/' + f'{data_name}_{P_value}_{VK_value}_SHARED_{Shared}_INIT_{Init}_ACC_0_BS64_LB1/{feat_type}/lr{lr}_wd{wd}/run1/mask_tokens/{cls_token_mask}_soft_tokens_to_mask.json'
                    mask_tokens_pieces_path = root_path + '/output_before_pruning/' + f'{data_name}_{P_value}_{VK_value}_SHARED_{Shared}_INIT_{Init}_ACC_0_BS64_LB1/{feat_type}/lr{lr}_wd{wd}/run1/mask_tokens_pieces/{cls_token_pieces_mask}_soft_tokens_pieces_to_mask.json'
                elif dataset_type == 'fgvc_rewind':
                    root_path = job_path.split('/output_fgvc_rewind')[0]
                    mask_tokens_path = root_path + '/output_fgvc_before_pruning/' + f'{data_name}_{P_value}_{VK_value}_SHARED_{Shared}_INIT_{Init}_ACC_0_BS128_LB1/{feat_type}/lr{lr}_wd{wd}/run1/mask_tokens/{cls_token_mask}_soft_tokens_to_mask.json'
                    mask_tokens_pieces_path = root_path + '/output_fgvc_before_pruning/' + f'{data_name}_{P_value}_{VK_value}_SHARED_{Shared}_INIT_{Init}_ACC_0_BS128_LB1/{feat_type}/lr{lr}_wd{wd}/run1/mask_tokens_pieces/{cls_token_pieces_mask}_soft_tokens_pieces_to_mask.json'
                
                soft_token_to_mask = load_soft_token_mask_file(mask_tokens_path) 
                prompt_soft_tokens_mask_cls_token, parameter_cls_token_mask = mask_soft_tokens(P_value, soft_token_to_mask)
            
                soft_tokens_pieces_to_mask = load_soft_tokens_pieces_mask_file(mask_tokens_pieces_path) 
                # TODO: make cfg available here.
                CLS_TOKEN_P_PIECES_NUM = 16 # same to the config file as 16 
                prompt_soft_tokens_pieces_mask_cls_token, parameter_cls_token_piece_mask = mask_soft_tokens_pieces(P_value, soft_tokens_pieces_to_mask, CLS_TOKEN_P_PIECES_NUM)
                
                
                if "swin" not in feat_type:
                    # 12, 768 for total_dimension and prompt_dim
                    # notice the difference here, the prompt embeddings should be repeat 12 times as the parameter num.
                    prompt_embeddings = nn.Parameter(torch.ones(int(P_value[1:]), 768)) # for a single layer
                    
                    # print(prompt_embeddings.shape) # torch.Size([12, 10, 768])
                    soft_token_chunks_num_cls_token = int(768/CLS_TOKEN_P_PIECES_NUM)
                    prompt_embeddings = prompt_embeddings * prompt_soft_tokens_pieces_mask_cls_token.repeat((1,soft_token_chunks_num_cls_token))
                    # print('masking_map_pieces', prompt_soft_tokens_pieces_mask_cls_token)
                    prompt_embeddings = prompt_embeddings * prompt_soft_tokens_mask_cls_token.view(-1, 1).repeat(1, prompt_embeddings.shape[1])
                    # print('masking_map_cls_token', prompt_soft_tokens_mask_cls_token)
                    
                    prompt_embeddings_parameters = 12 * prompt_embeddings.shape[0] * prompt_embeddings.shape[1]
                    
                    # print('prompt_embeddings_origin', prompt_embeddings_origin)
                    # print('masked_percentage:', (prompt_embeddings_parameters - int(torch.sum(12 * prompt_embeddings))) / prompt_embeddings_parameters)
                    
                    masked_percentage = (prompt_embeddings_parameters - int(torch.sum(12 * prompt_embeddings))) / prompt_embeddings_parameters
                    
                    prompt_embeddings_parameters_filtered = prompt_embeddings_parameters - int(torch.sum(12 * prompt_embeddings))
                    # print(total_params)
                    parameter_added = (parameter_cls_token_mask + parameter_cls_token_piece_mask)
                    # total_params += parameter_added #.data[0] should not add more (already included)
                else:
                    # 1, 128 for total_dimension and prompt dim
                    # only pass the first layer, the layers after that are not covered.
                    prompt_embeddings = nn.Parameter(torch.ones(int(P_value[1:]), 128)) # for a single layer
                    soft_token_chunks_num_cls_token = int(128/CLS_TOKEN_P_PIECES_NUM)
                    prompt_embeddings = prompt_embeddings * prompt_soft_tokens_pieces_mask_cls_token.repeat((1,soft_token_chunks_num_cls_token))
                    prompt_embeddings = prompt_embeddings * prompt_soft_tokens_mask_cls_token.view(-1, 1).repeat(1, prompt_embeddings.shape[1])
                    prompt_embeddings_parameters = prompt_embeddings.shape[0] * prompt_embeddings.shape[1]
                    masked_percentage = (prompt_embeddings_parameters - int(torch.sum(1 * prompt_embeddings))) / prompt_embeddings_parameters
                    prompt_embeddings_parameters_filtered = prompt_embeddings_parameters - int(torch.sum(1 * prompt_embeddings))


                gradiented_params -= prompt_embeddings_parameters_filtered
                
                
        if "Rank of current process:" in line:
            num_jobs += 1
        if num_jobs == 2:
            break
        if "average train loss:" in line:
            loss_train = float(line.split("average train loss: ")[-1])
            train_loss.append(loss_train)
        if "average loss:" in line and "Inference (val):" in line:
            loss_val = float(line.split("average loss: ")[-1])
            val_loss.append(loss_val)
        if "average loss:" in line and "Inference (test):" in line:
            loss_test = float(line.split("verage loss: ")[-1])
            test_loss.append(loss_test)
        if " Classification results with " in line:
            # print(line)
            update_eval(line, eval_dict, data_name)
    
    if dataset_type == 'vtab_rewind' or dataset_type == 'fgvc_rewind':
        masked_percentage_value = masked_percentage
        cls_token_mask_value = cls_token_mask
        cls_token_pieces_mask_value = cls_token_pieces_mask
    else:
        masked_percentage_value = None
        cls_token_mask_value = None
        cls_token_pieces_mask_value = None
    

    meta_dict = {
        "data": data_name,
        "feature": feat_type,
        "lr": float(lr) * 256 / int(batch_size),
        "wd": wd,
        "total_params": total_params,
        "tuned_params": gradiented_params,
        "tuned / total (%)": round(gradiented_params / total_params * 100, 4),
        "batch_size": batch_size,
        "Prompt_length": P_value,
        "VK_length": VK_value,
        "Shared": Shared,
        "Init": Init,
        "Masked_Percentage": masked_percentage_value,
        "Mask_Value": cls_token_mask_value,
        "Piece_Value": cls_token_pieces_mask_value,
    }
    v_top1, t_top1 = None, None
    return train_loss, val_loss, test_loss, eval_dict, meta_dict, (v_top1, t_top1)

def get_training_dataACC(job_path, model_type, job_root, MODEL_NAME, dataset_type):
    # data_name, feat_type, lr, wd = get_meta(job_root, job_path, model_type, MODEL_NAME)
    data_name, feat_type, P_value, VK_value, Shared, lr, wd, Init = get_meta(job_root, job_path, model_type, MODEL_NAME, dataset_type)
    # print(data_name, feat_type, P_value, VK_value, Shared, lr, wd, Init)
    
    with open(job_path) as f:
        lines = f.readlines()

    # get training loss per epoch, 
    # cls results for both val and test
    train_loss = []
    test_loss = []
    val_loss = []
    
    # get corresponding accuracy as well
    trainACC, valACC, testACC = [], [], []
    
    eval_dict = defaultdict(list)
#     best_epoch = -1
    num_jobs = 0
    total_params = -1
    gradiented_params = -1
    batch_size = None
    for line in lines:  #, leave=False):
        if "{'BATCH_SIZE'" in line and batch_size is None:
            batch_size = int(line.split("'BATCH_SIZE': ")[-1].split(",")[0])
            
        if "Total Parameters: " in line:
            total_params = int(line.split("Total Parameters: ")[-1].split("\t")[0])
            gradiented_params = int(line.split("Gradient Parameters: ")[-1].split("\n")[0])
            
            # for rewind approach, consider subtraction on coresponding parameters.
            if dataset_type == 'vtab_rewind' or dataset_type == 'fgvc_rewind':

                cls_token_mask = int(job_path.split('_mt')[1].split('_mtr')[0])
                cls_token_pieces_mask = int(job_path.split('_mtr')[1].split('/run')[0])
                # print('!!!!',cls_token_mask, cls_token_pieces_mask)
                if dataset_type == 'vtab_rewind':
                    root_path = job_path.split('/output_rewind')[0]# print('cls_token_mask', cls_token_mask)
                    # print('cls_token_pieces_mask', cls_token_pieces_mask)
                    # default as onVK = False # if you wanna change, make changes here
                    # ONVK_0 BS64_LB0
                    mask_tokens_path = root_path + '/output_before_pruning/' + f'{data_name}_{P_value}_{VK_value}_SHARED_{Shared}_INIT_{Init}_ACC_0_BS64_LB1/{feat_type}/lr{lr}_wd{wd}/run1/mask_tokens/{cls_token_mask}_soft_tokens_to_mask.json'
                    mask_tokens_pieces_path = root_path + '/output_before_pruning/' + f'{data_name}_{P_value}_{VK_value}_SHARED_{Shared}_INIT_{Init}_ACC_0_BS64_LB1/{feat_type}/lr{lr}_wd{wd}/run1/mask_tokens_pieces/{cls_token_pieces_mask}_soft_tokens_pieces_to_mask.json'
                elif dataset_type == 'fgvc_rewind':
                    root_path = job_path.split('/output_fgvc_rewind')[0]
                    mask_tokens_path = root_path + '/output_fgvc_before_pruning/' + f'{data_name}_{P_value}_{VK_value}_SHARED_{Shared}_INIT_{Init}_ACC_0_BS128_LB1/{feat_type}/lr{lr}_wd{wd}/run1/mask_tokens/{cls_token_mask}_soft_tokens_to_mask.json'
                    mask_tokens_pieces_path = root_path + '/output_fgvc_before_pruning/' + f'{data_name}_{P_value}_{VK_value}_SHARED_{Shared}_INIT_{Init}_ACC_0_BS128_LB1/{feat_type}/lr{lr}_wd{wd}/run1/mask_tokens_pieces/{cls_token_pieces_mask}_soft_tokens_pieces_to_mask.json'
                
                soft_token_to_mask = load_soft_token_mask_file(mask_tokens_path) 
                prompt_soft_tokens_mask_cls_token, parameter_cls_token_mask = mask_soft_tokens(P_value, soft_token_to_mask)
            
                soft_tokens_pieces_to_mask = load_soft_tokens_pieces_mask_file(mask_tokens_pieces_path) 
                # TODO: make cfg available here.
                CLS_TOKEN_P_PIECES_NUM = 16 # same to the config file as 16 
                prompt_soft_tokens_pieces_mask_cls_token, parameter_cls_token_piece_mask = mask_soft_tokens_pieces(P_value, soft_tokens_pieces_to_mask, CLS_TOKEN_P_PIECES_NUM)
                
                
                if "swin" not in feat_type:
                    # 12, 768 for total_dimension and prompt_dim
                    # notice the difference here, the prompt embeddings should be repeat 12 times as the parameter num.
                    prompt_embeddings = nn.Parameter(torch.ones(int(P_value[1:]), 768)) # for a single layer
                    
                    # print(prompt_embeddings.shape) # torch.Size([12, 10, 768])
                    soft_token_chunks_num_cls_token = int(768/CLS_TOKEN_P_PIECES_NUM)
                    prompt_embeddings = prompt_embeddings * prompt_soft_tokens_pieces_mask_cls_token.repeat((1,soft_token_chunks_num_cls_token))
                    # print('masking_map_pieces', prompt_soft_tokens_pieces_mask_cls_token)
                    prompt_embeddings = prompt_embeddings * prompt_soft_tokens_mask_cls_token.view(-1, 1).repeat(1, prompt_embeddings.shape[1])
                    # print('masking_map_cls_token', prompt_soft_tokens_mask_cls_token)
                    
                    prompt_embeddings_parameters = 12 * prompt_embeddings.shape[0] * prompt_embeddings.shape[1]
                    
                    # print('prompt_embeddings_origin', prompt_embeddings_origin)
                    # print('masked_percentage:', (prompt_embeddings_parameters - int(torch.sum(12 * prompt_embeddings))) / prompt_embeddings_parameters)
                    
                    masked_percentage = (prompt_embeddings_parameters - int(torch.sum(12 * prompt_embeddings))) / prompt_embeddings_parameters
                    
                    prompt_embeddings_parameters_filtered = prompt_embeddings_parameters - int(torch.sum(12 * prompt_embeddings))
                    # print(total_params)
                    parameter_added = (parameter_cls_token_mask + parameter_cls_token_piece_mask)
                    # total_params += parameter_added #.data[0] should not add more (already included)
                else:
                    # 1, 128 for total_dimension and prompt dim
                    # only pass the first layer, the layers after that are not covered.
                    prompt_embeddings = nn.Parameter(torch.ones(int(P_value[1:]), 128)) # for a single layer
                    soft_token_chunks_num_cls_token = int(128/CLS_TOKEN_P_PIECES_NUM)
                    prompt_embeddings = prompt_embeddings * prompt_soft_tokens_pieces_mask_cls_token.repeat((1,soft_token_chunks_num_cls_token))
                    prompt_embeddings = prompt_embeddings * prompt_soft_tokens_mask_cls_token.view(-1, 1).repeat(1, prompt_embeddings.shape[1])
                    prompt_embeddings_parameters = prompt_embeddings.shape[0] * prompt_embeddings.shape[1]
                    masked_percentage = (prompt_embeddings_parameters - int(torch.sum(1 * prompt_embeddings))) / prompt_embeddings_parameters
                    prompt_embeddings_parameters_filtered = prompt_embeddings_parameters - int(torch.sum(1 * prompt_embeddings))


                gradiented_params -= prompt_embeddings_parameters_filtered
                
                
        if "Rank of current process:" in line:
            num_jobs += 1
        if num_jobs == 2:
            break
        if "average train loss:" in line:
            loss_train = float(line.split("average train loss: ")[-1])
            train_loss.append(loss_train)
        if "average loss:" in line and "Inference (val):" in line:
            loss_val = float(line.split("average loss: ")[-1])
            val_loss.append(loss_val)
        if "average loss:" in line and "Inference (test):" in line:
            loss_test = float(line.split("verage loss: ")[-1])
            test_loss.append(loss_test)
        if "Classification results with val" in line:
            acc_val_top1 = float(line.split("top1: ")[-1].split("	top5:")[0])
            valACC.append(acc_val_top1)
        if "Classification results with test" in line:
            acc_test_top1 = float(line.split("top1: ")[-1].split("	top5:")[0])
            testACC.append(acc_test_top1)
        if " Classification results with " in line:
            # print(line)
            update_eval(line, eval_dict, data_name)
    # print('valACC', valACC)
    # print('testACC', testACC)
    # exit()
    
    if dataset_type == 'vtab_rewind' or dataset_type == 'fgvc_rewind':
        masked_percentage_value = masked_percentage
        cls_token_mask_value = cls_token_mask
        cls_token_pieces_mask_value = cls_token_pieces_mask
    else:
        masked_percentage_value = None
        cls_token_mask_value = None
        cls_token_pieces_mask_value = None
    

    meta_dict = {
        "data": data_name,
        "feature": feat_type,
        "lr": float(lr) * 256 / int(batch_size),
        "wd": wd,
        "total_params": total_params,
        "tuned_params": gradiented_params,
        "tuned / total (%)": round(gradiented_params / total_params * 100, 4),
        "batch_size": batch_size,
        "Prompt_length": P_value,
        "VK_length": VK_value,
        "Shared": Shared,
        "Init": Init,
        "Masked_Percentage": masked_percentage_value,
        "Mask_Value": cls_token_mask_value,
        "Piece_Value": cls_token_pieces_mask_value,
    }
    v_top1, t_top1 = None, None
    return train_loss, val_loss, test_loss, trainACC, valACC, testACC, eval_dict, meta_dict, (v_top1, t_top1)

def load_soft_token_mask_file(path):
    with open(path) as f:
        t = json.load(f)
    
    soft_token_to_mask = set()
    for mask_number, soft_token in t.items():
        for soft_token_i in soft_token:
            soft_token_to_mask.add(soft_token_i) 
    
    return soft_token_to_mask

def load_soft_tokens_pieces_mask_file(path):
    with open(path) as f:
        t = json.load(f)
    soft_tokens_pieces_to_mask = {}
    for soft_token_idx, soft_token_pieces in t.items():
        soft_tokens_pieces_to_mask[int(soft_token_idx)] = set(soft_token_pieces)
    return soft_tokens_pieces_to_mask

# torch.Size([12, cls_token_length, 768]) 对于整个来说 

def mask_soft_tokens(P_value, soft_tokens_to_mask):
    # TODO: 这两块应该也是有参数量的 这是否要考虑进去呢
    prompt_soft_tokens_mask_cls_token = nn.Parameter(torch.ones(int(P_value[1:])))
    for soft_token_idx in soft_tokens_to_mask:
        # print('soft_token_idx',soft_token_idx)
        prompt_soft_tokens_mask_cls_token.data[soft_token_idx] = 0   
    parameter_cls_token_mask = int(P_value[1:])
    return prompt_soft_tokens_mask_cls_token, parameter_cls_token_mask      
        
def mask_soft_tokens_pieces(P_value, soft_tokens_pieces_to_mask, CLS_TOKEN_P_PIECES_NUM=16):
    prompt_soft_tokens_pieces_mask_cls_token = nn.Parameter(torch.ones(int(P_value[1:]), int(CLS_TOKEN_P_PIECES_NUM)))
    for soft_token_id, soft_token_pieces in soft_tokens_pieces_to_mask.items():
        for soft_token_piece in soft_token_pieces:
            prompt_soft_tokens_pieces_mask_cls_token.data[soft_token_id][soft_token_piece] = 0
    parameter_cls_token_piece_mask = int(P_value[1:]) * int(CLS_TOKEN_P_PIECES_NUM)
    return prompt_soft_tokens_pieces_mask_cls_token, parameter_cls_token_piece_mask

def get_time(file):
    with open(file) as f:
        lines = f.readlines()
    start_time = lines[0].split("[")[1].split("]")[0]
    start_time = datetime.datetime.strptime(start_time, '%m/%d %H:%M:%S')

    end_time = lines[-1].split("[")[1].split("]")[0]
    end_time = datetime.datetime.strptime(end_time, '%m/%d %H:%M:%S')

    per_iter = None
    with open(file) as f:
        lines = f.readlines()

    per_batch = []
    per_batch_train = []
    for line in lines[::-1]:
#         print(line)"Test 6/6. loss: 6.097, "
        if ". loss:" in line and "Test" in line:
            per_iter = line.split(" s / batch")[0].split(",")[-1]
            per_batch.append(float(per_iter))
        if ". train loss:" in line:
            per_iter = line.split(" s / batch")[0].split(",")[-1]
            per_batch_train.append(float(per_iter))
            
    return datetime.timedelta(seconds=(end_time-start_time).total_seconds()), np.mean(per_batch), np.mean(per_batch_train)


def get_df(files, model_type, root, MODEL_NAME, is_best=True, is_last=True, max_epoch=300, dataset_type='vtab'):
    pd_dict = defaultdict(list)
    for job_path in tqdm(files, desc=model_type):
        train_loss, val_loss, test_loss, eval_results, meta_dict, (v_top1, t_top1) = get_training_data(job_path, model_type, root, MODEL_NAME, dataset_type)
        batch_size = meta_dict["batch_size"]
        
        # print('train_loss', train_loss)
        # print('val_loss', val_loss)
        # print('test_loss', test_loss)
        
        if len(eval_results) == 0:
            print(f"job {job_path} not ready in eval results")
            continue

        if len(eval_results["val_top1"]) == 0:
            print(f"job {job_path} not ready in eval results 'val_top1'")
            continue

        if "val_top1" not in eval_results or "test_top1" not in eval_results:
            print(f"inbalanced: {job_path}")
            continue
                
        for k, v in meta_dict.items():
            pd_dict[k].append(v)
        
        metric_b = "val_top1"
        best_epoch = np.argmax(eval_results[metric_b])
        # print('best_epoch', best_epoch)
        
        train_loss_atbest = train_loss[best_epoch]
        val_loss_atbest = val_loss[best_epoch]
        test_loss_atbest = test_loss[best_epoch]

        if is_best:
            for name, val in eval_results.items():
                if "top5" in name:
                    continue
                if len(val) == 0:
                    continue
                if not isinstance(val[0], list):
                    try:
                        pd_dict["b-" + name].append(val[best_epoch])
                    except:
                        pd_dict["b-" + name].append(-1)
                        # ongoing training process
                        print(name, best_epoch, val)
        # last epoch
        if is_last:
            if v_top1 is not None:
                pd_dict["l-val_top1"].append(v_top1)
                pd_dict["l-test_top1"].append(t_top1)
                val = eval_results["val_top1"]
            else:
                for name, val in eval_results.items():
                    if "top5" in name:
                        continue
                    if len(val) == 0:
                        continue
                    pd_dict["l-" + name].append(val[-1])

        pd_dict["best_epoch"].append(f"{best_epoch + 1} | {len(val)}")

        pd_dict["file"].append(job_path)
        total_time, _, _ = get_time(job_path)
        pd_dict["total_time"].append(total_time)
        pd_dict['train_loss_atbest'].append(train_loss_atbest)
        pd_dict['val_loss_atbest'].append(val_loss_atbest)
        pd_dict['test_loss_atbest'].append(test_loss_atbest)

    result_df = None
    if len(pd_dict) > 0:
        result_df = pd.DataFrame(pd_dict)
        result_df = result_df.sort_values(['data', "feature", "lr", "wd"])
    return result_df

def get_df_forPlot(files, model_type, root, MODEL_NAME, is_best=True, is_last=True, max_epoch=300, dataset_type='vtab', train_mode='PromptTuning'):
    pd_dict = defaultdict(list)
    for job_path in tqdm(files, desc=model_type):
        train_loss, val_loss, test_loss, eval_results, meta_dict, (v_top1, t_top1) = get_training_data(job_path, model_type, root, MODEL_NAME, dataset_type)
        batch_size = meta_dict["batch_size"]
        
        try:
            os.makedirs(f"output_folder/{meta_dict['data']}/{meta_dict['Prompt_length']}_{meta_dict['VK_length']}_{train_mode}", exist_ok=True)
            with open(f"output_folder/{meta_dict['data']}/{meta_dict['Prompt_length']}_{meta_dict['VK_length']}_{train_mode}/{model_type}.txt", 'x') as f:
                for i in range(len(train_loss)):
                    row = [str(train_loss[i]), str(val_loss[i]), str(test_loss[i])]
                    row_str = '\t'.join(row)  # separate values with tab
                    f.write(row_str + '\n')  # write row to file, followed by a newline character
        except:
            print("file already exists")
        
        if len(eval_results) == 0:
            print(f"job {job_path} not ready in eval results")
            continue

        if len(eval_results["val_top1"]) == 0:
            print(f"job {job_path} not ready in eval results 'val_top1'")
            continue

        if "val_top1" not in eval_results or "test_top1" not in eval_results:
            print(f"inbalanced: {job_path}")
            continue
                
        for k, v in meta_dict.items():
            pd_dict[k].append(v)
        
        metric_b = "val_top1"
        best_epoch = np.argmax(eval_results[metric_b])
        # print('best_epoch', best_epoch)
        
        train_loss_atbest = train_loss[best_epoch]
        val_loss_atbest = val_loss[best_epoch]
        test_loss_atbest = test_loss[best_epoch]

        if is_best:
            for name, val in eval_results.items():
                if "top5" in name:
                    continue
                if len(val) == 0:
                    continue
                if not isinstance(val[0], list):
                    try:
                        pd_dict["b-" + name].append(val[best_epoch])
                    except:
                        pd_dict["b-" + name].append(-1)
                        # ongoing training process
                        print(name, best_epoch, val)
        # last epoch
        if is_last:
            if v_top1 is not None:
                pd_dict["l-val_top1"].append(v_top1)
                pd_dict["l-test_top1"].append(t_top1)
                val = eval_results["val_top1"]
            else:
                for name, val in eval_results.items():
                    if "top5" in name:
                        continue
                    if len(val) == 0:
                        continue
                    pd_dict["l-" + name].append(val[-1])

        pd_dict["best_epoch"].append(f"{best_epoch + 1} | {len(val)}")

        pd_dict["file"].append(job_path)
        total_time, _, _ = get_time(job_path)
        pd_dict["total_time"].append(total_time)
        pd_dict['train_loss_atbest'].append(train_loss_atbest)
        pd_dict['val_loss_atbest'].append(val_loss_atbest)
        pd_dict['test_loss_atbest'].append(test_loss_atbest)

    result_df = None
    if len(pd_dict) > 0:
        result_df = pd.DataFrame(pd_dict)
        result_df = result_df.sort_values(['data', "feature", "lr", "wd"])
    return result_df

def get_df_forPlotACC(files, model_type, root, MODEL_NAME, is_best=True, is_last=True, max_epoch=300, dataset_type='vtab', train_mode='PromptTuning'):
    pd_dict = defaultdict(list)
    for job_path in tqdm(files, desc=model_type):
        # currenly trainACC not available
        train_loss, val_loss, test_loss, trainACC, valACC, testACC, eval_results, meta_dict, (v_top1, t_top1) = get_training_dataACC(job_path, model_type, root, MODEL_NAME, dataset_type)
        batch_size = meta_dict["batch_size"]
        
        # print('trainACC', trainACC)
        # print('valACC', valACC)
        # print('testACC', testACC)
        # saveIndice = files[0].split('run')[1].split('/')[0]
        try:
            os.makedirs(f"output_folderACC/{meta_dict['data']}/{meta_dict['Prompt_length']}_{meta_dict['VK_length']}_{train_mode}", exist_ok=True)
            with open(f"output_folderACC/{meta_dict['data']}/{meta_dict['Prompt_length']}_{meta_dict['VK_length']}_{train_mode}/{model_type}.txt", 'x') as f:
                # since trainACC is not available, use valACC instead for length
                for i in range(len(valACC)): 
                    # print(i)
                    # row = [str(trainACC[i]), str(valACC[i]), str(testACC[i])]
                    row = [str(valACC[i]), str(testACC[i])]
                    row_str = '\t'.join(row)  # separate values with tab
                    f.write(row_str + '\n')  # write row to file, followed by a newline character
        except:
            print("file already exists")
        
        if len(eval_results) == 0:
            print(f"job {job_path} not ready in eval results")
            continue

        if len(eval_results["val_top1"]) == 0:
            print(f"job {job_path} not ready in eval results 'val_top1'")
            continue

        if "val_top1" not in eval_results or "test_top1" not in eval_results:
            print(f"inbalanced: {job_path}")
            continue
                
        for k, v in meta_dict.items():
            pd_dict[k].append(v)
        
        metric_b = "val_top1"
        best_epoch = np.argmax(eval_results[metric_b])
        # print('best_epoch', best_epoch)
        
        train_loss_atbest = train_loss[best_epoch]
        val_loss_atbest = val_loss[best_epoch]
        test_loss_atbest = test_loss[best_epoch]

        if is_best:
            for name, val in eval_results.items():
                if "top5" in name:
                    continue
                if len(val) == 0:
                    continue
                if not isinstance(val[0], list):
                    try:
                        pd_dict["b-" + name].append(val[best_epoch])
                    except:
                        pd_dict["b-" + name].append(-1)
                        # ongoing training process
                        print(name, best_epoch, val)
        # last epoch
        if is_last:
            if v_top1 is not None:
                pd_dict["l-val_top1"].append(v_top1)
                pd_dict["l-test_top1"].append(t_top1)
                val = eval_results["val_top1"]
            else:
                for name, val in eval_results.items():
                    if "top5" in name:
                        continue
                    if len(val) == 0:
                        continue
                    pd_dict["l-" + name].append(val[-1])

        pd_dict["best_epoch"].append(f"{best_epoch + 1} | {len(val)}")

        pd_dict["file"].append(job_path)
        total_time, _, _ = get_time(job_path)
        pd_dict["total_time"].append(total_time)
        pd_dict['train_loss_atbest'].append(train_loss_atbest)
        pd_dict['val_loss_atbest'].append(val_loss_atbest)
        pd_dict['test_loss_atbest'].append(test_loss_atbest)

    result_df = None
    if len(pd_dict) > 0:
        result_df = pd.DataFrame(pd_dict)
        result_df = result_df.sort_values(['data', "feature", "lr", "wd"])
    return result_df

def delete_ckpts(f):
    # delete saved ckpts for re
    f_dir, _ = os.path.split(f)
    for f_delete in glob.glob(f_dir + "/*.pth"):
        os.remove(f_delete)
        print(f"removed {f_delete}")


def average_df(df, metric_names=["l-val_top1", "l-val_base_top1"], take_average=True):
    # for each data and features and train type, display the averaged results
    data_names = set(list(df["data"]))
    f_names = set(list(df["feature"]))
    t_names = set(list(df["type"]))
    hp_names = [
        c for c in df.columns if c not in ["data", "feature", "type", "file", "best_epoch"] + metric_names]
    data_dict = defaultdict(list)
    for d_name in data_names:
        for f_name in f_names:
            for t_name in t_names:

                result = df[df.data == d_name]
                result = result[result.feature == f_name]
                result = result[result.type == t_name]
                # take average here
                if len(result) == 0:
                    continue
                data_dict["data"].append(d_name)
                data_dict["feature"].append(f_name)
                data_dict["type"].append(t_name)
                data_dict["total_runs"].append(len(result))
        
                for m in metric_names:
                    if take_average:
                        data_dict[m].append("{:.2f}".format(
                            np.mean([r for i, r in enumerate(result[m])]),
                        ))
                        data_dict[f"{m}-std"].append("{:.2f}".format(
                            np.std([r for i, r in enumerate(result[m])])
                        ))
                    else:
                        data_dict[m].append("{:.2f}".format(
                            np.median([r for i, r in enumerate(result[m])]),
                        ))
                for h_name in hp_names:
                    data_dict[h_name].append(result[h_name].iloc[0])

    df = pd.DataFrame(data_dict)
    df = df.sort_values(["data", "feature", "type"])
    return df


def filter_df(df, sorted_cols, max_num):
    # for each data and features, display only top max_num runs
    data_names = set(list(df["data"]))
    f_names = set(list(df["feature"]))
    t_names = set(list(df["type"]))
    df_list = []
    for d_name in data_names:
        for f_name in f_names:
            for t_name in t_names:
                result = df[df.data == d_name]
                result = result[result.feature == f_name]
                result = result[result.type == t_name]
                if len(result) == 0:
                    continue
                cols = [c for c in sorted_cols if c in result.columns]
                result = result.sort_values(cols, ignore_index=True)

                _num = min([max_num, len(result)])
    #             print(result.iloc[-_num:])
                df_list.append(result.iloc[-_num:])
    return pd.concat(df_list)


def display_results(df, sorted_cols=["data", "feature", "type", "l-val_top1"], max_num=1):
    cols = [c for c in df.columns if c not in []]
    df = df[cols]
    if max_num is not None:
        df = filter_df(df, sorted_cols[3:], max_num)
    return df.sort_values(sorted_cols).reset_index(drop=True)
