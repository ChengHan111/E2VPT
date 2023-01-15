#!/usr/bin/env python3
"""
major actions here for training VTAB datasets: use val200 to find best lr/wd, and retrain on train800val200, report results on test
"""
import glob
import numpy as np
import os
import torch
import warnings
import random
import json

from time import sleep
from random import randint
from fvcore.common.checkpoint import Checkpointer

import src.utils.logging as logging
from src.configs.config import get_cfg
from src.data import loader as data_loader
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.models.build_model import build_model
from src.utils.file_io import PathManager

from launch import default_argument_parser, logging_train_setup
warnings.filterwarnings("ignore")
DATA2CLS = {
    'caltech101': 102,
    'cifar(num_classes=100)': 100,
    'dtd': 47,
    'oxford_flowers102': 102,
    'oxford_iiit_pet': 37,
    'patch_camelyon': 2,
    'sun397': 397,
    'svhn': 10,
    'resisc45': 45,
    'eurosat': 10,
    'dmlab': 6,
    'kitti(task="closest_vehicle_distance")': 4,
    'smallnorb(predicted_attribute="label_azimuth")': 18,
    'smallnorb(predicted_attribute="label_elevation")': 9,
    'dsprites(predicted_attribute="label_x_position",num_classes=16)': 16,
    'dsprites(predicted_attribute="label_orientation",num_classes=16)': 16,
    'clevr(task="closest_object_distance")': 6,
    'clevr(task="count_all")': 8,
    'diabetic_retinopathy(config="btgraham-300")': 5
}


def find_best_lrwd(files, data_name):
    
    t_name = "val_" + data_name
    best_lr = None
    best_wd = None
    best_val_acc = -1
    for f in files:
        try:
            results_dict = torch.load(f, "cpu")
            epoch = len(results_dict) - 1
            val_result = results_dict[f"epoch_{epoch}"]["classification"][t_name]["top1"]
            val_result = float(val_result)
            # print(val_result)
        except Exception as e:
            print(f"Encounter issue: {e} for file {f}")
            continue

        if val_result == best_val_acc:
            frag_txt = f.split("/run")[0]
            cur_lr = float(frag_txt.split("/lr")[-1].split("_wd")[0])
            cur_wd = float(frag_txt.split("_wd")[-1])
            if best_lr is not None and cur_lr < best_lr:
                # get the smallest lr to break tie for stability
                best_lr = cur_lr
                best_wd = cur_wd
                best_val_acc = val_result

        elif val_result > best_val_acc:
            best_val_acc = val_result
            frag_txt = f.split("/run")[0]
            best_lr = float(frag_txt.split("/lr")[-1].split("_wd")[0])
            best_wd = float(frag_txt.split("_wd")[-1])
    return best_lr, best_wd

def find_best_MtMtp(files, data_name):
    
    t_name = "val_" + data_name
    best_mask_token = None
    best_mask_token_piece = None
    best_val_acc = -1
    for f in files:
        try:
            results_dict = torch.load(f, "cpu")
            epoch = len(results_dict) - 1
            val_result = results_dict[f"epoch_{epoch}"]["classification"][t_name]["top1"]
            val_result = float(val_result)
            # print(val_result)
        except Exception as e:
            print(f"Encounter issue: {e} for file {f}")
            continue

        if val_result == best_val_acc:
            frag_txt = f.split("run1")[1]
            # cur_lr = float(frag_txt.split("/lr")[-1].split("_wd")[0])
            # cur_wd = float(frag_txt.split("_wd")[-1])

            cur_mask_token = float(frag_txt.split("/rewind")[-1].split('_tokens')[0])
            cur_mask_token_piece = float(frag_txt.split("tokens_")[-1].split('_pieces')[0])
            # 这里不一样的点是选择了尽可能大的mask
            # change into default setting
            if best_mask_token is not None and best_mask_token < cur_mask_token :
                # get the smallest lr to break tie for stability
                print('pass best_mask_token < cur_mask_token situation')
                best_mask_token = cur_mask_token
                best_mask_token_piece = cur_mask_token_piece
                best_val_acc = val_result

        # larger is better for val results
        elif val_result > best_val_acc:
            print('PASS!!!!')
            best_val_acc = val_result
            frag_txt = f.split("run1")[1]
            # best_lr = float(frag_txt.split("/lr")[-1].split("_wd")[0])
            # best_wd = float(frag_txt.split("_wd")[-1])
            best_mask_token = int(frag_txt.split("/rewind_")[-1].split('_tokens')[0])
            best_mask_token_piece = int(frag_txt.split("tokens_")[-1].split('_pieces')[0])
            print('best_mask_token', best_mask_token, best_mask_token_piece)
        
    return best_mask_token, best_mask_token_piece


def setup(args, lr, wd, final_runs='init_train', run_idx=None, seed=None):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.SEED = seed

    # create the clsemb_path for this dataset, only support vitb-sup experiments
    if cfg.DATA.FEATURE == "sup_vitb16_imagenet21k":
        cfg.MODEL.PROMPT.CLSEMB_PATH = os.path.join(
            cfg.MODEL.PROMPT.CLSEMB_FOLDER, "{}.npy".format(cfg.DATA.NAME))
        
    if 'P_VK' in cfg.MODEL.TRANSFER_TYPE:
        P_NUM = cfg.MODEL.P_VK.NUM_TOKENS_P
        VK_NUM = cfg.MODEL.P_VK.NUM_TOKENS
        SHARED = cfg.MODEL.P_VK.SHARE_PARAM_KV
        INIT = cfg.MODEL.P_VK.ORIGIN_INIT
        SHARED_ACC = cfg.MODEL.P_VK.SHARED_ACCROSS
        if SHARED == True:
            marker = 1
        else:
            marker = 0
        if INIT == 0:
            init = 0
        elif INIT == 1:
            init = 1
        else:
            init = 2
        if SHARED_ACC == True:
            shared_acc = 1
        else:
            shared_acc = 0
        Data_Name_With_PVK = cfg.DATA.NAME + f"_P{P_NUM}_VK{VK_NUM}_SHARED_{marker}_INIT_{init}_ACC_{shared_acc}"


    if final_runs == 'init_train':
        cfg.RUN_N_TIMES = 1
        cfg.MODEL.SAVE_CKPT = False
        cfg.MODEL.SAVE_CKPT_FINALRUNS = False
        
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + "_val"
        lr = lr / 256 * cfg.DATA.BATCH_SIZE  # update lr based on the batchsize
        cfg.SOLVER.BASE_LR = lr
        cfg.SOLVER.WEIGHT_DECAY = wd
    
    elif final_runs == 'before_pruning':
        cfg.RUN_N_TIMES = 1
        cfg.MODEL.SAVE_CKPT_FINALRUNS = True # enable ckpt saving during 'before_pruning' stage(need gradient during pruning)
        cfg.MODEL.SAVE_CKPT = False
        # find the best lr and best wd
        if 'P_VK' in cfg.MODEL.TRANSFER_TYPE:
            files = glob.glob(f"{cfg.OUTPUT_DIR}_val/{Data_Name_With_PVK}/{cfg.DATA.FEATURE}/*/run1/eval_results.pth")
            lr, wd = find_best_lrwd(files, cfg.DATA.NAME)
            print('!!!!!!!', lr)
            print('@@@@@@', wd)
        else:
            files = glob.glob(f"{cfg.OUTPUT_DIR}_val/{cfg.DATA.NAME}/{cfg.DATA.FEATURE}/*/run1/eval_results.pth")
            lr, wd = find_best_lrwd(files, cfg.DATA.NAME)
            
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + "_before_pruning"
        cfg.SOLVER.BASE_LR = lr # this change the corresponding lr and wd (no need to change during pruning)
        cfg.SOLVER.WEIGHT_DECAY = wd

    # rewind process
    elif final_runs == 'final_runs':
        cfg.RUN_N_TIMES = 5
        cfg.MODEL.SAVE_CKPT_FINALRUNS = False # change this to true to enable model saving
        cfg.MODEL.SAVE_CKPT = False
        # find the best lr and best wd
        if 'P_VK' in cfg.MODEL.TRANSFER_TYPE:
            
            cfg.SOLVER.BASE_LR = lr
            cfg.SOLVER.WEIGHT_DECAY = wd
            
            files = glob.glob(f"{cfg.OUTPUT_DIR}_before_pruning/{Data_Name_With_PVK}/{cfg.DATA.FEATURE}/lr{lr}_wd{wd}/run1/rewind/rewind_?_tokens_?_pieces_to_mask/eval_results.pth")
            print('should not be longer than 72', len(files))
            # notice that mask tokens and mask token pieces are selected in this process(before)
            
            # print('length of files (should be 72)', len(files))
            mt, mtr = find_best_MtMtp(files, cfg.DATA.NAME)
            mt, mtr = int(mt), int(mtr)
        
        else:
            files = glob.glob(f"{cfg.OUTPUT_DIR}_before_pruning/{Data_Name_With_PVK}/{cfg.DATA.FEATURE}/lr{lr}_wd{wd}/run1/rewind/rewind_?_tokens_?_pieces_to_mask/eval_results.pth")
            mt, mtr = find_best_MtMtp(files, cfg.DATA.NAME)
            mt, mtr = int(mt), int(mtr)
        
        print('Go through here (AT least!)')
        print('cfg.OUTPUT_DIR', cfg.OUTPUT_DIR)
        sleep(10)
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + "_rewind"
        
        cfg.MODEL.P_VK.REWIND_MASK_CLS_TOKEN_NUM = mt
        cfg.MODEL.P_VK.REWIND_MASK_CLS_TOKEN_PIECE_NUM = mtr
        cfg.MODEL.P_VK.REWIND_STATUS = True
        # cfg.MODEL.P_VK.PRUNING_SAVING_PATH = f"output_before_pruning/{Data_Name_With_PVK}/{cfg.DATA.FEATURE}/lr{cfg.SOLVER.BASE_LR}_wd{cfg.SOLVER.WEIGHT_DECAY}/run1"
        cfg.MODEL.P_VK.REWIND_OUTPUT_DIR = f"output_before_pruning/{Data_Name_With_PVK}/{cfg.DATA.FEATURE}/lr{cfg.SOLVER.BASE_LR}_wd{cfg.SOLVER.WEIGHT_DECAY}/run1"
        # print('00000', cfg.MODEL.P_VK.REWIND_OUTPUT_DIR)
        # print('11111', cfg.MODEL.P_VK.REWIND_MASK_CLS_TOKEN_NUM)
        # print('22222', cfg.MODEL.P_VK.REWIND_MASK_CLS_TOKEN_PIECE_NUM)
        print('At final runs:', cfg.MODEL.P_VK.REWIND_OUTPUT_DIR)
        
    else:
        raise ValueError(
                f"Unsupported setup config! Check tune_vtab setup for more detail")

    # setup output dir
    # output_dir / data_name / feature_name / lr_wd / run1
    output_dir = cfg.OUTPUT_DIR
    
    if final_runs != 'final_runs':
        output_folder = os.path.join(Data_Name_With_PVK, cfg.DATA.FEATURE, f"lr{lr}_wd{wd}")
    else:
        output_folder = os.path.join(Data_Name_With_PVK, cfg.DATA.FEATURE, f"lr{lr}_wd{wd}_mt{mt}_mtr{mtr}")
        # print(f"lr{cfg.SOLVER.BASE_LR}_wd{cfg.SOLVER.WEIGHT_DECAY}_mt{mt}_mtr{mtr}")
        
    # train cfg.RUN_N_TIMES times
    if run_idx is None:
        count = 1
        while count <= cfg.RUN_N_TIMES:
            output_path = os.path.join(output_dir, output_folder, f"run{count}")
            # pause for a random time, so concurrent process with same setting won't interfere with each other. # noqa
            sleep(randint(1, 5))
            if not PathManager.exists(output_path):
                PathManager.mkdirs(output_path)
                cfg.OUTPUT_DIR = output_path
                break
            else:
                count += 1
        if count > cfg.RUN_N_TIMES:
            raise ValueError(
                f"Already run {cfg.RUN_N_TIMES} times for {output_folder}, no need to run more")
    else:
        output_path = os.path.join(output_dir, output_folder, f"run{run_idx}")
        if not PathManager.exists(output_path):
            PathManager.mkdirs(output_path)
            cfg.OUTPUT_DIR = output_path
        else:
            raise ValueError(
                f"Already run run-{run_idx} for {output_folder}, no need to run more")

    cfg.freeze()
    return cfg


def get_loaders(cfg, logger, final_runs=False):
    # support two training paradims:
    # 1) train / val / test, using val to tune
    # 2) train / val: for imagenet

    if not final_runs:
        logger.info("Loading training data...")
        train_loader = data_loader.construct_train_loader(cfg)

        logger.info("Loading validation data...")
        val_loader = data_loader.construct_val_loader(cfg)
        # not really nessecary to check the results of test set.
        test_loader = None

    else:
        logger.info("Loading training data...")
        train_loader = data_loader.construct_trainval_loader(cfg)

        # not really nessecary to check the results of val set, but the trainer class does not support no-validation loader yet  # noqa
        logger.info("Loading validation data...")
        val_loader = data_loader.construct_val_loader(cfg)

        logger.info("Loading test data...")
        test_loader = data_loader.construct_test_loader(cfg)

    return train_loader, val_loader, test_loader


def train(cfg, args, final_runs):
    # clear up residual cache from previous runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # main training / eval actions here

    # fix the seed for reproducibility
    if cfg.SEED is not None:
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        random.seed(0)

    # setup training env including loggers
    logging_train_setup(args, cfg)
    logger = logging.get_logger("visual_prompt")

    train_loader, val_loader, test_loader = get_loaders(
        cfg, logger, final_runs)
    logger.info("Constructing models...")
    model, cur_device = build_model(cfg)

    logger.info("Setting up Evalutator...")
    evaluator = Evaluator()
    logger.info("Setting up Trainer...")
    trainer = Trainer(cfg, model, evaluator, cur_device)

    if train_loader:
        trainer.train_classifier(train_loader, val_loader, test_loader)
        # save the evaluation results
        torch.save(
            evaluator.results,
            os.path.join(cfg.OUTPUT_DIR, "eval_results.pth")
        )
    else:
        print("No train loader presented. Exit")

def cal_mt_mtp(cfg, args, final_runs):
    # clear up residual cache from previous runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # main training / eval actions here

    # setup training env including loggers
    logging_train_setup(args, cfg)
    logger = logging.get_logger("visual_prompt")

    train_loader, val_loader, test_loader = get_loaders(
        cfg, logger, final_runs)
    logger.info("Constructing models...")
    model_init, cur_device_init = build_model(cfg)
    
    model = model_init
    cur_device = cur_device_init

    logger.info("Setting up Evalutator...")
    evaluator = Evaluator()
    logger.info("Setting up Eval_self(for masking stage)")
    trainer = Trainer(cfg, model, evaluator, cur_device)

    # if train_loader and cfg.MODEL.SAVE_CKPT_FINALRUNS is True:
    if train_loader:
        cls_token_pieces_num = cfg.MODEL.P_VK.CLS_TOKEN_P_PIECES_NUM
        cls_token_num = cfg.MODEL.P_VK.NUM_TOKENS_P
        soft_tokens_importance, soft_tokens_pieces_importance = trainer.calculate_importance(
            cfg, model, train_loader, n_pieces_token=cls_token_pieces_num, n_soft_tokens=cls_token_num)
        
        logger.info("Soft prompt tokens importance scores")

        tokens_mask_sequence, tokens_pieces_mask_sequence = trainer.determine_mask_sequence(
            cfg, n_pieces_token=cls_token_pieces_num, n_soft_tokens=cls_token_num)
        
        logger.info("Find Prompt Masks on CLS_TOKEN")
        
        soft_tokens_to_mask = set()
        total_masked = 0
        soft_token_mask_dir = os.path.join(cfg.OUTPUT_DIR, 'mask_tokens')
        os.makedirs(soft_token_mask_dir, exist_ok=True)
        
        for step, n_to_mask in enumerate(tokens_mask_sequence):
            soft_tokens_to_mask = trainer.what_tokens_to_mask(
                soft_tokens_importance,
                n_to_mask,
                soft_tokens_to_mask,
                cfg.MODEL.P_VK.MIN_NUMBER_CLS_TOKEN,
                n_pieces_token = cls_token_pieces_num,
                n_soft_tokens = cls_token_num,
                reverse = cfg.MODEL.P_VK.MASK_RESERVE
            )
            total_masked += n_to_mask
            logger.info("Number of soft tokens to be masked: {}".format(total_masked))
            
            soft_token_mask_file = os.path.join(soft_token_mask_dir, "{}_soft_tokens_to_mask.json".format(total_masked))
            soft_tokens_to_mask_json_map = {total_masked:soft_tokens_to_mask}
            trainer.dump(soft_token_mask_file, soft_tokens_to_mask_json_map)
            logger.info("{} number of soft tokens be masked have been saved in {}".format(total_masked, soft_token_mask_file))

        soft_tokens_pieces_to_mask = {}
        total_masked_tokens_pieces = 0
        soft_tokens_pieces_mask_dir = os.path.join(cfg.OUTPUT_DIR, 'mask_tokens_pieces')
        os.makedirs(soft_tokens_pieces_mask_dir, exist_ok=True)
        
        for step, n_to_mask in enumerate(tokens_pieces_mask_sequence):
            soft_tokens_pieces_to_mask = trainer.what_tokens_pieces_to_mask(
                soft_tokens_pieces_importance,
                n_to_mask,
                soft_tokens_pieces_to_mask,
                cfg.MODEL.P_VK.MIN_NUMBER_CLS_TOKEN_PIECE,
                n_pieces_token = cls_token_pieces_num,
                n_soft_tokens = cls_token_num,
                reverse = cfg.MODEL.P_VK.MASK_RESERVE
            )
            total_masked_tokens_pieces += n_to_mask
            logger.info("Number of soft tokens pieces to be masked: {}".format(total_masked_tokens_pieces))
            soft_tokens_pieces_mask_file = os.path.join(soft_tokens_pieces_mask_dir, "{}_soft_tokens_pieces_to_mask.json".format(total_masked_tokens_pieces))
            trainer.dump(soft_tokens_pieces_mask_file, soft_tokens_pieces_to_mask)
            logger.info("{} number of soft tokens pieces be masked have been saved in {}".format(total_masked_tokens_pieces, soft_tokens_pieces_mask_file))  
        
    else:
        print("No train loader presented. Exit")

def rewind_train(cfg, args, cls_token_id, cls_token_pieces_id, rewind_model_output_dir, final_runs):
    # clear up residual cache from previous runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # regenerate related config here: before the config setup stage.
    cfg.defrost()
    cfg.MODEL.P_VK.REWIND_MASK_CLS_TOKEN_NUM = cls_token_id
    cfg.MODEL.P_VK.REWIND_MASK_CLS_TOKEN_PIECE_NUM = cls_token_pieces_id
    cfg.MODEL.P_VK.REWIND_STATUS = True # change rewind status to true to enable rewind process
    cfg.MODEL.P_VK.REWIND_OUTPUT_DIR = cfg.OUTPUT_DIR
    cfg.freeze()
    
    print('before cfg setup stage!!!!!!')
    print('cfg.MODEL.P_VK.REWIND_OUTPUT_DIR', cfg.MODEL.P_VK.REWIND_OUTPUT_DIR)
    sleep(5)
    
    # main training / eval actions here

    # setup training env including loggers
    logging_train_setup(args, cfg)
    logger = logging.get_logger("visual_prompt")

    train_loader, val_loader, test_loader = get_loaders(
        cfg, logger, final_runs)
    logger.info("Constructing models...")
    model_init, cur_device_init = build_model(cfg)
    
    model = model_init
    cur_device = cur_device_init

    logger.info("Setting up Evalutator...")
    evaluator = Evaluator()
    logger.info("Setting up Eval_self(for masking stage)")
    trainer = Trainer(cfg, model, evaluator, cur_device)
    # if cfg.DO_REWIND is True: 
    logger.info('Rewind & train')
    
    # cls_token_pieces_num = cfg.MODEL.P_VK.CLS_TOKEN_P_PIECES_NUM
    # cls_token_num = cfg.MODEL.P_VK.NUM_TOKENS_P
    
    # TODO: remove this line in formal ver.
    # assert cls_token_id and cls_token_pieces_id here for debugging
    # cls_token_id, cls_token_pieces_id = 3, 1
            
    if train_loader:
        trainer.train_classifier(train_loader, val_loader, test_loader)
        # save the evaluation results
        torch.save(
            evaluator.results,
            os.path.join(rewind_model_output_dir, "eval_results.pth")
        )
    else:
        print("No train loader presented. Exit")

def get_lrwd_range(args):

    if args.train_type == "finetune":
        lr_range = [0.001, 0.0001, 0.0005, 0.005]
        wd_range = [0.01, 0.001, 0.0001, 0.0]

    elif args.train_type == "finetune_resnet":
        lr_range = [
            0.0005, 0.00025,
            0.5, 0.25, 0.05, 0.025, 0.005, 0.0025,
        ]
        wd_range = [0.01, 0.001, 0.0001, 0.0]

    elif args.train_type == "linear":
        lr_range = [
            50.0, 25., 10.0,
            5.0, 2.5, 1.0,
            0.5, 0.25, 0.1, 0.05
        ]
        wd_range = [0.01, 0.001, 0.0001, 0.0]

    elif args.train_type == "linear_mae":
        lr_range = [
            50.0, 25., 10.0,
            5.0, 2.5, 1.0,
            0.5, 0.25, 0.1, 0.05,
            0.025, 0.005, 0.0025,
        ]
        wd_range = [0.01, 0.001, 0.0001, 0.0]

    elif args.train_type == "prompt":
        lr_range = [
            5.0, 2.5, 1.0,
            50.0, 25., 10.0,
            0.5, 0.25, 0.1, 0.05
        ]
        wd_range = [0.01, 0.001, 0.0001, 0.0]

    elif args.train_type == "prompt_largerlr":
        lr_range = [
            500, 1000, 250., 100.0,
        ]
        wd_range = [0.01, 0.001, 0.0001, 0.0]

    elif args.train_type == "prompt_resnet":
        lr_range = [
            0.05, 0.025, 0.01, 0.5, 0.25, 0.1,
            1.0, 2.5, 5.
        ]
        wd_range = [0.01, 0.001, 0.0001, 0.0]
    
    elif args.train_type == "QKV" or args.train_type == "P_VK":
        lr_range = [
            5.0, 2.5, 1.0,
            50.0, 25., 10.0,
            0.5, 0.25, 0.1, 0.05
        ]
        wd_range = [0.01, 0.001, 0.0001, 0.0]
        # lr_range = [5.0, 2.5]
        # wd_range = [0.01, 0.001]

    elif args.train_type == "QKV_largerlr" or args.train_type == "P_KV_largerlr":
        lr_range = [
            500, 1000, 250., 100.0,
        ]
        wd_range = [0.01, 0.001, 0.0001, 0.0]

    # elif args.train_type == "prompt_resnet":
    #     lr_range = [
    #         0.05, 0.025, 0.01, 0.5, 0.25, 0.1,
    #         1.0, 2.5, 5.
    #     ]
        # wd_range = [0.01, 0.001, 0.0001, 0.0]

    return lr_range, wd_range


def main(args):
    """main function to call from workflow"""
    # tuning lr and wd first:
    lr_range, wd_range = get_lrwd_range(args)

    # try all combinations
    for lr in sorted(lr_range, reverse=True):
        for wd in sorted(wd_range, reverse=True):
            try:
                cfg = setup(args, lr, wd, final_runs='init_train')
            except ValueError:
                continue
            train(cfg, args, final_runs=False)
    
    # run and save best lr, wd combination model (only 1 time) before pruning
    cfg = setup(args, 0.1, 0.1, final_runs='before_pruning')
    train(cfg, args, final_runs=False) # originally True here.
    
    # keep the same config as before_pruning (but create mask token and mask token pieces json.)
    cal_mt_mtp(cfg, args, final_runs=False)

    # train rewind
    # cls_token_id
    cls_token_id_dir = cfg.OUTPUT_DIR + '/mask_tokens'
    cls_token_pieces_id_dir = cfg.OUTPUT_DIR + '/mask_tokens_pieces'
    
    cls_token_id_list, cls_token_pieces_id_list = [], []
    for cls_token_file_name in os.listdir(cls_token_id_dir):
        cls_token_id_list.append(int(cls_token_file_name.split('_soft')[0]))
        
    for cls_token_pieces_file_name in os.listdir(cls_token_pieces_id_dir):
        cls_token_pieces_id_list.append(int(cls_token_pieces_file_name.split('_soft')[0]))
    
    # print('cls_token_id_list', cls_token_id_list)
    # print('cls_token_pieces_id_list', cls_token_pieces_id_list)
    
    assert cls_token_id_list is not None
    assert cls_token_pieces_id_list is not None
    
    # TODO: remove this line in formal ver.
    # assert cls_token_id and cls_token_pieces_id here for debugging
    # cls_token_id, cls_token_pieces_id = 9, 8
    # rewind_model_output_dir = os.path.join(
    #     cfg.OUTPUT_DIR, f"rewind_{cls_token_id}_tokens_{cls_token_pieces_id}_pieces_to_mask")
    # os.makedirs(rewind_model_output_dir, exist_ok=True)
    # rewind_train(cfg, args, cls_token_id, cls_token_pieces_id, rewind_model_output_dir, final_runs=False)
    
    # cls_token_id, cls_token_pieces_id = 6, 3
    # rewind_model_output_dir = os.path.join(
    # cfg.OUTPUT_DIR, f"rewind_{cls_token_id}_tokens_{cls_token_pieces_id}_pieces_to_mask")
    # os.makedirs(rewind_model_output_dir, exist_ok=True)
    # rewind_train(cfg, args, cls_token_id, cls_token_pieces_id, rewind_model_output_dir, final_runs=False)
    
    
    for run_idx, cls_token_id in enumerate(cls_token_id_list):
        for run_idx_2, cls_token_pieces_id in enumerate(cls_token_pieces_id_list):
            rewind_model_output_dir = os.path.join(
                cfg.OUTPUT_DIR, f"rewind/rewind_{cls_token_id}_tokens_{cls_token_pieces_id}_pieces_to_mask")
            os.makedirs(rewind_model_output_dir, exist_ok=True)
            rewind_train(cfg, args, cls_token_id, cls_token_pieces_id, rewind_model_output_dir, final_runs=False)
    
    print('Finish rewind process, get final runs')
    sleep(5)
    
    # get best results on rewind options
    # final run 5 times with fixed seed
    random_seeds = [42, 44, 82, 100, 800]
    for run_idx, seed in enumerate(random_seeds):
        try:
            cfg = setup(
                args, cfg.SOLVER.BASE_LR, cfg.SOLVER.WEIGHT_DECAY, final_runs='final_runs', run_idx=run_idx+1, seed=seed)
        except ValueError:
            # already ran
            continue
        train(cfg, args, final_runs=True)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)