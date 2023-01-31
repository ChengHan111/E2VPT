"""
tune lr, wd for fgvc datasets and other datasets with train / val / test splits, should find the best results among 5 runs manually
"""
import os
import warnings

import torch
import glob
from time import sleep
from random import randint

import src.utils.logging as logging
from src.configs.config import get_cfg
from src.utils.file_io import PathManager
from src.data import loader as data_loader
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.models.build_model import build_model

from train import train as train_main
from launch import default_argument_parser, logging_train_setup
warnings.filterwarnings("ignore")

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


def setup(args, lr, wd, final_runs, check_runtime=True, run_idx=None, seed=None):
    """
    Create configs and perform basic setups.
    overwrite the 2 parameters in cfg and args
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)


    # overwrite below four parameters
    if final_runs != 'final_runs':
        lr = lr / 256 * cfg.DATA.BATCH_SIZE  # update lr based on the batchsize
        cfg.SOLVER.BASE_LR = lr
        cfg.SOLVER.WEIGHT_DECAY = wd
    
    if 'P_VK' in cfg.MODEL.TRANSFER_TYPE:
        P_NUM = cfg.MODEL.P_VK.NUM_TOKENS_P
        VK_NUM = cfg.MODEL.P_VK.NUM_TOKENS
        SHARED = cfg.MODEL.P_VK.SHARE_PARAM_KV
        INIT = cfg.MODEL.P_VK.ORIGIN_INIT
        SHARED_ACC = cfg.MODEL.P_VK.SHARED_ACCROSS
        BS = cfg.DATA.BATCH_SIZE
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
        Data_Name_With_PVK = cfg.DATA.NAME + f"_P{P_NUM}_VK{VK_NUM}_SHARED_{marker}_INIT_{init}_ACC_{shared_acc}_BS{BS}_ORIGIN"
    
    # setup output dir
    # output_dir / data_name / feature_name / lr_wd / run1
    
    # train cfg.RUN_N_TIMES times
    if final_runs == 'init_train':
        cfg.RUN_N_TIMES = 1
        cfg.MODEL.SAVE_CKPT = False
        cfg.MODEL.SAVE_CKPT_FINALRUNS = False
    
    elif final_runs == 'check_best_lrwd':
        
        cfg.RUN_N_TIMES = 1
        cfg.MODEL.SAVE_CKPT_FINALRUNS = False 
        cfg.MODEL.SAVE_CKPT = False
        # find the best lr and best wd
        if 'P_VK' in cfg.MODEL.TRANSFER_TYPE:
            # files = glob.glob(f"{cfg.OUTPUT_DIR}/{Data_Name_With_PVK}/{cfg.DATA.FEATURE}/*/run1/logs.txt")
            folder = f"{cfg.OUTPUT_DIR}/{Data_Name_With_PVK}/{cfg.DATA.FEATURE}"
            lr, wd = find_best_lrwd(folder, cfg.DATA.NAME)
            print('!!!!!!!', lr)
            print('@@@@@@', wd)
        else:
            raise ValueError("Not supported")
            
        cfg.SOLVER.BASE_LR = lr # this change the corresponding lr and wd (no need to change during pruning)
        cfg.SOLVER.WEIGHT_DECAY = wd

    # rewind process
    elif final_runs == 'final_runs':
        cfg.SEED = seed # put input seed here
        cfg.RUN_N_TIMES = 5
        cfg.MODEL.SAVE_CKPT_FINALRUNS = False # change this to true to enable model saving
        cfg.MODEL.SAVE_CKPT = False
        cfg.SOLVER.BASE_LR = lr
        cfg.SOLVER.WEIGHT_DECAY = wd
        print('lr', lr)
        print('wd', wd)

        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + "_fgvc_finalfinal"
        
    else:
        raise ValueError(
                f"Unsupported setup config! Check tune_vtab setup for more detail")

    # setup output dir
    output_dir = cfg.OUTPUT_DIR
    
    if final_runs != 'check_best_lrwd':
        output_folder = os.path.join(Data_Name_With_PVK, cfg.DATA.FEATURE, f"lr{lr}_wd{wd}")

    # only for lr, wd setup for check_best_lrwd stage.
    if final_runs != 'check_best_lrwd':
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

def find_best_lrwd(files, data_name):
    best_lr = None
    best_wd = None
    best_val_acc = -1
    for idx, folder in enumerate(os.listdir(str(files))):
        log_path = files + '/' + folder + '/run1/logs.txt'
        try:
            f = open(log_path, encoding="utf-8")
        except Exception as e:
            print(f"Encounter issue: {e} for file {f}")
            continue
        
        line = f.readline()
        cnt = 1
        while line:
            # print("Line {}: {}".format(cnt, line.strip()))
            val_name = 'val_' + data_name
            if val_name in line: # change test_files here for reference
                # print('exist!')
                val_result = float(line.split('top1:')[1].split('top5:')[0][1:-1])
                
                if val_result == best_val_acc:
                    frag_txt = folder
                    cur_lr = float(frag_txt.split("lr")[-1].split("_wd")[0])
                    cur_wd = float(frag_txt.split("_wd")[-1])
                    if best_lr is not None and cur_lr < best_lr:
                        # get the smallest lr to break tie for stability
                        best_lr = cur_lr
                        best_wd = cur_wd
                        best_val_acc = val_result

                elif val_result > best_val_acc:
                    best_val_acc = val_result
                    frag_txt = folder
                    best_lr = float(frag_txt.split("lr")[-1].split("_wd")[0])
                    best_wd = float(frag_txt.split("_wd")[-1])
                
                
            line = f.readline()
            cnt += 1
    
    # list useful info
    print('Combinations:', idx + 1)
    print('best_lr:', best_lr)
    print('best_wd', best_wd)
    
    return best_lr, best_wd     

def QKV_main(args):
    # normal lr range and wd_range
    lr_range = [
        5.0, 2.5, 1.0,
        50.0, 25., 10.0,
        0.5, 0.25, 0.1,
    ]
    wd_range = [0.01, 0.001, 0.0001, 0.0]
    # lr_range = [
    #     0.5, 0.25
    # ]
    # wd_range = [0.01]
    
    for lr in lr_range:
        for wd in wd_range:
            # set up cfg and args
            try:
                cfg = setup(args, lr, wd, final_runs='init_train')
            except ValueError:
                continue
            train_main(cfg, args)
            sleep(randint(1, 10))
    
    # run and save best lr, wd to cfg
    cfg = setup(args, 0.1, 0.1, final_runs='check_best_lrwd')

    # get best results on rewind options
    # final run 5 times with fixed seed
    random_seeds = [42, 44, 82, 100, 800]
    for run_idx, seed in enumerate(random_seeds):
        try:
            cfg = setup(
                args, cfg.SOLVER.BASE_LR, cfg.SOLVER.WEIGHT_DECAY, final_runs='final_runs', run_idx=run_idx+1, seed=seed)
        except ValueError:
            continue
        train_main(cfg, args)


def QKV_main_largerrange(args):
    lr_range = [
        500, 1000,    # for parralel-based prompt for stanford cars
        250., 100.0,  # for parralel-based prompt for stanford cars
    ]
    wd_range = [0.0, 0.01, 0.001, 0.0001]
    for lr in lr_range:
        for wd in wd_range:
            # set up cfg and args
            try:
                cfg = setup(args, lr, wd)
            except ValueError:
                continue
            train_main(cfg, args)
            sleep(randint(1, 10))

def main(args):
    # default for train_type=='P_VK' (design for it)
    QKV_main(args)

if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)
