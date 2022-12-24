"""
tune lr, wd for fgvc datasets and other datasets with train / val / test splits, should find the best results among 5 runs manually
"""
import os
import warnings

from time import sleep
from random import randint

from src.configs.config import get_cfg
from src.utils.file_io import PathManager

from train import train as train_main
from launch import default_argument_parser
warnings.filterwarnings("ignore")
# make small changes

def setup(args, lr, wd, check_runtime=True, seed=None):
    """
    Create configs and perform basic setups.
    overwrite the 2 parameters in cfg and args
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    cfg.SEED = seed
    
    # manually set to 5 for 5 runs.
    cfg.RUN_N_TIMES = 5

    # overwrite below four parameters
    lr = lr / 256 * cfg.DATA.BATCH_SIZE  # update lr based on the batchsize
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.WEIGHT_DECAY = wd
    
    if 'P_VK' in cfg.MODEL.TRANSFER_TYPE:
        P_NUM = cfg.MODEL.P_VK.NUM_TOKENS_P
        VK_NUM = cfg.MODEL.P_VK.NUM_TOKENS
        SHARED = cfg.MODEL.P_VK.SHARE_PARAM_KV
        if SHARED == True:
            marker = 1
        else:
            marker = 0
        Data_Name_With_PVK = cfg.DATA.NAME + f"_P{P_NUM}_VK{VK_NUM}_SHARED_{marker}"
    
    # setup output dir
    # output_dir / data_name / feature_name / lr_wd / run1
    output_dir = cfg.OUTPUT_DIR + "_val_fgvc"
    if 'P_VK' in cfg.MODEL.TRANSFER_TYPE:
        output_folder = os.path.join(
        Data_Name_With_PVK, cfg.DATA.FEATURE, f"lr{lr}_wd{wd}"
    )
    else:   
        output_folder = os.path.join(
            cfg.DATA.NAME, cfg.DATA.FEATURE, f"lr{lr}_wd{wd}"
        )
    # output_folder = os.path.splitext(os.path.basename(args.config_file))[0]

    # train cfg.RUN_N_TIMES times
    if check_runtime:
        count = 1
        # print('Should run times:', cfg.RUN_N_TIMES)
        # print('Current time', count)
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
        # only used for dummy config file
        output_path = os.path.join(output_dir, output_folder, f"run1")
        cfg.OUTPUT_DIR = output_path

    cfg.freeze()
    return cfg


def finetune_main(args):
    lr_range = [0.001, 0.0001, 0.0005, 0.005]
    wd_range = [0.01, 0.001, 0.0001, 0.0]
    for wd in wd_range:
        for lr in lr_range:
            # set up cfg and args
            try:
                cfg = setup(args, lr, wd)
            except ValueError:
                continue
            train_main(cfg, args)


def finetune_rn_main(args):
    lr_range = [
        0.05, 0.025, 0.005, 0.0025
    ]
    wd_range = [0.01, 0.001, 0.0001, 0.0]
    for wd in wd_range:
        for lr in lr_range:
            # set up cfg and args
            try:
                cfg = setup(args, lr, wd)
            except ValueError as e:
                print(e)
                continue
            train_main(cfg, args)


def prompt_rn_main(args):
    lr_range = [
        0.05, 0.025, 0.01, 0.5, 0.25, 0.1,
        1.0, 2.5, 5.
    ]
    wd_range = [0.01, 0.001, 0.0001, 0.0]
    for lr in sorted(lr_range, reverse=True):
        for wd in wd_range:
            # set up cfg and args
            try:
                cfg = setup(args, lr, wd)
            except ValueError as e:
                print(e)
                continue
            train_main(cfg, args)


def linear_main(args):
    lr_range = [
        50.0, 25., 10.0,
        5.0, 2.5, 1.0,
        0.5, 0.25, 0.1, 0.05
    ]
    wd_range = [0.01, 0.001, 0.0001, 0.0]
    for lr in lr_range:
        for wd in wd_range:
            # set up cfg and args
            try:
                cfg = setup(args, lr, wd)
            except ValueError:
                continue
            train_main(cfg, args)
            sleep(randint(1, 10))


def linear_mae_main(args):
    lr_range = [
        50.0, 25., 10.0,
        5.0, 2.5, 1.0,
        0.5, 0.25, 0.1, 0.05,
        0.025, 0.005, 0.0025,
    ]
    wd_range = [0.01, 0.001, 0.0001, 0.0]
    for lr in lr_range:
        for wd in wd_range:
            # set up cfg and args
            try:
                cfg = setup(args, lr, wd)
            except ValueError:
                continue
            train_main(cfg, args)
            sleep(randint(1, 10))


def prompt_main(args):
    lr_range = [
        5.0, 2.5, 1.0,
        50.0, 25., 10.0,
        0.5, 0.25, 0.1,
    ]
    wd_range = [0.01, 0.001, 0.0001, 0.0]
    for lr in lr_range:
        for wd in wd_range:
            # set up cfg and args
            try:
                cfg = setup(args, lr, wd)
            except ValueError:
                continue
            train_main(cfg, args)
            sleep(randint(1, 10))


def prompt_main_largerrange(args):
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

def MainSelf(args, files, data_name):

    lr, wd = find_best_lrwd(files, data_name)
    # final run 5 times with fixed seed
    random_seeds = [42, 44, 82, 100, 800]
    for run_idx, seed in enumerate(random_seeds):
        try:
            # cfg = setup(args, lr, wd, run_idx=run_idx+1, seed=seed)
            cfg = setup(args, lr, wd, seed=seed, check_runtime=True)
        except ValueError:
            continue
        train_main(cfg, args)
        sleep(randint(1, 10))

def find_best_lrwd(files, data_name):
    t_name =  data_name # just data name
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
            if 'test_CUB' in line: # change test_files here for reference
                val_result = float(line.split('top1:')[1].split('top5:')[0][1:-1])
                # print(val_result)
                
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
    
    print('Combinations:', idx + 1)
    # print(best_lr)
    # print(best_wd)
        
    return best_lr, best_wd        


def main(args):
    """main function to call from workflow"""
    if args.train_type == "finetune":
        finetune_main(args)
    elif args.train_type == "finetune_resnet":
        finetune_rn_main(args)

    elif args.train_type == "linear":
        linear_main(args)
    elif args.train_type == "linear_mae":
        linear_mae_main(args)

    elif args.train_type == "prompt":
        prompt_main(args)
    elif args.train_type == "prompt_resnet":
        prompt_rn_main(args)
    elif args.train_type == "prompt_largerrange" or args.train_type == "prompt_largerlr":  # noqa
        prompt_main_largerrange(args)
    
    elif args.train_type == "QKV" or "P_VK":
        # print('!!!find_best_lrwd!!! In 1_find_best_lr_wd_fgvc.py')
        data_name = 'CUB'
        file = '/home/ch7858/vpt/output/CUB_P5_VK5_SHARED_1/sup_vitb16_224'
        MainSelf(args, file, data_name)
    # elif args.train_type == "QKV_resnet":
        # prompt_rn_main(args)
    elif args.train_type == "QKV_largerrange" or args.train_type == "QKV_largerlr" or args.train_type == "P_VK_largerrange" or args.train_type == "P_VK_largerlr":  # noqa
        MainSelf(args, file, data_name)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)
