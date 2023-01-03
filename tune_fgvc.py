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

def setup(args, lr, wd, check_runtime=True):
    """
    Create configs and perform basic setups.
    overwrite the 2 parameters in cfg and args
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # setup dist # comment here
    # cfg.DIST_INIT_PATH = "tcp://{}:4000".format(os.environ["SLURMD_NODENAME"])

    # overwrite below four parameters
    lr = lr / 256 * cfg.DATA.BATCH_SIZE  # update lr based on the batchsize
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.WEIGHT_DECAY = wd
    
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
        # print(f"_P{P_NUM}_VK{VK_NUM}_SHARED_{marker}_INIT_{init}_ACC_{shared_acc}")
        Data_Name_With_PVK = cfg.DATA.NAME + f"_P{P_NUM}_VK{VK_NUM}_SHARED_{marker}_INIT_{init}_ACC_{shared_acc}"
    
    # setup output dir
    # output_dir / data_name / feature_name / lr_wd / run1
    output_dir = cfg.OUTPUT_DIR
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
        print('Should run times:', cfg.RUN_N_TIMES)
        print('Current time', count)
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

def QKV_main(args):
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
        QKV_main(args)
    # elif args.train_type == "QKV_resnet":
        # prompt_rn_main(args)
    elif args.train_type == "QKV_largerrange" or args.train_type == "QKV_largerlr" or args.train_type == "P_VK_largerrange" or args.train_type == "P_VK_largerlr":  # noqa
        QKV_main_largerrange(args)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)
