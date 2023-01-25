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

def setup(args, lr, wd, final_runs, check_runtime=True):
    """
    Create configs and perform basic setups.
    overwrite the 2 parameters in cfg and args
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)


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
    if final_runs == 'init_train':
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
        cfg.SOLVER.BASE_LR = lr
        cfg.SOLVER.WEIGHT_DECAY = wd

        if 'P_VK' in cfg.MODEL.TRANSFER_TYPE:
            
            files = glob.glob(f"{cfg.OUTPUT_DIR}_before_pruning/{Data_Name_With_PVK}/{cfg.DATA.FEATURE}/lr{lr}_wd{wd}/run1/rewind/*/eval_results.pth")
            print('should not be longer than 72', len(files))
            # print(files)
            # notice that mask tokens and mask token pieces are selected in this process(before)
            
            # print('length of files (should be 72)', len(files))
            mt, mtr = find_best_MtMtp(files, cfg.DATA.NAME)
            mt, mtr = int(mt), int(mtr)
        
        else:
            files = glob.glob(f"{cfg.OUTPUT_DIR}_before_pruning/{Data_Name_With_PVK}/{cfg.DATA.FEATURE}/lr{lr}_wd{wd}/run1/rewind/*/eval_results.pth")
            mt, mtr = find_best_MtMtp(files, cfg.DATA.NAME)
            mt, mtr = int(mt), int(mtr)
        
        # print('cfg.OUTPUT_DIR', cfg.OUTPUT_DIR)
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

def QKV_main(args):
    # normal lr range and wd_range
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
                cfg = setup(args, lr, wd, final_runs='init_train')
            except ValueError:
                continue
            train_main(cfg, args)
            sleep(randint(1, 10))
    
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
    
    for run_idx, cls_token_id in enumerate(cls_token_id_list):
        for run_idx_2, cls_token_pieces_id in enumerate(cls_token_pieces_id_list):
            rewind_model_output_dir = os.path.join(
                cfg.OUTPUT_DIR, f"rewind/rewind_{cls_token_id}_tokens_{cls_token_pieces_id}_pieces_to_mask")
            os.makedirs(rewind_model_output_dir, exist_ok=True)
            rewind_train(cfg, args, cls_token_id, cls_token_pieces_id, rewind_model_output_dir, final_runs=False)
    
    print('Finish rewind process, get final runs')
    # sleep(5)
    
    # get best results on rewind options
    # final run 5 times with fixed seed
    random_seeds = [42, 44, 82, 100, 800]
    for run_idx, seed in enumerate(random_seeds):
        try:
            cfg = setup(
                args, cfg.SOLVER.BASE_LR, cfg.SOLVER.WEIGHT_DECAY, final_runs='final_runs', run_idx=run_idx+1, seed=seed)
        except ValueError:
            continue
        train(cfg, args, final_runs=True)


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
    
    if args.train_type == "QKV" or "P_VK":
        QKV_main(args)
    # elif args.train_type == "QKV_resnet":
        # prompt_rn_main(args)
    # elif args.train_type == "QKV_largerrange" or args.train_type == "QKV_largerlr" or args.train_type == "P_VK_largerrange" or args.train_type == "P_VK_largerlr":  # noqa
    #     QKV_main_largerrange(args)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)
