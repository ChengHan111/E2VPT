#!/usr/bin/env python3

"""Config system (based on Detectron's)."""

from .config_node import CfgNode


# Global config object
_C = CfgNode()
# Example usage:
#   from configs.config import cfg

_C.DBG = False
_C.OUTPUT_DIR = "./output_fgvc"
_C.RUN_N_TIMES = 5
# Perform benchmarking to select the fastest CUDNN algorithms to use
# Note that this may increase the memory usage and will likely not result
# in overall speedups when variable size inputs are used (e.g. COCO training)
_C.CUDNN_BENCHMARK = False

# Number of GPUs to use (applies to both training and testing)
_C.NUM_GPUS = 1
_C.NUM_SHARDS = 1

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries
_C.SEED = None
# _C.SAVE_VTAB_RESULTS_PTH = False # self added here for not saving tune_vtab pth results

# ----------------------------------------------------------------------
# Model options
# ----------------------------------------------------------------------
_C.MODEL = CfgNode()
_C.MODEL.TRANSFER_TYPE = "linear"  # one of linear, end2end, prompt, adapter, side, partial-1, tinytl-bias
_C.MODEL.WEIGHT_PATH = ""  # if resume from some checkpoint file
_C.MODEL.SAVE_CKPT = False
_C.MODEL.SAVE_CKPT_FINALRUNS = False # save models at final 5 runs (currently available at vtab)

_C.MODEL.MODEL_ROOT = "./models"  # root folder for pretrained model weights (changed here!)

_C.MODEL.TYPE = "vit"
_C.MODEL.MLP_NUM = 0

_C.MODEL.LINEAR = CfgNode()
_C.MODEL.LINEAR.MLP_SIZES = []
_C.MODEL.LINEAR.DROPOUT = 0.1

# ----------------------------------------------------------------------
# Prompt options
# ----------------------------------------------------------------------
_C.MODEL.PROMPT = CfgNode()
_C.MODEL.PROMPT.NUM_TOKENS = 5
_C.MODEL.PROMPT.LOCATION = "prepend"
# prompt initalizatioin: 
    # (1) default "random"
    # (2) "final-cls" use aggregated final [cls] embeddings from training dataset
    # (3) "cls-nolastl": use first 12 cls embeddings (exclude the final output) for deep prompt
    # (4) "cls-nofirstl": use last 12 cls embeddings (exclude the input to first layer)
_C.MODEL.PROMPT.INITIATION = "random"  # "final-cls", "cls-first12"
_C.MODEL.PROMPT.CLSEMB_FOLDER = ""
_C.MODEL.PROMPT.CLSEMB_PATH = ""
_C.MODEL.PROMPT.PROJECT = -1  # "projection mlp hidden dim"
_C.MODEL.PROMPT.DEEP = False # "whether do deep prompt or not, only for prepend location"


_C.MODEL.PROMPT.NUM_DEEP_LAYERS = None  # if set to be an int, then do partial-deep prompt tuning
_C.MODEL.PROMPT.REVERSE_DEEP = False  # if to only update last n layers, not the input layer
_C.MODEL.PROMPT.DEEP_SHARED = False  # if true, all deep layers will be use the same prompt emb
_C.MODEL.PROMPT.FORWARD_DEEP_NOEXPAND = False  # if true, will not expand input sequence for layers without prompt
# how to get the output emb for cls head:
    # original: follow the orignial backbone choice,
    # img_pool: image patch pool only
    # prompt_pool: prompt embd pool only
    # imgprompt_pool: pool everything but the cls token
_C.MODEL.PROMPT.VIT_POOL_TYPE = "original"
_C.MODEL.PROMPT.DROPOUT = 0.0
_C.MODEL.PROMPT.SAVE_FOR_EACH_EPOCH = False
# ----------------------------------------------------------------------
# adapter options
# ----------------------------------------------------------------------
_C.MODEL.ADAPTER = CfgNode()
_C.MODEL.ADAPTER.REDUCATION_FACTOR = 8
_C.MODEL.ADAPTER.STYLE = "Pfeiffer"

# ----------------------------------------------------------------------
# Solver options
# ----------------------------------------------------------------------
_C.SOLVER = CfgNode()
_C.SOLVER.LOSS = "softmax"
_C.SOLVER.LOSS_ALPHA = 0.01

_C.SOLVER.OPTIMIZER = "sgd"  # or "adamw"
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.PATIENCE = 300


_C.SOLVER.SCHEDULER = "cosine"

_C.SOLVER.BASE_LR = 0.01
_C.SOLVER.BIAS_MULTIPLIER = 1.              # for prompt + bias

_C.SOLVER.WARMUP_EPOCH = 5
_C.SOLVER.TOTAL_EPOCH = 30
_C.SOLVER.LOG_EVERY_N = 1000


_C.SOLVER.DBG_TRAINABLE = False # if True, will print the name of trainable params

# ----------------------------------------------------------------------
# QKV_insert options (inherited from 'prompt')
# ----------------------------------------------------------------------

_C.MODEL.QKV_insert = CfgNode()
_C.MODEL.QKV_insert.NUM_TOKENS = 5
# _C.MODEL.QKV_insert.LOCATION = "prepend"
# # prompt initalizatioin: 
#     # (1) default "random"
#     # (2) "final-cls" use aggregated final [cls] embeddings from training dataset
#     # (3) "cls-nolastl": use first 12 cls embeddings (exclude the final output) for deep prompt
#     # (4) "cls-nofirstl": use last 12 cls embeddings (exclude the input to first layer)
# _C.MODEL.QKV_insert.INITIATION = "random"  # "final-cls", "cls-first12"
# _C.MODEL.QKV_insert.CLSEMB_FOLDER = ""
# _C.MODEL.QKV_insert.CLSEMB_PATH = ""
# _C.MODEL.QKV_insert.PROJECT = -1  # "projection mlp hidden dim"
_C.MODEL.QKV_insert.DEEP = True # "whether do deep QKV or not, only for prepend location"


# _C.MODEL.QKV_insert.NUM_DEEP_LAYERS = None  # if set to be an int, then do partial-deep prompt tuning
# _C.MODEL.QKV_insert.REVERSE_DEEP = False  # if to only update last n layers, not the input layer
# _C.MODEL.QKV_insert.DEEP_SHARED = False  # if true, all deep layers will be use the same prompt emb
# _C.MODEL.QKV_insert.FORWARD_DEEP_NOEXPAND = False  # if true, will not expand input sequence for layers without prompt
# # how to get the output emb for cls head:
#     # original: follow the orignial backbone choice,
#     # img_pool: image patch pool only
#     # prompt_pool: prompt embd pool only
#     # imgprompt_pool: pool everything but the cls token
_C.MODEL.QKV_insert.VIT_POOL_TYPE = "original"
_C.MODEL.QKV_insert.DROPOUT = 0.0
_C.MODEL.QKV_insert.LAYER_BEHIND = True # to put layers behind new-added prompt in key/value
_C.MODEL.QKV_insert.SHARE_PARAM_KV = True # change it to False to init two parameters
# _C.MODEL.QKV_insert.SAVE_FOR_EACH_EPOCH = False


# ----------------------------------------------------------------------
# P_VK (Prompt with key and value) options
# ----------------------------------------------------------------------
_C.MODEL.P_VK = CfgNode()
_C.MODEL.P_VK.NUM_TOKENS_P = 5
_C.MODEL.P_VK.LOCATION = "prepend"
# prompt initalizatioin: 
    # (1) default "random"
    # (2) "final-cls" use aggregated final [cls] embeddings from training dataset
    # (3) "cls-nolastl": use first 12 cls embeddings (exclude the final output) for deep prompt
    # (4) "cls-nofirstl": use last 12 cls embeddings (exclude the input to first layer)
_C.MODEL.P_VK.INITIATION = "random"  # "final-cls", "cls-first12"
_C.MODEL.P_VK.CLSEMB_FOLDER = ""
_C.MODEL.P_VK.CLSEMB_PATH = ""
_C.MODEL.P_VK.PROJECT = -1  # "projection mlp hidden dim"
_C.MODEL.P_VK.DEEP_P = True # "whether do deep prompt or not, only for prepend location"


_C.MODEL.P_VK.NUM_DEEP_LAYERS = None  # if set to be an int, then do partial-deep prompt tuning
_C.MODEL.P_VK.REVERSE_DEEP = False  # if to only update last n layers, not the input layer
_C.MODEL.P_VK.DEEP_SHARED = False  # if true, all deep layers will be use the same prompt emb
_C.MODEL.P_VK.FORWARD_DEEP_NOEXPAND = False  # if true, will not expand input sequence for layers without prompt
# how to get the output emb for cls head:
    # original: follow the orignial backbone choice,
    # img_pool: image patch pool only
    # prompt_pool: prompt embd pool only
    # imgprompt_pool: pool everything but the cls token
_C.MODEL.P_VK.VIT_POOL_TYPE = "original"
_C.MODEL.P_VK.DROPOUT_P = 0.0
_C.MODEL.P_VK.SAVE_FOR_EACH_EPOCH = False

_C.MODEL.P_VK.NUM_TOKENS = 5
_C.MODEL.P_VK.DEEP = True # "whether do deep QKV or not, only for prepend location"
_C.MODEL.P_VK.DROPOUT = 0.0
_C.MODEL.P_VK.LAYER_BEHIND = True # to put layers behind new-added prompt in key/value
_C.MODEL.P_VK.SHARE_PARAM_KV = True # change it to False to init two parameters
_C.MODEL.P_VK.ORIGIN_INIT = 2 # 0 for default, 1 for trunc_norm, 2 for kaiming init 
_C.MODEL.P_VK.SHARED_ACCROSS = False # share vk value accross multi-attn layers

# Turn it to False when considering without MASK_CLS_TOKEN
_C.MODEL.P_VK.MASK_CLS_TOKEN = True # set as the MAIN trigger to all cls token masked program(prouning and rewind process).
_C.MODEL.P_VK.NORMALIZE_SCORES_BY_TOKEN = False # new added for normalized token (apply as xprompt)
_C.MODEL.P_VK.CLS_TOKEN_MASK = True # new added for cls token mask (own or disown whole prompt)
_C.MODEL.P_VK.CLS_TOKEN_MASK_PERCENT_NUM = None # set specific num of percent to mask
_C.MODEL.P_VK.CLS_TOKEN_MASK_PERCENT = [10, 20, 30, 40, 50, 60, 70, 80, 90] # percentage applied during selected
_C.MODEL.P_VK.MIN_NUMBER_CLS_TOKEN = 1 # set the lower boundary to avoid overkilled

_C.MODEL.P_VK.CLS_TOKEN_MASK_PIECES = True # new added for individual cls token mask (made pieces)
_C.MODEL.P_VK.CLS_TOKEN_PIECE_MASK_PERCENT_NUM = None # set specific num of percent to mask
_C.MODEL.P_VK.CLS_TOKEN_PIECE_MASK_PERCENT = [10, 20, 30, 40, 50, 60, 70, 80, 90] # percentage applied during selected
_C.MODEL.P_VK.MIN_NUMBER_CLS_TOKEN_PIECE = 4 # set the lower boundary to avoid overkilled

_C.MODEL.P_VK.CLS_TOKEN_P_PIECES_NUM = 16 # new added to devided the pieces of token(for cls_token temporarily) 16
_C.MODEL.P_VK.MASK_RESERVE = False # reserve the order of mask or not.

_C.MODEL.P_VK.REWIND_MASK_CLS_TOKEN_NUM = -1 # change correpsondingly during train
_C.MODEL.P_VK.REWIND_MASK_CLS_TOKEN_PIECE_NUM = -1 # change correpsondingly during train
_C.MODEL.P_VK.REWIND_STATUS = False # status mark for rewind process
_C.MODEL.P_VK.REWIND_OUTPUT_DIR = ""
_C.MODEL.P_VK.SAVE_REWIND_MODEL = False 

# Based on MASK_CLS_TOKEN == True
_C.MODEL.P_VK.MASK_CLS_TOKEN_ON_VK = False # mask value and key instead of cls_token (unfinished, does not make sense)

# ----------------------------------------------------------------------
# Dataset options
# ----------------------------------------------------------------------
_C.DATA = CfgNode()

_C.DATA.NAME = ""
_C.DATA.DATAPATH = "" # (changed here!)
_C.DATA.FEATURE = ""  # e.g. inat2021_supervised

_C.DATA.PERCENTAGE = 1.0
_C.DATA.NUMBER_CLASSES = -1
_C.DATA.MULTILABEL = False
_C.DATA.CLASS_WEIGHTS_TYPE = "none"

_C.DATA.CROPSIZE = 224  # or 384

_C.DATA.NO_TEST = False
_C.DATA.BATCH_SIZE = 32
# Number of data loader workers per training process
_C.DATA.NUM_WORKERS = 4
# Load data to pinned host memory
_C.DATA.PIN_MEMORY = True

_C.DIST_BACKEND = "nccl"
_C.DIST_INIT_PATH = "env://" 
_C.DIST_INIT_FILE = ""


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
