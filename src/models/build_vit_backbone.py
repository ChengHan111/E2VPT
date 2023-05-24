#!/usr/bin/env python3
import numpy as np
import torch
import os
from .vit_backbones.swin_transformer import SwinTransformer
from .vit_backbones.vit import VisionTransformer
from .vit_backbones.vit_changeVK import VisionTransformer_changedVK
from .vit_backbones.vit_moco import vit_base
from .vit_backbones.vit_mae import build_model as mae_vit_model

from .vit_prompt.vit import PromptedVisionTransformer
from .vit_prompt.vit_prompt_VK import PromptedVisionTransformer_Prompt_VK
from .vit_prompt.vit_exp_self import PromptedVisionTransformer_EXPSELF
from .vit_prompt.swin_transformer import PromptedSwinTransformer
from .vit_prompt.swin_transformer_prompt_VK import PromptedSwinTransformer_Prompt_VK
from .vit_prompt.vit_moco import vit_base as prompt_vit_base
from .vit_prompt.vit_mae import build_model as prompt_mae_vit_model
from .vit_prompt.vit_mae_prompt_VK import build_model as prompt_VK_mae_vit_model # new added
from .vit_prompt.vit_moco_prompt_VK import vit_base as prompt_VK_vit_base # new added

# from .vit_prompt.vit_ablations import PromptedAblationVisionTransformer 通过这行可以加入vpt的ablations
# 先暂时不加 先弄一个vpt的原始版本来进行测试

from .vit_adapter.vit_mae import build_model as adapter_mae_vit_model
from .vit_adapter.vit_moco import vit_base as adapter_vit_base

from .vit_adapter.vit import ADPT_VisionTransformer
MODEL_ZOO = {
    "swint_imagenet": "swin_tiny_patch4_window7_224.pth",
    "swint_imagenet_ssl": "moby_swin_t_300ep_pretrained.pth",
    "swins_imagenet": "swin_small_patch4_window7_224.pth",
    "swinb_imagenet_224": "swin_base_patch4_window7_224.pth",
    "swinb_imagenet_384": "swin_base_patch4_window12_384.pth",
    "swinb_imagenet22k_224":  "swin_base_patch4_window7_224_22k.pth",
    "swinb_imagenet22k_384": "swin_base_patch4_window12_384_22k.pth",
    "swinl_imagenet22k_224": "swin_large_patch4_window7_224_22k.pth",
    "sup_vitb8": "ViT-B_8.npz",
    "sup_vitb16_224": "ViT-B_16-224.npz",
    # enable the below line for two step training
    # "sup_vitb16_224": 'specific model path here',
    "sup_vitb16": "ViT-B_16.npz",
    "sup_vitl16_224": "ViT-L_16-224.npz",
    "sup_vitl16": "ViT-L_16.npz",
    "sup_vitb8_imagenet21k": "imagenet21k_ViT-B_8.npz",
    "sup_vitb32_imagenet21k": "imagenet21k_ViT-B_32.npz",
    "sup_vitb16_imagenet21k": "imagenet21k_ViT-B_16.npz",
    "sup_vitl16_imagenet21k": "imagenet21k_ViT-L_16.npz",
    "sup_vitl32_imagenet21k": "imagenet21k_ViT-L_32.npz",
    "sup_vith14_imagenet21k": "imagenet21k_ViT-H_14.npz",
    "mae_vith14": "mae_pretrain_vit_huge.pth",
    "mae_vitb16": "mae_pretrain_vit_base.pth",
    "mae_vitl16": "mae_pretrain_vit_large.pth",
}


def build_mae_model(
    model_type, crop_size, prompt_cfg, p_vk_cfg, model_root, adapter_cfg=None
): # new added p_vk_cfg
    if prompt_cfg is not None:
        model = prompt_mae_vit_model(model_type, prompt_cfg)
    elif adapter_cfg is not None:
        model = adapter_mae_vit_model(model_type, adapter_cfg)
    elif p_vk_cfg is not None:
        print('3. Go through P_VK_cfg --- build_mae_model.py') 
        # print('should be None (True)', prompt_cfg)
        model = prompt_VK_mae_vit_model(model_type, p_vk_cfg)
    else:
        model = mae_vit_model(model_type)
    out_dim = model.embed_dim

    ckpt = os.path.join(model_root, MODEL_ZOO[model_type])

    checkpoint = torch.load(ckpt, map_location="cpu")
    state_dict = checkpoint['model']

    model.load_state_dict(state_dict, strict=False)
    model.head = torch.nn.Identity()
    return model, out_dim


def build_mocov3_model(
    model_type, crop_size, prompt_cfg, p_vk_cfg, model_root, adapter_cfg=None
):
    if model_type != "mocov3_vitb":
        raise ValueError("Does not support other arch")
    if prompt_cfg is not None:
        model = prompt_vit_base(prompt_cfg)
    elif adapter_cfg is not None:
        model = adapter_vit_base(adapter_cfg)
    elif p_vk_cfg is not None:
        print('3. Go through P_VK_cfg --- build_mocov3_model.py') 
        # print('should be None (True)', prompt_cfg)
        model = prompt_VK_vit_base(p_vk_cfg)
    else:
        model = vit_base()
    out_dim = 768
    # print('load_ckpt mocov3_linear-vit-b-300ep.pth.tar')
    ckpt = os.path.join(model_root,"mocov3_linear-vit-b-300ep.pth.tar")

    checkpoint = torch.load(ckpt, map_location="cpu")
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.'):
            # remove prefix
            state_dict[k[len("module."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    model.head = torch.nn.Identity()
    return model, out_dim


def build_swin_model(model_type, crop_size, prompt_cfg, p_vk_cfg, model_root):
    if prompt_cfg is not None:
        return _build_prompted_swin_model(
            model_type, crop_size, prompt_cfg, p_vk_cfg, model_root)
    elif p_vk_cfg is not None:
        return _build_promptedVK_swin_model(
            model_type, crop_size, prompt_cfg, p_vk_cfg, model_root)
    else:
        return _build_swin_model(model_type, crop_size, model_root)

def _build_promptedVK_swin_model(model_type, crop_size, prompt_cfg, p_vk_cfg, model_root):
    if model_type == "swint_imagenet":
        model = PromptedSwinTransformer_Prompt_VK(
            p_vk_cfg,
            img_size=crop_size,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.2,
            num_classes=-1,
        )
        embed_dim = 96
        num_layers = 4
    elif model_type == "swint_imagenet_ssl":
        model = PromptedSwinTransformer_Prompt_VK(
            p_vk_cfg,
            img_size=crop_size,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.2,
            num_classes=-1,
        )
        embed_dim = 96
        num_layers = 4

    elif model_type == "swins_imagenet":
        model = PromptedSwinTransformer_Prompt_VK(
            p_vk_cfg,
            img_size=crop_size,
            embed_dim=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.3,
            num_classes=-1,
        )
        embed_dim = 96
        num_layers = 4
    elif model_type == "swinb_imagenet_224":
        model = PromptedSwinTransformer_Prompt_VK(
            p_vk_cfg,
            img_size=crop_size,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 128
        num_layers = 4
    elif model_type == "swinb_imagenet_384":
        model = PromptedSwinTransformer_Prompt_VK(
            p_vk_cfg,
            img_size=384,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=12,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 128
        num_layers = 4

    elif model_type == "swinb_imagenet22k_224":
        model = PromptedSwinTransformer_Prompt_VK(
            p_vk_cfg,
            img_size=crop_size,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 128
        num_layers = 4
    elif model_type == "swinb_imagenet22k_384":
        model = PromptedSwinTransformer_Prompt_VK(
            p_vk_cfg,
            img_size=384,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=12,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 128
        num_layers = 4
    elif model_type == "swinl_imagenet22k_224":
        model = PromptedSwinTransformer_Prompt_VK(
            p_vk_cfg,
            img_size=crop_size,
            embed_dim=192,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=7,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 192
        num_layers = 4

    feat_dim = int(embed_dim * 2 ** (num_layers - 1))
    # load checkpoint
    model_w = os.path.join(model_root, MODEL_ZOO[model_type])
    checkpoint = torch.load(model_w, map_location='cpu')
    state_dict = checkpoint['model']

    if crop_size == 448:
        for k in list(state_dict.keys()):
            if "attn_mask" not in k:
                # remove prefix
                state_dict[k] = state_dict[k]
            # delete renamed or unused k
            else:
                del state_dict[k]

    # rename some keys for ssl models
    if model_type.endswith("ssl"):
        # rename moco pre-trained keys
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('encoder.'):
                # remove prefix
                state_dict[k[len("encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

    model.load_state_dict(state_dict, strict=False)

    return model, feat_dim



def _build_prompted_swin_model(model_type, crop_size, prompt_cfg, p_vk_cfg, model_root):
    if model_type == "swint_imagenet":
        model = PromptedSwinTransformer(
            prompt_cfg,
            img_size=crop_size,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.2,
            num_classes=-1,
        )
        embed_dim = 96
        num_layers = 4
    elif model_type == "swint_imagenet_ssl":
        model = PromptedSwinTransformer(
            prompt_cfg,
            img_size=crop_size,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.2,
            num_classes=-1,
        )
        embed_dim = 96
        num_layers = 4

    elif model_type == "swins_imagenet":
        model = PromptedSwinTransformer(
            prompt_cfg,
            img_size=crop_size,
            embed_dim=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.3,
            num_classes=-1,
        )
        embed_dim = 96
        num_layers = 4
    elif model_type == "swinb_imagenet_224":
        model = PromptedSwinTransformer(
            prompt_cfg,
            img_size=crop_size,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 128
        num_layers = 4
    elif model_type == "swinb_imagenet_384":
        model = PromptedSwinTransformer(
            prompt_cfg,
            img_size=384,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=12,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 128
        num_layers = 4

    elif model_type == "swinb_imagenet22k_224":
        model = PromptedSwinTransformer(
            prompt_cfg,
            img_size=crop_size,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 128
        num_layers = 4
    elif model_type == "swinb_imagenet22k_384":
        model = PromptedSwinTransformer(
            prompt_cfg,
            img_size=384,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=12,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 128
        num_layers = 4
    elif model_type == "swinl_imagenet22k_224":
        model = PromptedSwinTransformer(
            prompt_cfg,
            img_size=crop_size,
            embed_dim=192,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=7,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 192
        num_layers = 4

    feat_dim = int(embed_dim * 2 ** (num_layers - 1))
    # load checkpoint
    model_w = os.path.join(model_root, MODEL_ZOO[model_type])
    checkpoint = torch.load(model_w, map_location='cpu')
    state_dict = checkpoint['model']

    if crop_size == 448:
        for k in list(state_dict.keys()):
            if "attn_mask" not in k:
                # remove prefix
                state_dict[k] = state_dict[k]
            # delete renamed or unused k
            else:
                del state_dict[k]

    # rename some keys for ssl models
    if model_type.endswith("ssl"):
        # rename moco pre-trained keys
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('encoder.'):
                # remove prefix
                state_dict[k[len("encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

    model.load_state_dict(state_dict, strict=False)

    return model, feat_dim


def _build_swin_model(model_type, crop_size, model_root):
    if model_type == "swint_imagenet":
        model = SwinTransformer(
            img_size=crop_size,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.2,
            num_classes=-1,  # setting to a negative value will make head as identity
        )
        embed_dim = 96
        num_layers = 4
    elif model_type == "swint_imagenet_ssl":
        model = SwinTransformer(
            img_size=crop_size,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.2,
            num_classes=-1,
        )
        embed_dim = 96
        num_layers = 4

    elif model_type == "swins_imagenet":
        model = SwinTransformer(
            img_size=crop_size,
            embed_dim=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.3,
            num_classes=-1,
        )
        embed_dim = 96
        num_layers = 4
    elif model_type == "swinb_imagenet_224":
        model = SwinTransformer(
            img_size=crop_size,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 128
        num_layers = 4
    elif model_type == "swinb_imagenet_384":
        model = SwinTransformer(
            img_size=384,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=12,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 128
        num_layers = 4

    elif model_type == "swinb_imagenet22k_224":
        model = SwinTransformer(
            img_size=crop_size,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 128
        num_layers = 4
    elif model_type == "swinb_imagenet22k_384":
        model = SwinTransformer(
            img_size=384,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=12,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 128
        num_layers = 4
    elif model_type == "swinl_imagenet22k_224":
        model = SwinTransformer(
            img_size=crop_size,
            embed_dim=192,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=7,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 192
        num_layers = 4

    feat_dim = int(embed_dim * 2 ** (num_layers - 1))
    # load checkpoint
    model_w = os.path.join(model_root, MODEL_ZOO[model_type])
    checkpoint = torch.load(model_w, map_location='cpu')
    state_dict = checkpoint['model']

    if crop_size == 448:
        for k in list(state_dict.keys()):
            if "attn_mask" not in k:
                # remove prefix
                state_dict[k] = state_dict[k]
            # delete renamed or unused k
            else:
                del state_dict[k]

    # rename some keys for ssl models
    if model_type.endswith("ssl"):
        # rename moco pre-trained keys
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('encoder.'):
                # remove prefix
                state_dict[k[len("encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

    model.load_state_dict(state_dict, strict=False)

    return model, feat_dim


def build_vit_sup_models(
    model_type, crop_size, prompt_cfg=None, model_root=None, adapter_cfg=None, load_pretrain=True, vis=False, qkv_cfg=None, p_vk_cfg=None
):
    # image size is the size of actual image
    m2featdim = {
        "sup_vitb16_224": 768,
        "sup_vitb16": 768,
        "sup_vitl16_224": 1024,
        "sup_vitl16": 1024,
        "sup_vitb8_imagenet21k": 768,
        "sup_vitb16_imagenet21k": 768,
        "sup_vitb32_imagenet21k": 768,
        "sup_vitl16_imagenet21k": 1024,
        "sup_vitl32_imagenet21k": 1024,
        "sup_vith14_imagenet21k": 1280,
    } 
    # PromptedVisionTransformer 下面这个函数原本是这个 替换成了自己的
    # self changes here
    if prompt_cfg is not None:
        # For further changes in [cls] token
        print('3. Go through Origin Prompt --- build_vit_backbone.py')
        model = PromptedVisionTransformer_EXPSELF(
            prompt_cfg, model_type,
            crop_size, num_classes=-1, vis=vis
        )
        
    elif adapter_cfg is not None:
        model = ADPT_VisionTransformer(model_type, crop_size, num_classes=-1, adapter_cfg=adapter_cfg)

    elif qkv_cfg is not None:
        print('3. Go through QKV --- build_vit_backbone.py') # 走到这里了!!
        model = VisionTransformer_changedVK(
            qkv_cfg, model_type, crop_size, num_classes=-1, vis=vis)
    
    elif p_vk_cfg is not None:
        print('3. Go through P_VK --- build_vit_backbone.py') # 走到这里了!!
        model = PromptedVisionTransformer_Prompt_VK(
            p_vk_cfg, model_type, crop_size, num_classes=-1, vis=vis)
    
    else:
        # 似乎不用加 在这里如果prompt_cfg和adapter_cfg都是None的话，直接就走这里了 (for qkv)
        print('3. Apply originally VisionTransformer')
        model = VisionTransformer(
            model_type, crop_size, num_classes=-1, vis=vis)
    
    if load_pretrain:
        
        resume_model = False
        if resume_model:
            # load checkpoint
            Remove_head_during_finetune = True
            # make changes here! Note to restore the original code
            ckpt = MODEL_ZOO[model_type]
            checkpoint = torch.load(ckpt, map_location="cpu")
            state_dict = checkpoint['model']
            # create a dictionary to map the old keys to the new keys
            
            new_keys = {}
            for key in state_dict.keys():
                new_key = key.replace("enc.transformer", "transformer")
                new_keys[key] = new_key

            # create a new state dict with the updated keys
            new_state_dict = {new_keys[k]: v for k, v in state_dict.items()}
            new_state_dict = {k:v for k,v in new_state_dict.items() if not k.startswith('head')}
            
            model.load_state_dict(new_state_dict, strict=False)
            # model.load_state_dict(state_dict, strict=True)
        else:
            # This is the origin version of loading the model
            model.load_from(np.load(os.path.join(model_root, MODEL_ZOO[model_type])))

    return model, m2featdim[model_type]

