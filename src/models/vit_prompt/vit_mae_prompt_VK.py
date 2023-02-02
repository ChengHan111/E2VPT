#!/usr/bin/env python3
"""
vit-moco-v3 with prompt
"""
import math
import torch
import torch.nn as nn
import torchvision as tv

from functools import partial, reduce
from operator import mul
from torch.nn import Conv2d, Dropout
from timm.models.vision_transformer import _cfg

# from ..vit_backbones.vit_mae import VisionTransformer
import os
import json
from ..vit_backbones.vit_mae_changeVK import VisionTransformer
from ...utils import logging
logger = logging.get_logger("visual_prompt")

#TODO add mae for the title
class PromptedVisionTransformer_Prompt_VK(VisionTransformer):
    def __init__(self, p_vk_cfg, **kwargs):
        super().__init__(p_vk_cfg, **kwargs)
        self.p_vk_cfg = p_vk_cfg
        if self.p_vk_cfg.DEEP_P and self.p_vk_cfg.LOCATION not in ["prepend", ]:
            raise ValueError("Deep-{} is not supported".format(self.p_vk_cfg.LOCATION))

        num_tokens_P = self.p_vk_cfg.NUM_TOKENS_P

        self.num_tokens_P = num_tokens_P
        self.prompt_dropout = Dropout(self.p_vk_cfg.DROPOUT)

        # initiate prompt:
        if self.p_vk_cfg.INITIATION == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))  # noqa

            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, num_tokens_P, self.embed_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if self.p_vk_cfg.DEEP_P:
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    len(self.blocks) - 1,
                    num_tokens_P, self.embed_dim
                ))
                # xavier_uniform initialization
                nn.init.uniform_(
                    self.deep_prompt_embeddings.data, -val, val)

        else:
            raise ValueError("Other initiation scheme is not supported")
        
        # print('self.embed_dim', self.embed_dim)
        # new added for cls_token masked
        if self.p_vk_cfg.MASK_CLS_TOKEN is True:
            if self.p_vk_cfg.CLS_TOKEN_MASK == True:
                self.prompt_soft_tokens_mask_cls_token = nn.Parameter(torch.ones(self.num_tokens_P), requires_grad=True)
            
            if self.p_vk_cfg.CLS_TOKEN_MASK_PIECES == True:
                self.prompt_soft_tokens_pieces_mask_cls_token = nn.Parameter(torch.ones(self.num_tokens_P, self.p_vk_cfg.CLS_TOKEN_P_PIECES_NUM), requires_grad=True)
                self.soft_token_chunks_num_cls_token = int(self.embed_dim/self.p_vk_cfg.CLS_TOKEN_P_PIECES_NUM)

            # Rewind status mark here.
            if self.p_vk_cfg.MASK_CLS_TOKEN and self.p_vk_cfg.REWIND_STATUS:
                
                soft_token_mask_dir = os.path.join(self.p_vk_cfg.REWIND_OUTPUT_DIR, 'mask_tokens')
                assert soft_token_mask_dir is not None

                soft_token_mask_file = os.path.join(soft_token_mask_dir, "{}_soft_tokens_to_mask.json".format(self.p_vk_cfg.REWIND_MASK_CLS_TOKEN_NUM))
                soft_token_to_mask = self.load_soft_token_mask_file(soft_token_mask_file) 
                self.mask_soft_tokens(soft_token_to_mask)
            
            if self.p_vk_cfg.CLS_TOKEN_MASK_PIECES and self.p_vk_cfg.REWIND_STATUS:
                soft_tokens_pieces_mask_dir = os.path.join(self.p_vk_cfg.REWIND_OUTPUT_DIR, 'mask_tokens_pieces')
                soft_tokens_pieces_mask_file = os.path.join(soft_tokens_pieces_mask_dir, "{}_soft_tokens_pieces_to_mask.json".format(self.p_vk_cfg.REWIND_MASK_CLS_TOKEN_PIECE_NUM)) # rewind_mask_token_pieces_number
                soft_tokens_pieces_to_mask = self.load_soft_tokens_pieces_mask_file(soft_tokens_pieces_mask_file)  
                self.mask_soft_tokens_pieces(soft_tokens_pieces_to_mask)
        
        # add drop-out or not
        self.prompt_dropout = Dropout(self.p_vk_cfg.DROPOUT_P)
    
    def load_soft_token_mask_file(self, path):
        with open(path) as f:
            t = json.load(f)
        
        soft_token_to_mask = set()
        for mask_number, soft_token in t.items():
            for soft_token_i in soft_token:
                soft_token_to_mask.add(soft_token_i) 
        
        return soft_token_to_mask

    def load_soft_tokens_pieces_mask_file(self, path):
        with open(path) as f:
            t = json.load(f)
        soft_tokens_pieces_to_mask = {}
        for soft_token_idx, soft_token_pieces in t.items():
            soft_tokens_pieces_to_mask[int(soft_token_idx)] = set(soft_token_pieces)
        return soft_tokens_pieces_to_mask

    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
        
        if self.p_vk_cfg.LOCATION == "prepend":
            # after CLS token, all before image patches
            
            prompt_embeddings = self.prompt_dropout(self.prompt_embeddings.expand(B, -1, -1))
            if self.p_vk_cfg.MASK_CLS_TOKEN is True: 
                if self.p_vk_cfg.CLS_TOKEN_MASK_PIECES is True:
                    # print('222', self.soft_tokens_pieces_mask_cls_token.shape) torch.Size([32, 16])
                    prompt_embeddings = prompt_embeddings * self.prompt_soft_tokens_pieces_mask_cls_token.repeat((1,self.soft_token_chunks_num_cls_token)).repeat(B, 1, 1)
                    # print('mark1', self.prompt_soft_tokens_pieces_mask_cls_token)
                    # print('prompt_embeddings', prompt_embeddings)
                if self.p_vk_cfg.CLS_TOKEN_MASK == True:
                    prompt_embeddings = prompt_embeddings * self.prompt_soft_tokens_mask_cls_token.view(-1, 1).repeat(1, prompt_embeddings.shape[2]).repeat(B, 1, 1)
                    # print('mark2', self.prompt_soft_tokens_mask_cls_token)
                    # print('prompt_embeddings', prompt_embeddings)
            
            x = torch.cat((
                    x[:, :1, :],
                    prompt_embeddings,
                    x[:, 1:, :]
                ), dim=1)
            # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)

        else:
            raise ValueError("Other prompt locations are not supported")
        return x

    def embeddings(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        return x

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.blocks.eval()
            self.patch_embed.eval()
            self.pos_drop.eval()
            self.prompt_dropout.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward_features(self, x):
        x = self.incorporate_prompt(x)

        if self.p_vk_cfg.DEEP_P:
            B = x.shape[0]
            num_layers = len(self.blocks)

            for i in range(num_layers):
                if i == 0:
                    x = self.blocks[i](x)
                else:
                    # prepend
                    deep_prompt_emb = self.prompt_dropout(self.deep_prompt_embeddings[i-1].expand(B, -1, -1))
                    
                    # 层数是通过每一层这样加进去体现的
                    if self.p_vk_cfg.MASK_CLS_TOKEN is True: 
                        if self.p_vk_cfg.CLS_TOKEN_MASK_PIECES is True:
                            # print(self.soft_tokens_pieces_mask_cls_token.repeat((1,self.soft_token_chunks_num_cls_token)).repeat(B, 1, 1).shape)
                            deep_prompt_emb = deep_prompt_emb * self.prompt_soft_tokens_pieces_mask_cls_token.repeat((1,self.soft_token_chunks_num_cls_token)).repeat(B, 1, 1)
                        if self.p_vk_cfg.CLS_TOKEN_MASK == True:
                            # print(self.soft_tokens_mask_cls_token.view(-1, 1).repeat(1, self.deep_prompt_embeddings.shape[2]).repeat(B, 1, 1))
                            deep_prompt_emb = deep_prompt_emb * self.prompt_soft_tokens_mask_cls_token.view(-1, 1).repeat(1, self.deep_prompt_embeddings.shape[2]).repeat(B, 1, 1)
                    
                    x = torch.cat((
                        x[:, :1, :],
                        deep_prompt_emb,
                        x[:, (1 + self.num_tokens_P):, :]
                    ), dim=1)
                    x = self.blocks[i](x)
        else:
            for blk in self.blocks:
                x = blk(x)

        if self.p_vk_cfg.VIT_POOL_TYPE == "imgprompt_pool":
            assert self.p_vk_cfg.LOCATION == "prepend"
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        elif self.p_vk_cfg.VIT_POOL_TYPE == "original":
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        elif self.p_vk_cfg.VIT_POOL_TYPE == "img_pool":
            assert self.p_vk_cfg.LOCATION == "prepend"
            x = x[:, self.num_tokens_P+1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        elif self.p_vk_cfg.VIT_POOL_TYPE == "prompt_pool":
            assert self.p_vk_cfg.LOCATION == "prepend"
            x = x[:, 1:self.num_tokens_P+1, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            raise ValueError("pooling type for output is not supported")

        return outcome


def build_model(model_type, p_vk_cfg):
    if "vitb" in model_type:
        return vit_base_patch16(p_vk_cfg)
    elif "vitl" in model_type:
        return vit_large_patch16(p_vk_cfg)
    elif "vith" in model_type:
        return vit_huge_patch14(p_vk_cfg)


def vit_base_patch16(p_vk_cfg, **kwargs):
    model = PromptedVisionTransformer_Prompt_VK(
        p_vk_cfg,
        drop_path_rate=0.1, global_pool=True,  # using default settings for mae-finetune
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(p_vk_cfg, **kwargs):
    model = PromptedVisionTransformer_Prompt_VK(
        p_vk_cfg,
        drop_path_rate=0.1, global_pool=True,  # using default settings for mae-finetune
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(p_vk_cfg, **kwargs):
    model = PromptedVisionTransformer_Prompt_VK(
        p_vk_cfg,
        drop_path_rate=0.1, global_pool=True,  # using default settings for mae-finetune
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


