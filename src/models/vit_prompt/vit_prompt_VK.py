#!/usr/bin/env python3
"""
vit with prompt: a clean version with the default settings of VPT
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import os
import json

from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d, Dropout
from scipy import ndimage

from ..vit_backbones.vit import CONFIGS, np2th
from ..vit_backbones.vit_changeVK import Transformer_changeVK, VisionTransformer_changedVK
from ...utils import logging

logger = logging.get_logger("visual_prompt")


class PromptedTransformer_Prompt_VK(Transformer_changeVK):
    def __init__(self, p_vk_config, config, img_size, vis):
        assert p_vk_config.LOCATION == "prepend"
        assert p_vk_config.INITIATION == "random"
        assert p_vk_config.NUM_DEEP_LAYERS is None
        assert not p_vk_config.DEEP_SHARED
        super(PromptedTransformer_Prompt_VK, self).__init__(
            p_vk_config, config, img_size, vis)
        
        self.p_vk_config = p_vk_config
        self.vit_config = config

        img_size = _pair(img_size)
        patch_size = _pair(config.patches["size"])

        num_tokens_P = self.p_vk_config.NUM_TOKENS_P
        self.num_tokens_P = num_tokens_P  # number of prompted tokens (promot 数量)
        
        # new added for cls_token masked
        if self.p_vk_config.MASK_CLS_TOKEN is True:
            if self.p_vk_config.CLS_TOKEN_MASK == True:
                self.prompt_soft_tokens_mask_cls_token = nn.Parameter(torch.ones(self.num_tokens_P), requires_grad=True)
            
            if self.p_vk_config.CLS_TOKEN_MASK_PIECES == True:
                self.prompt_soft_tokens_pieces_mask_cls_token = nn.Parameter(torch.ones(self.num_tokens_P, self.p_vk_config.CLS_TOKEN_P_PIECES_NUM), requires_grad=True)
                self.soft_token_chunks_num_cls_token = int(config.hidden_size/self.p_vk_config.CLS_TOKEN_P_PIECES_NUM)

            # Rewind status mark here.
            if self.p_vk_config.MASK_CLS_TOKEN and self.p_vk_config.REWIND_STATUS:
                
                soft_token_mask_dir = os.path.join(self.p_vk_config.REWIND_OUTPUT_DIR, 'mask_tokens')
                assert soft_token_mask_dir is not None

                soft_token_mask_file = os.path.join(soft_token_mask_dir, "{}_soft_tokens_to_mask.json".format(self.p_vk_config.REWIND_MASK_CLS_TOKEN_NUM))
                soft_token_to_mask = self.load_soft_token_mask_file(soft_token_mask_file) 
                self.mask_soft_tokens(soft_token_to_mask)
            
            if self.p_vk_config.CLS_TOKEN_MASK_PIECES and self.p_vk_config.REWIND_STATUS:
                soft_tokens_pieces_mask_dir = os.path.join(self.p_vk_config.REWIND_OUTPUT_DIR, 'mask_tokens_pieces')
                soft_tokens_pieces_mask_file = os.path.join(soft_tokens_pieces_mask_dir, "{}_soft_tokens_pieces_to_mask.json".format(self.p_vk_config.REWIND_MASK_CLS_TOKEN_PIECE_NUM)) # rewind_mask_token_pieces_number
                soft_tokens_pieces_to_mask = self.load_soft_tokens_pieces_mask_file(soft_tokens_pieces_mask_file)  
                self.mask_soft_tokens_pieces(soft_tokens_pieces_to_mask)
        
        # add drop-out or not
        self.prompt_dropout = Dropout(self.p_vk_config.DROPOUT_P)

        # if project the prompt embeddings
        if self.p_vk_config.PROJECT > -1:
            # only for prepend / add
            prompt_dim = self.p_vk_config.PROJECT
            self.prompt_proj = nn.Linear(
                prompt_dim, config.hidden_size)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = config.hidden_size
            # print('prompt_dim', prompt_dim) 768
            self.prompt_proj = nn.Identity()

        # initiate prompt:
        if self.p_vk_config.INITIATION == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, num_tokens_P, prompt_dim))
            # print('self.prompt_embeddings.shape', self.prompt_embeddings.shape) # torch.Size([1, 10, 768])
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if self.p_vk_config.DEEP_P:  # noqa

                total_d_layer = config.transformer["num_layers"]-1
                # 初始化是一起生成的
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    total_d_layer, num_tokens_P, prompt_dim))
                # print('self.deep_prompt_embeddings.shape', self.deep_prompt_embeddings.shape) # torch.Size([11, 10, 768])
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

        else:
            raise ValueError("Other initiation scheme is not supported")
        
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
        # print('B', B) # 128
        # after CLS token, all before image patches
        x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
        
        prompt_embeddings = self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1))
        
        if self.p_vk_config.MASK_CLS_TOKEN is True: 
            if self.p_vk_config.CLS_TOKEN_MASK_PIECES is True:
                # print('222', self.soft_tokens_pieces_mask_cls_token.shape) torch.Size([32, 16])
                prompt_embeddings = prompt_embeddings * self.prompt_soft_tokens_pieces_mask_cls_token.repeat((1,self.soft_token_chunks_num_cls_token)).repeat(B, 1, 1)
                # print('mark1', self.prompt_soft_tokens_pieces_mask_cls_token)
                # print('prompt_embeddings', prompt_embeddings)
            if self.p_vk_config.CLS_TOKEN_MASK == True:
                prompt_embeddings = prompt_embeddings * self.prompt_soft_tokens_mask_cls_token.view(-1, 1).repeat(1, prompt_embeddings.shape[2]).repeat(B, 1, 1)
                # print('mark2', self.prompt_soft_tokens_mask_cls_token)
                # print('prompt_embeddings', prompt_embeddings)
        
        x = torch.cat((
                x[:, :1, :],
                prompt_embeddings,
                x[:, 1:, :]
            ), dim=1)
        # print('11111', self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1).shape) #torch.Size([128, 10, 768])
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)

        return x

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.encoder.eval()
            self.embeddings.eval()
            self.prompt_proj.train()
            self.prompt_dropout.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward_deep_prompt(self, embedding_output):
        attn_weights = []
        hidden_states = None
        weights = None
        B = embedding_output.shape[0]
        num_layers = self.vit_config.transformer["num_layers"]

        for i in range(num_layers):

            if i == 0:
                hidden_states, weights = self.encoder.layer[i](embedding_output)
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i-1]).expand(B, -1, -1))
                    
                    # 层数是通过每一层这样加进去体现的
                    if self.p_vk_config.MASK_CLS_TOKEN is True: 
                        if self.p_vk_config.CLS_TOKEN_MASK_PIECES is True:
                            # print(self.soft_tokens_pieces_mask_cls_token.repeat((1,self.soft_token_chunks_num_cls_token)).repeat(B, 1, 1).shape)
                            deep_prompt_emb = deep_prompt_emb * self.prompt_soft_tokens_pieces_mask_cls_token.repeat((1,self.soft_token_chunks_num_cls_token)).repeat(B, 1, 1)
                        if self.p_vk_config.CLS_TOKEN_MASK == True:
                            # print(self.soft_tokens_mask_cls_token.view(-1, 1).repeat(1, self.deep_prompt_embeddings.shape[2]).repeat(B, 1, 1))
                            deep_prompt_emb = deep_prompt_emb * self.prompt_soft_tokens_mask_cls_token.view(-1, 1).repeat(1, self.deep_prompt_embeddings.shape[2]).repeat(B, 1, 1)
                    
                    hidden_states = torch.cat((
                        hidden_states[:, :1, :],
                        deep_prompt_emb,
                        hidden_states[:, (1+self.num_tokens_P):, :]
                    ), dim=1)


                hidden_states, weights = self.encoder.layer[i](hidden_states)

            
            if self.encoder.vis:
                attn_weights.append(weights)

        encoded = self.encoder.encoder_norm(hidden_states)
        return encoded, attn_weights

    def forward(self, x):
        # this is the default version: 这样的写法是 第一层是单独加进去的 后面层是有deep再加入的
        embedding_output = self.incorporate_prompt(x)

        if self.p_vk_config.DEEP_P:
            encoded, attn_weights = self.forward_deep_prompt(
                embedding_output)
        else:
            encoded, attn_weights = self.encoder(embedding_output)

        return encoded, attn_weights
    
    def mask_soft_tokens(self, soft_tokens_to_mask):
        self.soft_tokens_to_mask = list(soft_tokens_to_mask)
        for soft_token_idx in self.soft_tokens_to_mask:
            # print('soft_token_idx',soft_token_idx)
            self.prompt_soft_tokens_mask_cls_token.data[soft_token_idx] = 0
        # Self added no grad during rewind
        self.prompt_soft_tokens_mask_cls_token.requires_grad_(False)            
            
    def mask_soft_tokens_pieces(self, soft_tokens_pieces_to_mask):
        for soft_token_id, soft_token_pieces in soft_tokens_pieces_to_mask.items():
            for soft_token_piece in soft_token_pieces:
                self.prompt_soft_tokens_pieces_mask_cls_token.data[soft_token_id][soft_token_piece] = 0
        # Self added no grad during rewind
        self.prompt_soft_tokens_pieces_mask_cls_token.requires_grad_(False) 

class PromptedVisionTransformer_Prompt_VK(VisionTransformer_changedVK):
    def __init__(
        self, p_vk_cfg, model_type,
        img_size=224, num_classes=21843, vis=False
    ):
        assert p_vk_cfg.VIT_POOL_TYPE == "original"
        super(PromptedVisionTransformer_Prompt_VK, self).__init__(
            p_vk_cfg, model_type, img_size, num_classes, vis)
        if p_vk_cfg is None:
            raise ValueError("p_vk_cfg cannot be None if using PromptedVisionTransformer_Prompt_VK")
        self.p_vk_cfg = p_vk_cfg
        vit_cfg = CONFIGS[model_type]
        self.transformer = PromptedTransformer_Prompt_VK(
            p_vk_cfg, vit_cfg, img_size, vis)

    def forward(self, x, vis=False):
        x, attn_weights = self.transformer(x)

        x = x[:, 0]

        logits = self.head(x)

        if not vis:
            return logits
        return logits, attn_weights
