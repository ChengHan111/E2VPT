#!/usr/bin/env python3
"""
vit with prompt: a clean version with the default settings of VPT
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv

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
        # self.soft_tokens_mask_cls_token = nn.Parameter(torch.ones(self.num_tokens_P), requires_grad=True)
        # self.soft_tokens_pieces_mask_cls_token = nn.Parameter(torch.ones(self.num_tokens_P, self.p_vk_config.CLS_TOKEN_P_PIECES_NUM), requires_grad=True)

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

    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        # print('B', B) # 128
        # after CLS token, all before image patches
        x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
        x = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
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
        # this is the default version:
        embedding_output = self.incorporate_prompt(x)

        if self.p_vk_config.DEEP_P:
            encoded, attn_weights = self.forward_deep_prompt(
                embedding_output)
        else:
            encoded, attn_weights = self.encoder(embedding_output)

        return encoded, attn_weights


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
