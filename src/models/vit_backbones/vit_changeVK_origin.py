#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
"""
models for vits, borrowed from
https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling_resnet.py
https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling.py
"""
import copy
import logging
import math

from os.path import join as pjoin
from turtle import forward

import torch
import torch.nn as nn
import numpy as np

from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from timm.models.layers import trunc_normal_

from ...configs import vit_configs as configs

from operator import mul
from functools import reduce
from torch.nn.modules.utils import _pair

from .vit import CONFIGS, np2th

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention_Origin_ViT(nn.Module):
    def __init__(self, config, vis):
        super(Attention_Origin_ViT, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states) # B, num_patches, head_size*num_head
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer) # B, num_head, num_patches, head_size
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer) # B, num_head, num_patches, head_size

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # B, num_head, num_patches, num_patches
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores) # B, num_head, num_patches(query), num_patches(key)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer) # B, num_head, num_patches, head_size
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Attention(nn.Module):
    def __init__(self, qkv_cfg, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads) # 768 / 12 (same as HyperPrompt discribe)
        self.all_head_size = self.num_attention_heads * self.attention_head_size # 12 *64
        
        # print('1', self.num_attention_heads) 12
        # print('2', self.attention_head_size) 64
        # print('3', self.all_head_size) 768
        self.config = config
        self.qkv_cfg = qkv_cfg

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)
        
        # self-added
        
        num_layers = self.config.transformer["num_layers"]
        num_tokens = self.qkv_cfg.NUM_TOKENS
        print('num_tokens', num_tokens)
        if self.qkv_cfg.NUM_TOKENS_P is not None:
            print('num_tokens_p', self.qkv_cfg.NUM_TOKENS_P)
        
        # add vk prompt layers separate
        if self.qkv_cfg.SHARE_PARAM_KV == False:
            head_fixed, num_patches_QKV_V, num_patches_QKV_K, head_size_fixed = self.num_attention_heads, num_tokens, num_tokens, self.attention_head_size

            self.deep_QKV_embeddings_V = nn.Parameter(torch.zeros(
                        head_fixed, num_patches_QKV_V, head_size_fixed))
            self.deep_QKV_embeddings_K = nn.Parameter(torch.zeros(
                        head_fixed, num_patches_QKV_K, head_size_fixed))
            
            if self.qkv_cfg.ORIGIN_INIT == 0:
                # xavier_uniform initialization
                patch_size = _pair(self.config.patches["size"]) # print('patch_size', patch_size) # 16, 16
                
                val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + 16))
                # val = math.sqrt(6. / float(3 * reduce(mul, query_layer.shape[0], 1) + 16)) # 现在是随便设置的， 需要后期改
                nn.init.uniform_(self.deep_QKV_embeddings_V.data, -val, val)
                nn.init.uniform_(self.deep_QKV_embeddings_K.data, -val, val)
            elif self.qkv_cfg.ORIGIN_INIT == 1:
                # apply timm trunc norm for init
                trunc_normal_(self.deep_QKV_embeddings_V, std=0.02)
                trunc_normal_(self.deep_QKV_embeddings_K, std=0.02)
                
            # kaiming init # untested(to be continued)
            else: 
                torch.nn.init.kaiming_uniform_(self.deep_QKV_embeddings_V, a=0, mode='fan_in', nonlinearity='leaky_relu')
                torch.nn.init.kaiming_uniform_(self.deep_QKV_embeddings_K, a=0, mode='fan_in', nonlinearity='leaky_relu')
                
            ''' (Under construction)
            if self.qkv_cfg.DEEP == False:
                # self.num_attention_heads set this to 1 for the first attention head.
                head_fixed, num_patches_QKV_V, num_patches_QKV_K, head_size_fixed = 1, num_tokens, num_tokens, self.attention_head_size

                self.deep_QKV_embeddings_V = nn.Parameter(torch.zeros(
                            head_fixed, num_patches_QKV_V, head_size_fixed))
                self.deep_QKV_embeddings_K = nn.Parameter(torch.zeros(
                            head_fixed, num_patches_QKV_K, head_size_fixed))
                # xavier_uniform initialization
                patch_size = _pair(self.config.patches["size"])
                # print('patch_size', patch_size) # 16, 16
                val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + 16))
                # val = math.sqrt(6. / float(3 * reduce(mul, query_layer.shape[0], 1) + 16)) # 现在是随便设置的， 需要后期改
                nn.init.uniform_(self.deep_QKV_embeddings_V.data, -val, val)
                nn.init.uniform_(self.deep_QKV_embeddings_K.data, -val, val)
            '''  
        else:
            head_fixed, num_patches_QKV, head_size_fixed = self.num_attention_heads, num_tokens, self.attention_head_size
            self.deep_QKV_embeddings = nn.Parameter(torch.zeros(
                        head_fixed, num_patches_QKV, head_size_fixed))
            if self.qkv_cfg.ORIGIN_INIT == 0:
                # xavier_uniform initialization
                patch_size = _pair(self.config.patches["size"])
                # print('patch_size', patch_size) # 16, 16
                val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + 16))
                # val = math.sqrt(6. / float(3 * reduce(mul, query_layer.shape[0], 1) + 16)) # 现在是随便设置的， 需要后期改
                nn.init.uniform_(self.deep_QKV_embeddings.data, -val, val)
            elif self.qkv_cfg.ORIGIN_INIT == 1:
                trunc_normal_(self.deep_QKV_embeddings, std=0.02)
            # kaiming init (to be continued, untested)
            
            else:
                torch.nn.init.kaiming_uniform_(self.deep_QKV_embeddings, a=0, mode='fan_in', nonlinearity='leaky_relu')

            
        
        
        self.QKV_proj = nn.Identity()
        # self.prompt_config.DROPOUT
        self.QKV_dropout = Dropout(self.qkv_cfg.DROPOUT) # should add config here
    


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        # print('mark', hidden_states)
        
        mixed_query_layer = self.query(hidden_states) # B, num_patches, head_size*num_head
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer) # B, num_head, num_patches, head_size
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer) # B, num_head, num_patches, head_size
        # head, sequence length (L) 应该加在这一维, dimension of each head
        # print('1', query_layer.shape) torch.Size([128, 12, 197, 64])
        # print('2', key_layer.shape) torch.Size([128, 12, 197, 64])
        # print('3', value_layer.shape) torch.Size([128, 12, 197, 64])
        
        """        
        # self-added
        B = query_layer.shape[0]
        num_layers = self.config.transformer["num_layers"]
        # print('num_layers(should be 12)', num_layers) True
        
        # add vk prompt layers
        head_fixed, num_patches_QKV_V, num_patches_QKV_K, head_size_fixed = query_layer.shape[1], 16, 16, query_layer.shape[-1]
        if torch.cuda.is_available():
            self.deep_QKV_embeddings_V = nn.Parameter(torch.zeros(
                        head_fixed, num_patches_QKV_V, head_size_fixed)).cuda()
            self.deep_QKV_embeddings_K = nn.Parameter(torch.zeros(
                        head_fixed, num_patches_QKV_K, head_size_fixed)).cuda()
        else:
            self.deep_QKV_embeddings_V = nn.Parameter(torch.zeros(
                        head_fixed, num_patches_QKV_V, head_size_fixed))
            self.deep_QKV_embeddings_K = nn.Parameter(torch.zeros(
                        head_fixed, num_patches_QKV_K, head_size_fixed))
            print('Warning: Not running on cuda! Reduce significantly in speed!')
        
        # ['forward1.0.weight1']
        # self.attention_norm
        
        # xavier_uniform initialization
        patch_size = _pair(self.config.patches["size"])
        # print('patch_size', patch_size) # 16, 16
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + 16))
        # val = math.sqrt(6. / float(3 * reduce(mul, query_layer.shape[0], 1) + 16)) # 现在是随便设置的， 需要后期改
        nn.init.uniform_(self.deep_QKV_embeddings_V.data, -val, val)
        nn.init.uniform_(self.deep_QKV_embeddings_K.data, -val, val)
        """
        
        # torch.Size([128, 12, 197, 64])
        # print('1', key_layer.shape) 
        # torch.Size([128, 12, 16, 64])
        # print('2', self.QKV_dropout(self.QKV_proj(self.deep_QKV_embeddings_V).expand(B, -1, -1, -1)).shape)
        # torch.Size([128, 12, 197, 64])
        # print('3', value_layer.shape)
        # torch.Size([128, 12, 16, 64])
        # print('4', self.QKV_dropout(self.QKV_proj(self.deep_QKV_embeddings_K).expand(B, -1, -1, -1)).shape)
        
        B = query_layer.shape[0]
        if self.qkv_cfg.SHARE_PARAM_KV == False:
            # B, num_head, num_patches, head_size
            if self.qkv_cfg.LAYER_BEHIND == False:
                key_layer = torch.cat((key_layer, self.QKV_dropout(self.QKV_proj(self.deep_QKV_embeddings_K).expand(B, -1, -1, -1))), dim=2)
                value_layer = torch.cat((value_layer, self.QKV_dropout(self.QKV_proj(self.deep_QKV_embeddings_V).expand(B, -1, -1, -1))), dim=2)
            else:
                key_layer = torch.cat((self.QKV_dropout(self.QKV_proj(self.deep_QKV_embeddings_K).expand(B, -1, -1, -1)), key_layer), dim=2)
                value_layer = torch.cat((self.QKV_dropout(self.QKV_proj(self.deep_QKV_embeddings_V).expand(B, -1, -1, -1)), value_layer), dim=2)
        else:
            if self.qkv_cfg.LAYER_BEHIND == False:
                key_layer = torch.cat((key_layer, self.QKV_dropout(self.QKV_proj(self.deep_QKV_embeddings).expand(B, -1, -1, -1))), dim=2)
                value_layer = torch.cat((value_layer, self.QKV_dropout(self.QKV_proj(self.deep_QKV_embeddings).expand(B, -1, -1, -1))), dim=2)
            else:
                key_layer = torch.cat((self.QKV_dropout(self.QKV_proj(self.deep_QKV_embeddings).expand(B, -1, -1, -1)), key_layer), dim=2)
                value_layer = torch.cat((self.QKV_dropout(self.QKV_proj(self.deep_QKV_embeddings).expand(B, -1, -1, -1)), value_layer), dim=2)
                
                
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # B, num_head, num_patches, num_patches (turn into patches*patches)                    
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # 接着做softmax了 和论文中的一致
        attention_probs = self.softmax(attention_scores) # B, num_head, num_patches(query), num_patches(key)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)
        
        # print('1', attention_probs.shape) # torch.Size([B, 12, 197, 197+(num_token)])
        # print('2', value_layer.shape) # torch.Size([B, 12, 197+(num_token), 64])
        context_layer = torch.matmul(attention_probs, value_layer) # B, num_head, num_patches, head_size 
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        
        return attention_output, weights


class Attention_SHARED_ACCROSS(nn.Module):
    def __init__(self, Prompt, Prompt_K, qkv_cfg, config, vis):
        super(Attention_SHARED_ACCROSS, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads) # 768 / 12 (same as HyperPrompt discribe)
        self.all_head_size = self.num_attention_heads * self.attention_head_size # 12 *64
        
        # print('1', self.num_attention_heads) 12
        # print('2', self.attention_head_size) 64
        # print('3', self.all_head_size) 768
        
        self.config = config
        self.qkv_cfg = qkv_cfg

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)
        
        num_layers = self.config.transformer["num_layers"]
        
        self.QKV_dropout = Dropout(self.qkv_cfg.DROPOUT) 
        # named as inherited layer since it update together in the first QKV layer
        if Prompt_K is None:
            print('goes shared+shared version(shared param for v+k and across layers)')
            self.inherited_layer = Prompt
        else:
            print('goes unshared+shared version(unshared param for v+k but share across layers)')
            self.inherited_layer = Prompt
            self.inherited_layer_K = Prompt_K
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        # print('mark', hidden_states)
        
        mixed_query_layer = self.query(hidden_states) # B, num_patches, head_size*num_head
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer) # B, num_head, num_patches, head_size
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer) # B, num_head, num_patches, head_size
        # head, sequence length (L) 应该加在这一维, dimension of each head
        # print('1', query_layer.shape) torch.Size([128, 12, 197, 64])
        # print('2', key_layer.shape) torch.Size([128, 12, 197, 64])
        # print('3', value_layer.shape) torch.Size([128, 12, 197, 64]
        

        B = query_layer.shape[0]
        if self.qkv_cfg.SHARE_PARAM_KV == False:
            # B, num_head, num_patches, head_size
            if self.qkv_cfg.LAYER_BEHIND == False:
                key_layer = torch.cat((key_layer, self.QKV_dropout(self.inherited_layer_K.expand(B, -1, -1, -1))), dim=2)
                value_layer = torch.cat((value_layer, self.QKV_dropout(self.inherited_layer.expand(B, -1, -1, -1))), dim=2)
            else:
                key_layer = torch.cat((self.QKV_dropout(self.inherited_layer_K.expand(B, -1, -1, -1)), key_layer), dim=2)
                value_layer = torch.cat((self.QKV_dropout(self.inherited_layer.expand(B, -1, -1, -1)), value_layer), dim=2)
        else:
            if self.qkv_cfg.LAYER_BEHIND == False:
                key_layer = torch.cat((key_layer, self.QKV_dropout(self.inherited_layer.expand(B, -1, -1, -1))), dim=2)
                value_layer = torch.cat((value_layer, self.QKV_dropout(self.inherited_layer.expand(B, -1, -1, -1))), dim=2)
            else:
                key_layer = torch.cat((self.QKV_dropout(self.inherited_layer.expand(B, -1, -1, -1)), key_layer), dim=2)
                value_layer = torch.cat((self.QKV_dropout(self.inherited_layer.expand(B, -1, -1, -1)), value_layer), dim=2)
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # B, num_head, num_patches, num_patches (turn into patches*patches)                    
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # 接着做softmax了 和论文中的一致
        attention_probs = self.softmax(attention_scores) # B, num_head, num_patches(query), num_patches(key)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)
        
        # print('1', attention_probs.shape) # torch.Size([B, 12, 197, 197+(num_token)])
        # print('2', value_layer.shape) # torch.Size([B, 12, 197+(num_token), 64])
        context_layer = torch.matmul(attention_probs, value_layer) # B, num_head, num_patches, head_size 
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, qkv_cfg, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(qkv_cfg, config, vis)
        # self.qkv_cfg = qkv_cfg
        # if self.qkv_cfg.MASK_QUERY == True:
            # self.Linear_Projection_QKV = Linear()

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        # print('x', x.shape) # batchsize_patchsize+pad_dim attention map size changed
        # print('h', h.shape) # batchsize_patchsize_dim
        # if self.qkv_cfg.MASK_QUERY == True:
        #     x = x.view(x.shape[0], -1).contiguous()
        #     x = self.Linear_Projection_QKV(x)
        #     x = x.reshape(64, )
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))

class Block_SHARED_ACCROSS(nn.Module):
    def __init__(self, Prompt, Prompt_K, qkv_cfg, config, vis):
        super(Block_SHARED_ACCROSS, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention_SHARED_ACCROSS(Prompt, Prompt_K, qkv_cfg, config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))

class Block_Origin_ViT(nn.Module):
    def __init__(self, config, vis):
        super(Block_Origin_ViT, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention_Origin_ViT(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))



class Encoder(nn.Module):
    def __init__(self, qkv_cfg, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        print('Num_layers:', config.transformer["num_layers"])
        
        if qkv_cfg.SHARED_ACCROSS == False: 
            for i in range(config.transformer["num_layers"]):
                if qkv_cfg.DEEP == True:
                    layer = Block(qkv_cfg, config, vis)
                    self.layer.append(copy.deepcopy(layer))
                else:
                    if i == 0:
                        print('Apply on' + str(i) + 'layer')
                        layer = Block(qkv_cfg, config, vis)
                        self.layer.append(copy.deepcopy(layer))
                    else:
                        print('Apply origin layer (for VK shallow)')
                        layer = Block_Origin_ViT(config, vis)
                        self.layer.append(copy.deepcopy(layer))
        else:
            for i in range(config.transformer["num_layers"]):
                if qkv_cfg.DEEP == True:
                    # add vk prompt layers separate
                    num_tokens = qkv_cfg.NUM_TOKENS
                    # print('num_tokens', num_tokens)
                    # if qkv_cfg.NUM_TOKENS_P is not None:
                    #     print('num_tokens_p', qkv_cfg.NUM_TOKENS_P)
                    if qkv_cfg.SHARE_PARAM_KV == False:
                        head_fixed = config.transformer["num_heads"]
                        head_size_fixed = int(config.hidden_size / head_fixed)
                        num_patches_QKV_V, num_patches_QKV_K = num_tokens, num_tokens

                        self.deep_QKV_embeddings_V = nn.Parameter(torch.zeros(
                                    head_fixed, num_patches_QKV_V, head_size_fixed))
                        self.deep_QKV_embeddings_K = nn.Parameter(torch.zeros(
                                    head_fixed, num_patches_QKV_K, head_size_fixed))
                        
                        if qkv_cfg.ORIGIN_INIT == True:
                            # xavier_uniform initialization
                            patch_size = _pair(self.config.patches["size"]) # print('patch_size', patch_size) # 16, 16
                            
                            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + 16))
                            # val = math.sqrt(6. / float(3 * reduce(mul, query_layer.shape[0], 1) + 16)) # 现在是随便设置的， 需要后期改
                            nn.init.uniform_(self.deep_QKV_embeddings_V.data, -val, val)
                            nn.init.uniform_(self.deep_QKV_embeddings_K.data, -val, val)
                        else:
                            # apply timm trunc norm for init
                            trunc_normal_(self.deep_QKV_embeddings_V, std=0.02)
                            trunc_normal_(self.deep_QKV_embeddings_K, std=0.02)
                            
                        # kaiming init # untested(to be continued)
                        # else: 
                        #     torch.nn.init.kaiming_uniform_(self.deep_QKV_embeddings_V, a=0, mode='fan_in', nonlinearity='leaky_relu')
                        #     torch.nn.init.kaiming_uniform_(self.deep_QKV_embeddings_K, a=0, mode='fan_in', nonlinearity='leaky_relu')
                        self.QKV_proj = nn.Identity()
                        
                        Before_expand_prompt_V = self.QKV_proj(self.deep_QKV_embeddings_V)
                        Before_expand_prompt_K = self.QKV_proj(self.deep_QKV_embeddings_K)
                        
                        layer = Block_SHARED_ACCROSS(Before_expand_prompt_V, Before_expand_prompt_K, qkv_cfg, config, vis)
                        self.layer.append(copy.deepcopy(layer))
                        
                    else:
                        head_fixed = config.transformer["num_heads"]
                        head_size_fixed = int(config.hidden_size / head_fixed)
                        num_patches_QKV = num_tokens
                        
                        self.deep_QKV_embeddings = nn.Parameter(torch.zeros(
                                    head_fixed, num_patches_QKV, head_size_fixed))
                        if qkv_cfg.ORIGIN_INIT == True:
                            # xavier_uniform initialization
                            patch_size = _pair(self.config.patches["size"])
                            # print('patch_size', patch_size) # 16, 16
                            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + 16))
                            # val = math.sqrt(6. / float(3 * reduce(mul, query_layer.shape[0], 1) + 16)) # 现在是随便设置的， 需要后期改
                            nn.init.uniform_(self.deep_QKV_embeddings.data, -val, val)
                        else:
                            trunc_normal_(self.deep_QKV_embeddings, std=0.02)
                        # kaiming init (to be continued, untested)
                        # else:
                        #     torch.nn.init.kaiming_uniform_(self.deep_QKV_embeddings, a=0, mode='fan_in', nonlinearity='leaky_relu')

                        self.QKV_proj = nn.Identity()
                        
                        Before_expand_prompt = self.QKV_proj(self.deep_QKV_embeddings)

                        layer = Block_SHARED_ACCROSS(Before_expand_prompt, None, qkv_cfg, config, vis)
                        self.layer.append(copy.deepcopy(layer))
                    
                else:
                    print('under construction!!!!')


    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

    def forward_cls_layerwise(self, hidden_states):
        # hidden_states: B, 1+n_patches, dim

        if hidden_states.size(0) != 1:
            raise ValueError('not support batch-wise cls forward yet')
        
        cls_embeds = []
        cls_embeds.append(hidden_states[0][0])
        for i,layer_block in enumerate(self.layer):
            hidden_states, _ = layer_block(hidden_states)
            if i < len(self.layer)-1:
                cls_embeds.append(hidden_states[0][0])
        encoded = self.encoder_norm(hidden_states)
        cls_embeds.append(hidden_states[0][0])

        cls_embeds = torch.stack(cls_embeds) # 12, dim
        return cls_embeds


class Transformer_changeVK(nn.Module):
    def __init__(self, qkv_cfg, config, img_size, vis):
        super(Transformer_changeVK, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(qkv_cfg, config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)

        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights
    
    def forward_cls_layerwise(self, input_ids):
        embedding_output = self.embeddings(input_ids)

        cls_embeds = self.encoder.forward_cls_layerwise(embedding_output)
        return cls_embeds


class VisionTransformer_changedVK(nn.Module):
    def __init__(
        self, qkv_cfg, model_type,
        img_size=224, num_classes=21843, vis=False
    ):
        assert qkv_cfg.VIT_POOL_TYPE == "original"
        super(VisionTransformer_changedVK, self).__init__()
        if qkv_cfg is None:
            raise ValueError("p_vk cannot be None if using VisionTransformer_changedVK")
        self.qkv_cfg = qkv_cfg
        
        config = CONFIGS[model_type]
        self.num_classes = num_classes
        self.classifier = config.classifier

        self.transformer = Transformer_changeVK(
            qkv_cfg, config, img_size, vis)
        self.head = Linear(config.hidden_size, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, vis=False):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        if not vis:
            return logits
        return logits, attn_weights # attn_weights: num_layers, B, num_head, num_patches, num_patches
    
    def forward_cls_layerwise(self, x):
        cls_embeds = self.transformer.forward_cls_layerwise(x)
        return cls_embeds

    def load_from(self, weights):
        with torch.no_grad():
            # print('---go through here, no gradient---')
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        # The following will be unreadable if we split lines.
        # pylint: disable=line-too-long
        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        x = self.root(x)
        x = self.body(x)
        return x
