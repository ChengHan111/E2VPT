#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
"""
borrowed from https://github.com/facebookresearch/mae/blob/main/models_vit.py
"""
from functools import partial

import torch
import torch.nn as nn
from torch.nn import Dropout

# 这里说不定可以换成原始的
import timm.models.vision_transformer_changeVK # add vision_transformer_changeVK in conda env #TODO: make it clear on readme and instructions

class VisionTransformer(timm.models.vision_transformer_changeVK.VisionTransformer_changeVK):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, p_vk_cfg, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        self.p_vk_cfg = p_vk_cfg
        # print(p_vk_cfg)      
          
        del self.blocks
        
        # print(self.depth, self.num_heads, self.mlp_ratio,
        #       self.qkv_bias, self.drop_rate, self.attn_drop_rate,
        #       self.dpr, self.norm_layer,self.act_layer)
        
        self.blocks = nn.Sequential(*[
            Block_VK(
                p_vk_cfg=p_vk_cfg, dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, drop=self.drop_rate,
                attn_drop=self.attn_drop_rate, drop_path=self.dpr[i], norm_layer=self.norm_layer, act_layer=self.act_layer)
            for i in range(self.depth)])
        
        if self.global_pool:
            self.norm_layer = norm_layer = kwargs['norm_layer']
            self.embed_dim = embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

class Block_VK(nn.Module):

    def __init__(self, p_vk_cfg, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_VK(p_vk_cfg, dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Attention_VK(nn.Module):
    def __init__(self, p_vk_cfg, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.p_vk_cfg = p_vk_cfg
        num_tokens = self.p_vk_cfg.NUM_TOKENS
        print('num_tokens_VK', num_tokens)
        if self.p_vk_cfg.NUM_TOKENS_P is not None:
            print('num_tokens_p', self.p_vk_cfg.NUM_TOKENS_P)
        
        # add vk prompt layers jointly
        if self.p_vk_cfg.SHARE_PARAM_KV == True:
            
            head_fixed, num_patches_QKV, head_size_fixed = self.num_heads, num_tokens, head_dim
            self.deep_QKV_embeddings = nn.Parameter(torch.zeros(
                        head_fixed, num_patches_QKV, head_size_fixed))
            if self.p_vk_cfg.ORIGIN_INIT == '0':
                # xavier_uniform initialization
                patch_size = _pair(self.config.patches["size"])
                # print('patch_size', patch_size) # 16, 16
                val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + 16))
                # val = math.sqrt(6. / float(3 * reduce(mul, query_layer.shape[0], 1) + 16)) # 现在是随便设置的， 需要后期改
                nn.init.uniform_(self.deep_QKV_embeddings.data, -val, val)
            elif self.p_vk_cfg.ORIGIN_INIT == '1':
                trunc_normal_(self.deep_QKV_embeddings, std=0.02)
            else:
                torch.nn.init.kaiming_uniform_(self.deep_QKV_embeddings, a=0, mode='fan_in', nonlinearity='leaky_relu')
        else:
            raise ValueError("Not supported for unshare VK in MAE setting! Under construction")

        self.QKV_proj = nn.Identity()
        # self.prompt_config.DROPOUT
        self.QKV_dropout = Dropout(self.p_vk_cfg.DROPOUT) # should add config here
        
        
    def forward(self, x):
        # print('go through Attention!!!!')
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        
        B = q.shape[0] # should be the batch size
        # print('B', B)
        if self.p_vk_cfg.SHARE_PARAM_KV == True:
            if self.p_vk_cfg.LAYER_BEHIND == False:
                # print('K', k.shape)
                # print('V', v.shape)
                # print('self-defined layer shape', self.QKV_dropout(self.QKV_proj(self.deep_QKV_embeddings).expand(B, -1, -1, -1)).shape)

                k = torch.cat((k, self.QKV_dropout(self.QKV_proj(self.deep_QKV_embeddings).expand(B, -1, -1, -1))), dim=2)
                v = torch.cat((v, self.QKV_dropout(self.QKV_proj(self.deep_QKV_embeddings).expand(B, -1, -1, -1))), dim=2)
            else:
                # print('K', k.shape)
                # print('V', v.shape)
                # print('self-defined layer shape', self.QKV_dropout(self.QKV_proj(self.deep_QKV_embeddings).expand(B, -1, -1, -1)).shape)
                
                k = torch.cat((self.QKV_dropout(self.QKV_proj(self.deep_QKV_embeddings).expand(B, -1, -1, -1)), k), dim=2)
                v = torch.cat((self.QKV_dropout(self.QKV_proj(self.deep_QKV_embeddings).expand(B, -1, -1, -1)), v), dim=2)
        else:
            raise ValueError("Not supported for unshare VK in MAE setting! Under construction")
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def build_model(model_type):
    if "vitb" in model_type:
        return vit_base_patch16()
    elif "vitl" in model_type:
        return vit_large_patch16()
    elif "vith" in model_type:
        return vit_huge_patch14()


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        drop_path_rate=0.1, global_pool=True,  # using default settings for mae-finetune
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        drop_path_rate=0.1, global_pool=True,  # using default settings for mae-finetune
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        drop_path_rate=0.1, global_pool=True,  # using default settings for mae-finetune
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
