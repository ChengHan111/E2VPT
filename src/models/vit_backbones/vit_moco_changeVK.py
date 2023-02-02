#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
"""
borrowed from https://github.com/facebookresearch/moco-v3/blob/main/vits.py
"""
import math
import torch
import torch.nn as nn
from torch.nn import Dropout
from functools import partial, reduce
from operator import mul

from timm.models.vision_transformer_changeVK import VisionTransformer_changeVK, _cfg
from timm.models.layers.helpers import to_2tuple
from timm.models.layers import PatchEmbed

__all__ = [
    'vit_small',
    'vit_base',
    'vit_conv_small',
    'vit_conv_base',
]


class VisionTransformerMoCo(VisionTransformer_changeVK):
    def __init__(self, p_vk_cfg, stop_grad_conv1=False, **kwargs):
        super().__init__(**kwargs)
        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()
        
        ##
        self.p_vk_cfg = p_vk_cfg
        print('p_vk_cfg in VisionTransformerMoCo', p_vk_cfg)      
          
        del self.blocks
        
        print(self.depth, self.num_heads, self.mlp_ratio,
              self.qkv_bias, self.drop_rate, self.attn_drop_rate,
              self.dpr, self.norm_layer,self.act_layer)
        
        self.blocks = nn.Sequential(*[
            Block_VK(
                p_vk_cfg=p_vk_cfg, dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, drop=self.drop_rate,
                attn_drop=self.attn_drop_rate, drop_path=self.dpr[i], norm_layer=self.norm_layer, act_layer=self.act_layer)
            for i in range(self.depth)])
        ##

        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            # xavier_uniform initialization
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        assert self.num_tokens == 1, 'Assuming one and only one token, [cls]'
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False


class ConvStem(nn.Module):
    """
    ConvStem, from Early Convolutions Help Transformers See Better, Tete et al. https://arxiv.org/abs/2106.14881
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 16, 'ConvStem only supports patch size of 16'
        assert embed_dim % 8 == 0, 'Embed dimension must be divisible by 8 for ConvStem'

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # build stem, similar to the design in https://arxiv.org/abs/2106.14881
        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(4):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def vit_small(**kwargs):
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_base(**kwargs):
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_conv_small(**kwargs):
    # minus one ViT block
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=384, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_conv_base(**kwargs):
    # minus one ViT block
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=768, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model

# New added blocks for changing VK
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
