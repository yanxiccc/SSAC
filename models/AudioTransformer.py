#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import torch
import torch.nn as nn
import torchaudio
from torch import einsum
from collections import OrderedDict
from models.SpecAugment import SpecAugmentation
from models.main_finetune_as import PatchEmbed_new
from timm.models.vision_transformer import PatchEmbed, Block
from einops import rearrange
from einops import repeat
from einops.layers.torch import Rearrange
from timm.models.layers import to_2tuple, DropPath
""" Adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py"""


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight) # 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.) # 


def init_bn(bn):
    """Initialize a BatchNorm layer."""
    bn.bias.data.fill_(0.)  # 
    bn.weight.data.fill_(1.) # 



class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim=3072, drop=0.):
        super(Mlp, self).__init__()
#        out_features = out_features or dim
        #hidden_features = hidden_features or dim
        self.fc1 = nn.Linear(dim, 3072)##########
        self.act = nn.GELU()
        self.fc2 = nn.Linear(3072, dim)#############
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwags):
        output = self.norm(x)
        output = self.fn(output, **kwags)
        return output


class FeedForward1(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.):

        super(FeedForward1, self).__init__()
        
        self.mlp = Mlp(dim, hidden_features=hidden_dim)

    def forward(self, x):
      
        return self.mlp(x)   # 正向传播

class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.):

        super(FeedForward, self).__init__()
        self.mlp = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(dim, hidden_dim)),
            ('ac1', nn.GELU()),
            ('dropout1', nn.Dropout(dropout)),
            ('fc2', nn.Linear(hidden_dim, dim)),
            ('dropout2', nn.Dropout(dropout))
        ]))

    def forward(self, x):
        return self.mlp(x)


class Attention(nn.Module):

    def __init__(self, dim, heads=12, dim_head=64, dropout=0.):
        '''
        dim: dim of input
        dim_head: dim of q, k, v
        '''
        super(Attention, self).__init__()

        inner_dim = dim_head * heads
        project_out = not(heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.qkv = nn.Linear(dim, dim * 3)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        '''self.proj = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )''' if project_out else nn.Identity() # 


    def forward(self, x):

        b, n, _, h = *x.shape, self.heads
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  #


        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')  # 
        out = self.proj(out)
        out = self.proj_drop(out)
        return out




class Block(nn.Module): #Transformer
    def __init__(self, dim, depth, heads, dim_head, hidden_dim, mlp_dim, drop_path=0., dropout=0.):
        super(Block, self).__init__()
        #self.layers = nn.ModuleList([])
        #for _ in range(depth):
        self.norm1 = nn.LayerNorm(dim)#######
        self.attn = Attention(dim, heads, dim_head, dropout)

        '''self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),#对应attn
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))#对应ff]))'''
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)###########
        self.mlp_dim = mlp_dim * dim
        self.mlp = Mlp(dim, hidden_dim=mlp_dim)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        '''for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x'''
        return x

class PatchEmbed_new(nn.Module):
    """ Flexible Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=16):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
       
        self.img_size = img_size
        self.patch_size = patch_size
        
# 
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride) # with overlapped patches
        #self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        #self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        #self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        _, _, h, w = self.get_output_shape(img_size) # n, emb_dim, h, w  _表示n,emb_dim，但是用不到
        self.patch_hw = (h, w)
        self.num_patches = h*w

    def get_output_shape(self, img_size):
        # todo: don't be lazy..
        return self.proj(torch.randn(1,1,img_size[0],img_size[1])).shape 

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        #assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x) 
        x = x.flatten(2).transpose(1, 2)  #
        return x  # 
class AudioTransformer1(nn.Module):

    def __init__(self, patch_size, num_classes, dim, depth, heads, mlp_dim, img_size=(1000,64), drop_path_rate=0., dim_head=64, stride=16, embed_dim=768, in_chans=1, dropout=0.):
        super(AudioTransformer1, self).__init__()
        
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)#长度?的元�?        
        patch_height, patch_width = pair(patch_size)

        #patch_dim = patch_height * patch_width  # 64 * 4 = 256 (16 * 16)

        self.bn0 = nn.BatchNorm2d(64)  #?        #self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        #self.patch_embed = PatchEmbed_org(img_size, patch_size, in_chans, embed_dim)
        #self.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=16, in_chans=1, embed_dim=embed_dim, stride=16)
        '''self.patch_embed = nn.Sequential(OrderedDict([
            ('rerange', Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width)),
            ('proj', nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size))
        ]))'''

        self.spec_augmenter = SpecAugmentation(time_drop_width=64,
                                               time_stripes_num=2,
                                               freq_drop_width=8,
                                               freq_stripes_num=2,
                                               mask_type='zero_value')####

        #self.pos_embedding = nn.Parameter(torch.zeros(768, 125 + 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 512 + 1, dim))#?????
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim, depth, heads, dim_head, mlp_dim, dropout, drop_path=dpr[i])
        for i in range(depth)])


#self.blocks = Block(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()
        self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()
        self.fc_norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)#
        )

    def forward(self, spec):

        x = spec.unsqueeze(1)
        #print(x.size())
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        if self.training:
            x = self.spec_augmenter(x)
        x = self.patch_embed(x)
        
        b, n, _ = x.shape

        cls_token = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_token, x), dim=1)

        x += self.pos_embedding[:, :(n + 1)]

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)
        #x = self.blocks(x)

        x = self.to_latent(x)
    
        x = self.fc_norm(x)
        return self.head(x) #



    
