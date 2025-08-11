# vit.py

# Minimal Vision Transformer (ViT-B/16) for multi-label classification on COCO

# Author: you 🫶

import math

from typing import Optional

import torch

from torch import nn

from torch.nn import functional as F


# -------------------------

# Utils: DropPath (Stochastic Depth)

# -------------------------

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob

    shape = (x.shape[0],) + (1,) * (x.ndim - 1)

    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)

    random_tensor.floor_()

    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()

        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# -------------------------

# MLP block

# -------------------------

class MLP(nn.Module):

    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()

        hidden = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, hidden)

        self.act = nn.GELU()

        self.fc2 = nn.Linear(hidden, dim)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.act(x)

        x = self.drop(x)

        x = self.fc2(x)

        x = self.drop(x)

        return x


# -------------------------

# Multi-head Self-Attention

# -------------------------

class Attention(nn.Module):

    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads

        self.head_dim = dim // num_heads

        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x)  # (B, N, 3C)

        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        q, v = qkv[0], qkv[2]

        k = qkv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)

        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = attn @ v  # (B, heads, N, head_dim)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)

        x = self.proj_drop(x)

        return x


# -------------------------

# Transformer Block

# -------------------------

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,

                 drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)

        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


# -------------------------

# Patch Embedding

# -------------------------

class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = (img_size, img_size)

        patch_size = (patch_size, patch_size)

        self.img_size = img_size

        self.patch_size = patch_size

        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):  # x: (B,3,H,W)

        x = self.proj(x)  # (B,C,H/P,W/P)

        x = x.flatten(2).transpose(1, 2)  # (B, N, C)

        return x


# -------------------------

# Vision Transformer

# -------------------------

class VisionTransformer(nn.Module):

    def __init__(

            self,

            img_size=224,

            patch_size=16,

            in_chans=3,

            num_classes=80,

            embed_dim=768,

            depth=12,

            num_heads=12,

            mlp_ratio=4.0,

            qkv_bias=True,

            drop_rate=0.0,

            attn_drop_rate=0.0,

            drop_path_rate=0.1,

            global_pool: str = "cls",  # "cls" or "mean"

    ):

        super().__init__()

        self.num_classes = num_classes

        self.num_features = embed_dim

        self.global_pool = global_pool

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([

            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, dpr[i])

            for i in range(depth)

        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.apply(self._init_module)

    @staticmethod
    def _init_module(m):

        if isinstance(m, nn.Linear):

            nn.init.trunc_normal_(m.weight, std=0.02)

            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.LayerNorm):

            nn.init.ones_(m.weight)

            nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Conv2d):

            nn.init.kaiming_normal_(m.weight, mode="fan_out")

            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def interpolate_pos_encoding(self, x, H: int, W: int):

        # x: (B, N+1, C) with class token concatenated

        N = x.shape[1] - 1

        N0 = self.pos_embed.shape[1] - 1

        if N == N0 and H == self.patch_embed.grid_size[0] and W == self.patch_embed.grid_size[1]:
            return self.pos_embed

        # separate cls and patch pos

        cls_pos = self.pos_embed[:, :1]

        patch_pos = self.pos_embed[:, 1:]

        dim = x.shape[-1]

        # old grid size

        gs_old = int(math.sqrt(N0))

        patch_pos = patch_pos.reshape(1, gs_old, gs_old, dim).permute(0, 3, 1, 2)  # (1,C,gs,gs)

        patch_pos = F.interpolate(patch_pos, size=(H, W), mode="bicubic", align_corners=False)

        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, H * W, dim)

        return torch.cat((cls_pos, patch_pos), dim=1)

    def forward_features(self, x):

        B, _, H, W = x.shape

        x = self.patch_embed(x)  # (B, N, C) where N=H/P*W/P

        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1+N, C)

        # interpolate positional embeddings if size differs

        H_grid = H // self.patch_embed.patch_size[0]

        W_grid = W // self.patch_embed.patch_size[1]

        pos_embed = self.interpolate_pos_encoding(x, H_grid, W_grid)

        x = x + pos_embed

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if self.global_pool == "mean":

            return x[:, 1:, :].mean(dim=1)

        else:  # cls token

            return x[:, 0]

    def forward(self, x):

        feats = self.forward_features(x)

        logits = self.head(feats)

        return logits


def vit_b16(img_size=224, num_classes=80, drop_path_rate=0.1, global_pool="cls"):
    return VisionTransformer(

        img_size=img_size,

        patch_size=16,

        in_chans=3,

        num_classes=num_classes,

        embed_dim=768,

        depth=12,

        num_heads=12,

        mlp_ratio=4.0,

        qkv_bias=True,

        drop_rate=0.0,

        attn_drop_rate=0.0,

        drop_path_rate=drop_path_rate,

        global_pool=global_pool,

    )

