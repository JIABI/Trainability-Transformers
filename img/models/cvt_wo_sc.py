from __future__ import annotations
import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Residual gate (skip-connection ablation)
# -----------------------------
class ResidualAlpha(nn.Module):
    def __init__(self, fn, alpha: float = 1.0):
        """
        Wrap a sub-layer F(x, ...) into: x + alpha * F(x, ...)
        alpha=1 -> normal residual; alpha=0 -> skip removed.
        """
        super().__init__()
        self.fn = fn
        self.register_buffer("alpha", torch.tensor(float(alpha)))

    def forward(self, x, *args, **kwargs):
        return x + self.alpha * self.fn(x, *args, **kwargs)


# -----------------------------
# Stochastic depth
# -----------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rnd = x.new_empty(shape).bernoulli_(keep)
        return x * rnd.div(keep)


# -----------------------------
# Conv stem for each stage
# -----------------------------
class ConvPatchEmbed(nn.Module):
    """
    Conv embedding with stride for pyramid downsampling.
    img: (B, C_in, H, W) -> tokens (B, HW', C_out), H',W' reduced by stride
    """
    def __init__(self, in_chans: int, embed_dim: int, kernel_size: int, stride: int):
        super().__init__()
        pad = kernel_size // 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)  # (B, C, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H'*W', C)
        x = self.norm(x)
        return x, H, W


# -----------------------------
# MLP block
# -----------------------------
class Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


# -----------------------------
# Convolutional Projection Attention (CvT)
# -----------------------------
class ConvAttention(nn.Module):
    """
    Multi-head attention where q/k/v are produced by depthwise conv on 2D feature maps.
    """
    def __init__(self, dim: int, num_heads: int, kernel_size: int = 3,
                 qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 1x1 pointwise then depthwise conv for locality
        self.q_pw = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.k_pw = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.v_pw = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

        pad = kernel_size // 2
        self.q_dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=pad, groups=dim, bias=False)
        self.k_dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=pad, groups=dim, bias=False)
        self.v_dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=pad, groups=dim, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        x: (B, N, C) tokens, where N = H*W
        returns: (B, N, C)
        """
        B, N, C = x.shape
        assert N == H * W, "Token count mismatch with H,W"
        x2d = x.transpose(1, 2).reshape(B, C, H, W)  # (B, C, H, W)

        q = self.q_dw(self.q_pw(x2d))
        k = self.k_dw(self.k_pw(x2d))
        v = self.v_dw(self.v_pw(x2d))

        # flatten to tokens per head
        q = q.flatten(2).transpose(1, 2).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B,h,N,d)
        k = k.flatten(2).transpose(1, 2).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B,h,N,N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B,h,N,d)
        out = out.transpose(1, 2).reshape(B, N, C)  # (B,N,C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# -----------------------------
# Transformer block with ConvAttention (with residual gates)
# -----------------------------
class CvTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float,
                 attn_kernel: int, qkv_bias: bool, drop: float, attn_drop: float,
                 drop_path: float,
                 alpha_attn: float = 1.0, alpha_mlp: float = 1.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ConvAttention(dim, num_heads, kernel_size=attn_kernel,
                                  qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio=mlp_ratio, drop=drop)

        # residual gates wrap whole sub-branches (LN + sublayer + DropPath)
        self.resid_attn = ResidualAlpha(
            fn=lambda y, H, W: self.drop_path(self.attn(self.norm1(y), H, W)),
            alpha=alpha_attn
        )
        self.resid_mlp = ResidualAlpha(
            fn=lambda y: self.drop_path(self.mlp(self.norm2(y))),
            alpha=alpha_mlp
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.resid_attn(x, H, W)  # x + alpha_attn * DropPath(Attn(LN(x)))
        x = self.resid_mlp(x)         # x + alpha_mlp  * DropPath(MLP(LN(x)))
        return x


# -----------------------------
# Stage = Conv stem + multiple CvTBlocks
# -----------------------------
class CvTStage(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, depth: int, num_heads: int,
                 stem_kernel: int, stem_stride: int, attn_kernel: int,
                 mlp_ratio: float, qkv_bias: bool, drop: float, attn_drop: float,
                 drop_path: List[float], disable_downsample: bool = False,
                 alpha_attn: float = 1.0, alpha_mlp: float = 1.0):
        super().__init__()
        # stem (downsample unless disabled)
        stride = 1 if disable_downsample else stem_stride
        self.stem = ConvPatchEmbed(in_chans, embed_dim, kernel_size=stem_kernel, stride=stride)
        self.blocks = nn.ModuleList([
            CvTBlock(embed_dim, num_heads, mlp_ratio, attn_kernel, qkv_bias, drop, attn_drop,
                     drop_path[i] if isinstance(drop_path, list) else drop_path,
                     alpha_attn=alpha_attn, alpha_mlp=alpha_mlp)
            for i in range(depth)
        ])
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int, int]:
        # stem expects 4D image tensor (B,C,H,W)
        if x.ndim == 3:
            B, N, C = x.shape
            x = x.transpose(1, 2).reshape(B, C, H, W)
        x, H, W = self.stem(x)  # (B, H'*W', C)
        for blk in self.blocks:
            x = blk(x, H, W)
        return x, H, W, self.embed_dim


# -----------------------------
# CvT top-level classifier (with residual gates)
# -----------------------------
class CvT(nn.Module):
    """
    3-stage hierarchical CvT classifier (CvT-13 style by default)
    """
    def __init__(self,
                 img_size: int = 224,
                 in_chans: int = 3,
                 num_classes: int = 1000,
                 embed_dims: List[int] = [64, 192, 384],
                 depths: List[int] = [1, 2, 10],
                 num_heads: List[int] = [1, 3, 6],
                 stem_kernels: List[int] = [7, 3, 3],
                 stem_strides: List[int] = [4, 2, 2],
                 attn_kernels: List[int] = [3, 3, 3],
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.1,
                 global_pool: str = "mean",
                 disable_pyramid: bool = False,
                 # NEW: residual gates (apply to all blocks)
                 alpha_attn: float = 1.0,
                 alpha_mlp: float = 1.0):
        super().__init__()
        assert len(embed_dims) == len(depths) == len(num_heads) == 3
        self.num_classes = num_classes
        self.global_pool = global_pool

        # stochastic depth schedule across all blocks
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        cursor = 0

        # stages
        stages = []
        c_in = in_chans
        for i in range(3):
            stage = CvTStage(
                in_chans=c_in,
                embed_dim=embed_dims[i],
                depth=depths[i],
                num_heads=num_heads[i],
                stem_kernel=stem_kernels[i],
                stem_stride=stem_strides[i],
                attn_kernel=attn_kernels[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cursor: cursor + depths[i]],
                disable_downsample=disable_pyramid and i > 0,  # 禁用后两级下采样的选项
                alpha_attn=alpha_attn,
                alpha_mlp=alpha_mlp
            )
            stages.append(stage)
            cursor += depths[i]
            c_in = embed_dims[i]
        self.stages = nn.ModuleList(stages)

        self.norm = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # track H,W as we downsample
        H, W = x.shape[-2], x.shape[-1]
        out = x
        for st in self.stages:
            out, H, W, C = st(out, H, W)  # out: (B, H'*W', C)
        out = self.norm(out)             # (B, N, C_last)
        out = out.mean(dim=1)            # global mean pool over tokens
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x


# -----------------------------
# Factories
# -----------------------------
def cvt_13(img_size=224, num_classes=1000, drop_path_rate=0.1,
           global_pool="mean", disable_pyramid: bool = False,
           # NEW: residual gates
           alpha_attn: float = 1.0, alpha_mlp: float = 1.0):
    """
    CvT-13 style: dims [64,192,384], depths [1,2,10], heads [1,3,6]
    """
    return CvT(
        img_size=img_size, in_chans=3, num_classes=num_classes,
        embed_dims=[64, 192, 384],
        depths=[1, 2, 10],
        num_heads=[1, 3, 6],
        stem_kernels=[7, 3, 3],
        stem_strides=[4, 2, 2],
        attn_kernels=[3, 3, 3],
        mlp_ratio=4.0, qkv_bias=True,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=drop_path_rate,
        global_pool=global_pool, disable_pyramid=disable_pyramid,
        alpha_attn=alpha_attn, alpha_mlp=alpha_mlp
    )


def cvt_13_nopyramid(img_size=224, num_classes=1000, drop_path_rate=0.1,
                     global_pool="mean",
                     # NEW: residual gates
                     alpha_attn: float = 1.0, alpha_mlp: float = 1.0):
    """
    Ablation: disable downsampling in later stages (keep token count roughly constant).
    """
    return cvt_13(img_size=img_size, num_classes=num_classes,
                  drop_path_rate=drop_path_rate, global_pool=global_pool,
                  disable_pyramid=True,
                  alpha_attn=alpha_attn, alpha_mlp=alpha_mlp)
