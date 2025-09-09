from __future__ import annotations
import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- utils ----------
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

class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hid = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hid)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid, dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

# -------- Residual gate (for skip connection ablation) ----------
class ResidualAlpha(nn.Module):
    """
    Wrap a sublayer F into: x + alpha * F(x, ...)
    Set alpha=1.0 -> normal residual; alpha=0.0 -> residual removed (skip off).
    """
    def __init__(self, fn, alpha: float = 1.0):
        super().__init__()
        self.fn = fn
        self.register_buffer("alpha", torch.tensor(float(alpha)))
    def forward(self, x, *args, **kwargs):
        return x + self.alpha * self.fn(x, *args, **kwargs)

# -------- FAVOR+ random features ----------
def gaussian_orthogonal_random_matrix(d: int, m: int, device=None, dtype=torch.float32):
    blocks = []
    n_full, m_rem = divmod(m, d)
    for _ in range(n_full):
        A = torch.randn(d, d, device=device, dtype=dtype)
        Q, _ = torch.linalg.qr(A, mode="reduced")
        blocks.append(Q)
    if m_rem > 0:
        A = torch.randn(d, d, device=device, dtype=dtype)
        Q, _ = torch.linalg.qr(A, mode="reduced")
        blocks.append(Q[:, :m_rem])
    return torch.cat(blocks, dim=1)

def softmax_feature(x: torch.Tensor, proj: torch.Tensor, eps: float = 1e-6):
    # x: (B,N,H,Dh), proj: (H,Dh,M)
    xw = torch.einsum('bnhd,hdm->bnhm', x, proj)  # (B,N,H,M)
    x_sq = (x.pow(2).sum(dim=-1, keepdim=True)) * 0.5  # (B,N,H,1)
    max_xw = xw.max(dim=-1, keepdim=True).values
    phi = torch.exp(xw - max_xw) * torch.exp(-x_sq + max_xw) + eps
    return phi

# -------- Performer attention (token sequence) ----------
class PerformerAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, m_features: int = 128,
                 qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0,
                 redraw_interval: int = 0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.m_features = m_features

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.o = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.register_buffer("proj_matrix", None, persistent=False)
        self.redraw_interval = int(redraw_interval)
        self._calls = 0

    def _maybe_init_proj(self, device):
        if self.proj_matrix is None or self.proj_matrix.device != device:
            pm = []
            for _ in range(self.num_heads):
                W = gaussian_orthogonal_random_matrix(
                    self.head_dim, self.m_features, device=device, dtype=torch.float32
                )
                pm.append(W.unsqueeze(0))
            self.proj_matrix = torch.cat(pm, dim=0)  # (H,Dh,M)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        self._maybe_init_proj(x.device)

        q = self.q(x); k = self.k(x); v = self.v(x)
        def split(t): return t.view(B, N, self.num_heads, self.head_dim)

        q = split(q) * self.scale
        k = split(k)
        v = split(v)

        # mask: (B,1,1,N) with 0 for padding positions
        if attn_mask is not None:
            m = attn_mask.to(k.dtype)  # broadcastable (B,N,1,1) or (B,1,1,N) depending upstream
            k = k * m
            v = v * m

        # kernel features in float32
        qf = softmax_feature(q.float(), self.proj_matrix)
        kf = softmax_feature(k.float(), self.proj_matrix)

        kv   = torch.einsum('bnhm,bnhd->bhmd', kf, v.float())   # (B,H,M,Dh)
        ksum = kf.sum(dim=1)                                    # (B,H,M)
        denom = torch.einsum('bnhm,bhm->bhn', qf, ksum).clamp(min=1e-6).unsqueeze(-1)
        ctx = torch.einsum('bnhm,bhmd->bhnd', qf, kv) / denom   # (B,H,N,Dh)

        out = ctx.transpose(1, 2).contiguous().view(B, N, C)
        out = self.attn_drop(out)
        out = self.o(out)
        out = self.proj_drop(out)

        if self.redraw_interval > 0:
            self._calls += 1
            if self._calls % self.redraw_interval == 0:
                self.proj_matrix = None
        return out

class PerformerBlock(nn.Module):
    def __init__(self, dim, num_heads, m_features, mlp_ratio=4.0, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, qkv_bias=True, redraw_interval=0,
                 alpha_attn: float = 1.0, alpha_mlp: float = 1.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PerformerAttention(dim, num_heads, m_features, qkv_bias,
                                       attn_drop, drop, redraw_interval)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio, drop)

        # wrap with residual gates
        self.resid_attn = ResidualAlpha(
            fn=lambda y, mask=None: self.drop_path(self.attn(self.norm1(y), mask)),
            alpha=alpha_attn
        )
        self.resid_mlp = ResidualAlpha(
            fn=lambda y: self.drop_path(self.mlp(self.norm2(y))),
            alpha=alpha_mlp
        )

    def forward(self, x, attn_mask=None):
        x = self.resid_attn(x, attn_mask)  # x + alpha_attn * DropPath(Attn(LN(x)))
        x = self.resid_mlp(x)              # x + alpha_mlp  * DropPath(MLP(LN(x)))
        return x

# -------- Text Performer encoder + classifier ----------
class TextPerformer(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 num_classes: int = 2,
                 max_len: int = 1024,
                 embed_dim: int = 384,
                 depth: int = 8,
                 num_heads: int = 6,
                 m_features: int = 128,
                 mlp_ratio: float = 4.0,
                 drop_rate: float = 0.1,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.1,
                 use_cls_token: bool = True,
                 # NEW: residual gates
                 alpha_attn: float = 1.0,
                 alpha_mlp: float = 1.0,
                 redraw_interval: int = 0):
        super().__init__()
        self.use_cls_token = use_cls_token

        self.tok = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len + (1 if use_cls_token else 0), embed_dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_cls_token else None
        nn.init.trunc_normal_(self.pos, std=0.02)
        if self.cls is not None:
            nn.init.trunc_normal_(self.cls, std=0.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            PerformerBlock(embed_dim, num_heads, m_features,
                           mlp_ratio=mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate,
                           drop_path=dpr[i], qkv_bias=True, redraw_interval=redraw_interval,
                           alpha_attn=alpha_attn, alpha_mlp=alpha_mlp)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        input_ids: (B, N) token ids; attention_mask: (B, N) with 1 for real, 0 for pad
        """
        B, N = input_ids.shape
        x = self.tok(input_ids)  # (B,N,C)

        if self.use_cls_token:
            cls = self.cls.expand(B, -1, -1)  # (B,1,C)
            x = torch.cat([cls, x], dim=1)    # (B,N+1,C)
            pos = self.pos[:, :N + 1, :]
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask.new_ones(B, 1), attention_mask], dim=1)
        else:
            pos = self.pos[:, :N, :]

        x = x + pos

        attn_mask_feat = None
        if attention_mask is not None:
            # make it broadcastable for attention: (B,N,1,1)
            attn_mask_feat = attention_mask[:, :, None, None].to(x.dtype)

        for blk in self.blocks:
            x = blk(x, attn_mask_feat)

        x = self.norm(x)
        if self.use_cls_token:
            x = x[:, 0]
        else:
            # masked average
            if attention_mask is None:
                x = x.mean(dim=1)
            else:
                x = (x * attention_mask[:, :, None]).sum(dim=1) / (attention_mask.sum(dim=1, keepdim=True) + 1e-6)
        return self.head(x)

def performer_text_small(vocab_size: int, num_classes: int = 2, max_len: int = 1024,
                         embed_dim=384, depth=8, num_heads=6, m_features=128,
                         mlp_ratio=4.0, drop_rate=0.1, drop_path_rate=0.1,
                         use_cls_token=True,
                         # NEW: residual gates
                         alpha_attn: float = 1.0, alpha_mlp: float = 1.0,
                         redraw_interval: int = 0):
    return TextPerformer(
        vocab_size=vocab_size, num_classes=num_classes, max_len=max_len,
        embed_dim=embed_dim, depth=depth, num_heads=num_heads, m_features=m_features,
        mlp_ratio=mlp_ratio, drop_rate=drop_rate, attn_drop_rate=0.0,
        drop_path_rate=drop_path_rate, use_cls_token=use_cls_token,
        alpha_attn=alpha_attn, alpha_mlp=alpha_mlp, redraw_interval=redraw_interval
    )
