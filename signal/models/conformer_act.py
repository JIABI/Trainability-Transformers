from __future__ import annotations

from typing import Optional

import math

import torch

import torch.nn as nn

import torch.nn.functional as F


# -------------------------------

# Utils

# -------------------------------

class DropPath(nn.Module):

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()

        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep = 1.0 - self.drop_prob

        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        rnd = x.new_empty(shape).bernoulli_(keep)

        return x * rnd.div(keep)


class SinusoidalPositionalEncoding(nn.Module):
    """Absolute 1D sinusoidal PE added to (B, T, C)."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        pos = torch.arange(0, max_len).unsqueeze(1).float()

        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)

        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)

        return x + self.pe[:T, :].to(dtype=x.dtype, device=x.device)


# -------------------------------

# Conformer submodules

# -------------------------------

def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    elif name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "silu":
        return nn.SiLU()
    elif name == "selu":
        return nn.SELU()
    else:
        raise ValueError(f"Unknown activation: {name}")


# -------------------------------
# Positional Encoding
# -------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:T, :].to(dtype=x.dtype, device=x.device)


# -------------------------------
# Submodules
# -------------------------------
class FeedForward(nn.Module):
    """Macaron FFN: Linear -> Act -> Drop -> Linear"""

    def __init__(self, d_model: int, expansion: float = 4.0,
                 drop: float = 0.0, act_layer: str = "silu"):
        super().__init__()
        dff = int(d_model * expansion)
        self.fc1 = nn.Linear(d_model, dff)
        self.act = get_activation(act_layer)
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(dff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))


class MultiHeadSelfAttention1D(nn.Module):
    def __init__(self, d_model: int, num_heads: int,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=attn_drop, batch_first=True
        )
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        return self.proj_drop(out)


class ConvModule1D(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 15,
                 drop: float = 0.0, act_layer: str = "silu"):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                            padding=kernel_size // 2, groups=d_model, bias=True)
        self.bn = nn.BatchNorm1d(d_model)
        self.act = get_activation(act_layer)
        self.pw2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln(x)
        y = y.transpose(1, 2)
        y = self.pw1(y)
        y = F.glu(y, dim=1)
        y = self.dw(y)
        y = self.bn(y)
        y = self.act(y)
        y = self.pw2(y)
        y = y.transpose(1, 2)
        y = self.drop(y)
        return y


class ConformerBlock1D(nn.Module):
    def __init__(self, d_model: int, num_heads: int,
                 ffn_expansion: float = 4.0, conv_kernel: int = 15,
                 attn_drop: float = 0.0, ff_drop: float = 0.0,
                 conv_drop: float = 0.0, drop_path: float = 0.0,
                 act_layer: str = "silu"):
        super().__init__()
        self.ff1 = FeedForward(d_model, expansion=ffn_expansion, drop=ff_drop, act_layer=act_layer)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention1D(d_model, num_heads, attn_drop=attn_drop, proj_drop=ff_drop)
        self.ln2 = nn.LayerNorm(d_model)
        self.conv = ConvModule1D(d_model, kernel_size=conv_kernel, drop=conv_drop, act_layer=act_layer)
        self.ln3 = nn.LayerNorm(d_model)
        self.ff2 = FeedForward(d_model, expansion=ffn_expansion, drop=ff_drop, act_layer=act_layer)
        self.droppath = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + 0.5 * self.droppath(self.ff1(self.ln1(x)))
        x = x + self.droppath(self.attn(self.ln2(x), key_padding_mask=key_padding_mask))
        x = x + self.droppath(self.conv(x))
        x = x + 0.5 * self.droppath(self.ff2(self.ln3(x)))
        return x


# -------------------------------
# Subsampling
# -------------------------------
class ConvSubsample1D(nn.Module):
    def __init__(self, in_chans: int, d_model: int,
                 factor: int = 2, kernel_size: int = 5):
        super().__init__()
        assert factor in (1, 2, 4, 8)
        layers = [nn.Conv1d(in_chans, d_model, kernel_size=3, stride=1, padding=1, bias=False),
                  nn.GELU()]
        s = 1
        while s < factor:
            layers += [nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                                 stride=2, padding=kernel_size // 2, bias=False),
                       nn.GELU()]
            s *= 2
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------------
# Conformer-1D Classifier
# -------------------------------
class Conformer1D(nn.Module):
    def __init__(self, in_chans: int = 1, num_classes: int = 5,
                 d_model: int = 144, depth: int = 6, num_heads: int = 6,
                 ffn_expansion: float = 4.0, conv_kernel: int = 15,
                 attn_drop: float = 0.0, ff_drop: float = 0.1,
                 conv_drop: float = 0.1, drop_path_rate: float = 0.1,
                 subsample_factor: int = 2, max_len: int = 4096,
                 use_posenc: bool = True, global_pool: str = "mean",
                 act_layer: str = "silu"):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.subsample = ConvSubsample1D(in_chans, d_model, factor=subsample_factor, kernel_size=5)
        self.use_posenc = use_posenc
        if use_posenc:
            self.posenc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            ConformerBlock1D(
                d_model=d_model, num_heads=num_heads,
                ffn_expansion=ffn_expansion, conv_kernel=conv_kernel,
                attn_drop=attn_drop, ff_drop=ff_drop, conv_drop=conv_drop,
                drop_path=dpr[i], act_layer=act_layer
            ) for i in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, L = x.shape
        h = self.subsample(x)  # (B, d_model, T)
        B, C2, T = h.shape
        h = h.transpose(1, 2)  # (B, T, d_model)

        key_pad = None
        if lengths is not None:
            ratio = L / float(T)
            t_lens = torch.clamp((lengths.float() / ratio).ceil().long(), min=1, max=T)
            key_pad = torch.ones(B, T, device=x.device, dtype=torch.bool)
            for i in range(B):
                key_pad[i, :int(t_lens[i].item())] = False

        if self.use_posenc:
            h = self.posenc(h)

        for blk in self.blocks:
            h = blk(h, key_padding_mask=key_pad)

        h = self.norm(h)

        if self.global_pool in ("mean", "avg", "gap"):
            if key_pad is None:
                out = h.mean(dim=1)
            else:
                valid = (~key_pad).float()
                out = (h * valid.unsqueeze(-1)).sum(dim=1) / (valid.sum(dim=1, keepdim=True) + 1e-6)
        else:
            out = h.mean(dim=1)

        logits = self.head(out)
        return logits


# -------------------------------
# Factory
# -------------------------------
def conformer1d_small(num_classes: int = 5, in_chans: int = 1,
                      d_model: int = 144, depth: int = 6, num_heads: int = 6,
                      subsample_factor: int = 2, ff_drop: float = 0.1,
                      conv_drop: float = 0.1, drop_path_rate: float = 0.1,
                      conv_kernel: int = 15, ffn_expansion: float = 4.0,
                      attn_drop: float = 0.0, max_len: int = 4096,
                      use_posenc: bool = True, global_pool: str = "mean",
                      act_layer: str = "silu") -> Conformer1D:
    return Conformer1D(
        in_chans=in_chans, num_classes=num_classes, d_model=d_model, depth=depth,
        num_heads=num_heads, ffn_expansion=ffn_expansion, conv_kernel=conv_kernel,
        attn_drop=attn_drop, ff_drop=ff_drop, conv_drop=conv_drop,
        drop_path_rate=drop_path_rate, subsample_factor=subsample_factor,
        max_len=max_len, use_posenc=use_posenc, global_pool=global_pool,
        act_layer=act_layer
    )


def conformer1d_tiny(num_classes: int = 5, in_chans: int = 1,
                     act_layer: str = "silu") -> Conformer1D:
    return conformer1d_small(num_classes=num_classes, in_chans=in_chans,
                             d_model=96, depth=4, num_heads=4, subsample_factor=2,
                             ff_drop=0.1, conv_drop=0.1, drop_path_rate=0.05,
                             act_layer=act_layer)