# models/s4.py
# Minimal S4D-style 1D classifier for ECG/time-series:
# - Shared diagonal SSM states (N timescales) -> per-channel causal kernels (sum of exponentials)
# - Efficient grouped 1D causal convolution (groups = d_model)
# - Residual + LN + MLP blocks; global mean pool -> logits
#
# Input:  x (B, C=1, L)
# Output: logits (B, num_classes)

# -----------------------------
# Utils
# -----------------------------
from __future__ import annotations
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utils
# -----------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rnd = x.new_empty(shape).bernoulli_(keep)
        return x * rnd.div(keep)


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
        return x + self.pe[:T, :].to(x.dtype).to(x.device)


# -----------------------------
# MLP
# -----------------------------
class Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0,
                 drop: float = 0.0, act_layer: str = "gelu"):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = get_activation(act_layer)
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# -----------------------------
# S4D kernel generator
# -----------------------------
class S4DKernel(nn.Module):
    def __init__(self, d_model: int, state_dim: int = 64, dt_init: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.log_tau = nn.Parameter(torch.randn(state_dim) * 0.0 + math.log(1.0))
        self.mix = nn.Parameter(torch.randn(d_model, state_dim) * 0.02)
        self.D = nn.Parameter(torch.ones(d_model))
        self.log_dt = nn.Parameter(torch.log(torch.tensor(dt_init)))
        self._cache_T = None
        self._cache_base = None

    def _base_exponent_table(self, T: int, device, dtype=torch.float32) -> torch.Tensor:
        need_refresh = (self._cache_T != T) or (self._cache_base is None) \
                       or (self._cache_base.device != device)
        if need_refresh:
            with torch.no_grad():
                tau = F.softplus(self.log_tau).to(device=device, dtype=torch.float32)
                dt = F.softplus(self.log_dt).to(device=device, dtype=torch.float32)
                t = torch.arange(T, device=device, dtype=torch.float32)
                E = torch.exp(-tau.unsqueeze(1) * dt * t.unsqueeze(0))
                self._cache_T = T
                self._cache_base = E
        return self._cache_base

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        B, C, T = u.shape
        assert C == self.d_model
        base = self._base_exponent_table(T, device=u.device, dtype=torch.float32)
        K = torch.matmul(self.mix.float(), base)  # (C,T)
        w = K.flip(-1).unsqueeze(1)  # (C,1,T)
        y = F.conv1d(u, w.to(dtype=u.dtype), padding=T - 1, groups=C)[..., :T]
        y = y + (self.D.to(dtype=u.dtype).unsqueeze(0).unsqueeze(-1) * u)
        return y


# -----------------------------
# S4D Block
# -----------------------------
class S4DBlock(nn.Module):
    def __init__(self, d_model: int, state_dim: int = 64,
                 drop: float = 0.1, drop_path: float = 0.0,
                 act_layer: str = "gelu"):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.s4 = S4DKernel(d_model=d_model, state_dim=state_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = Mlp(d_model, mlp_ratio=4.0, drop=drop, act_layer=act_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h2 = self.s4(h.transpose(1, 2)).transpose(1, 2)
        x = x + self.drop_path(h2)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# -----------------------------
# Classifier
# -----------------------------
class S4DClassifier1D(nn.Module):
    def __init__(self, in_chans: int = 1, num_classes: int = 5,
                 d_model: int = 144, depth: int = 6, state_dim: int = 64,
                 subsample_factor: int = 1, use_posenc: bool = False,
                 max_len: int = 4096, drop: float = 0.1,
                 drop_path_rate: float = 0.1, global_pool: str = "mean",
                 act_layer: str = "gelu"):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        stride = subsample_factor
        self.embed = nn.Sequential(
            nn.Conv1d(in_chans, d_model, kernel_size=5, stride=stride, padding=2, bias=False),
            get_activation(act_layer),
        )
        self.use_posenc = use_posenc
        if use_posenc:
            self.posenc = SinusoidalPositionalEncoding(d_model, max_len=max_len)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            S4DBlock(d_model=d_model, state_dim=state_dim, drop=drop,
                     drop_path=dpr[i], act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.embed(x)  # (B, C, T)
        B, C, T = h.shape
        key_pad = None
        if lengths is not None:
            t_lens = torch.clamp((lengths.float() / float(self.embed[0].stride[0])).long(),
                                 min=1, max=T)
            key_pad = torch.ones(B, T, device=x.device, dtype=torch.bool)
            for i in range(B):
                key_pad[i, :int(t_lens[i].item())] = False

        h = h.transpose(1, 2)
        if self.use_posenc:
            h = self.posenc(h)

        for blk in self.blocks:
            h = blk(h)

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


# -----------------------------
# Factories
# -----------------------------
def s4d1d_small(num_classes: int = 5, in_chans: int = 1,
                d_model: int = 144, depth: int = 6, state_dim: int = 64,
                subsample_factor: int = 1, use_posenc: bool = False,
                drop: float = 0.1, drop_path_rate: float = 0.1,
                global_pool: str = "mean", act_layer: str = "gelu") -> S4DClassifier1D:
    return S4DClassifier1D(
        in_chans=in_chans, num_classes=num_classes, d_model=d_model,
        depth=depth, state_dim=state_dim, subsample_factor=subsample_factor,
        use_posenc=use_posenc, drop=drop, drop_path_rate=drop_path_rate,
        global_pool=global_pool, act_layer=act_layer
    )


def s4d1d_tiny(num_classes: int = 5, in_chans: int = 1,
               act_layer: str = "gelu") -> S4DClassifier1D:
    return s4d1d_small(num_classes=num_classes, in_chans=in_chans,
                       d_model=96, depth=4, state_dim=32, subsample_factor=1,
                       drop=0.1, drop_path_rate=0.05, act_layer=act_layer)