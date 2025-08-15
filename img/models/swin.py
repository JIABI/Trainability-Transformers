from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# -----------------------------
# Utils: DropPath (stochastic depth)
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
# Patch Embedding (Conv stem)
# -----------------------------
class PatchEmbed(nn.Module):
    """Image to patch embedding via Conv2d"""

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        # x: (B, 3, H, W) -> (B, C, H/ps, W/ps)
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        x = self.norm(x)
        return x, H, W


# -----------------------------
# Window partition / reverse
# -----------------------------
def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    x: (B, H, W, C) -> (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, nWh, nWw, ws, ws, C)
    x = x.view(-1, window_size, window_size, C)
    return x


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    windows: (num_windows*B, ws, ws, C) -> (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


# -----------------------------
# Relative position bias for windows
# -----------------------------
class RelativePositionBias(nn.Module):
    def __init__(self, window_size: Tuple[int, int], num_heads: int):
        super().__init__()
        ws_h, ws_w = window_size
        self.num_heads = num_heads
        self.window_size = (ws_h, ws_w)
        # (2*ws_h-1)*(2*ws_w-1), per-head table
        self.bias_table = nn.Parameter(torch.zeros((2 * ws_h - 1) * (2 * ws_w - 1), num_heads))
        # pairwise index
        coords_h = torch.arange(ws_h)
        coords_w = torch.arange(ws_w)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, ws_h, ws_w)
        coords_flat = torch.flatten(coords, 1)  # (2, ws_h*ws_w)
        rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, N, N)
        rel_coords = rel_coords.permute(1, 2, 0).contiguous()  # (N, N, 2)
        rel_coords[:, :, 0] += ws_h - 1
        rel_coords[:, :, 1] += ws_w - 1
        rel_coords[:, :, 0] *= 2 * ws_w - 1
        relative_position_index = rel_coords.sum(-1)  # (N, N)
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.bias_table, std=.02)

    def forward(self) -> torch.Tensor:
        # return (num_heads, N, N)
        idx = self.relative_position_index.view(-1)
        bias = self.bias_table[idx].view(self.window_size[0] * self.window_size[1],
                                         self.window_size[0] * self.window_size[1],
                                         self.num_heads)  # (N, N, H)
        return bias.permute(2, 0, 1).contiguous()


# -----------------------------
# Window Multi-Head Self-Attention
# -----------------------------
class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int,
                 qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rel_pos_bias = RelativePositionBias((window_size, window_size), num_heads)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        x: (B*num_windows, N=ws*ws, C)
        mask: (num_windows, N, N) or None
        """
        Bnw, N, C = x.shape
        qkv = self.qkv(x).reshape(Bnw, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, Bnw, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (Bnw, heads, N, N)

        # relative position bias
        bias = self.rel_pos_bias().unsqueeze(0)  # (1, heads, N, N)
        attn = attn + bias

        if mask is not None:
            # mask: (num_windows, N, N)
            nw = mask.shape[0]
            attn = attn.view(-1, nw, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)  # broadcast
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(Bnw, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# -----------------------------
# MLP
# -----------------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x);
        x = self.act(x);
        x = self.drop(x)
        x = self.fc2(x);
        x = self.drop(x)
        return x


# -----------------------------
# Swin Block (with shift)
# -----------------------------
class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution: Tuple[int, int],
                 num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size if min(input_resolution) > window_size else 0
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads,
                                    qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

        self.register_buffer("attn_mask", None, persistent=False)

    def build_attn_mask(self, H: int, W: int, device):
        if self.shift_size == 0:
            self.attn_mask = None
            return
        img_mask = torch.zeros((1, H, W, 1), device=device)  # (1,H,W,1)
        ws = self.window_size
        ss = self.shift_size
        # partition windows into 3x3 regions w.r.t shift
        cnt = 0
        for h in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
            for w in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, ws).view(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        self.attn_mask = attn_mask  # (num_windows, N, N)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        x: (B, H*W, C)
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        ws = self.window_size
        # pad if needed
        pad_b = (ws - H % ws) % ws
        pad_r = (ws - W % ws) % ws
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))  # (B, H', W', C)
        Hp, Wp = x.shape[1], x.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # windows
        x_windows = window_partition(x, ws).view(-1, ws * ws, C)

        # mask
        if self.attn_mask is None or self.attn_mask.shape[0] != x_windows.shape[0]:
            self.build_attn_mask(Hp, Wp, x_windows.device)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        x = attn_windows.view(-1, ws, ws, C)
        x = window_reverse(x, ws, Hp, Wp)  # (B, Hp, Wp, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        # remove padding
        if pad_b > 0 or pad_r > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# -----------------------------

# Patch Merging (downsample H,W by 2; dim x2)

# -----------------------------

class PatchMerging(nn.Module):

    def __init__(self, input_resolution: Tuple[int, int], dim: int):
        super().__init__()

        self.input_resolution = input_resolution

        self.dim = dim

        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        B, L, C = x.shape

        assert L == H * W, "input has wrong size"

        x = x.view(B, H, W, C)

        # pad if odd

        pad_b = H % 2

        pad_r = W % 2

        if pad_b or pad_r:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))  # (B, H', W', C)

            H = x.shape[1];
            W = x.shape[2]

        x0 = x[:, 0::2, 0::2, :]

        x1 = x[:, 1::2, 0::2, :]

        x2 = x[:, 0::2, 1::2, :]

        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)

        x = x.view(B, -1, 4 * C)

        x = self.norm(x)

        x = self.reduction(x)  # (B, H/2*W/2, 2C)

        H, W = H // 2, W // 2

        return x, H, W


# -----------------------------

# Stage: multiple SwinBlocks + optional merging

# -----------------------------

class SwinStage(nn.Module):

    def __init__(self, dim, depth, input_resolution, num_heads, window_size,

                 mlp_ratio, qkv_bias, drop, attn_drop, drop_path, downsample: bool):

        super().__init__()

        self.blocks = nn.ModuleList()

        for i in range(depth):
            self.blocks.append(SwinBlock(

                dim=dim, input_resolution=input_resolution,

                num_heads=num_heads, window_size=window_size,

                shift_size=0 if (i % 2 == 0) else window_size // 2,

                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,

                drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path

            ))

        self.down = PatchMerging(input_resolution, dim) if downsample else None

        self.downsample = downsample

        self.input_resolution = input_resolution

        self.dim = dim

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int, int]:

        for blk in self.blocks:
            x = blk(x, H, W)

        out_dim = self.dim

        if self.down is not None:
            x, H, W = self.down(x, H, W)

            out_dim = self.dim * 2

        return x, H, W, out_dim


# -----------------------------

# Swin Transformer (Classifier)

# -----------------------------

class SwinTransformer(nn.Module):

    def __init__(self,

                 img_size=224, patch_size=4, in_chans=3, num_classes=1000,

                 embed_dim=96, depths: List[int] = [2, 2, 6, 2],

                 num_heads: List[int] = [3, 6, 12, 24],

                 window_size=7, mlp_ratio=4., qkv_bias=True,

                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,

                 global_pool: str = "mean",

                 disable_patch_merging: bool = False):

        super().__init__()

        self.num_classes = num_classes

        self.num_layers = len(depths)

        self.embed_dim = embed_dim

        self.global_pool = global_pool

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        # stochastic depth decay rule

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # stages

        self.stages = nn.ModuleList()

        H = W = None  # dynamic from input

        dims = []

        in_dim = embed_dim

        cursor = 0

        for i in range(self.num_layers):
            depth_i = depths[i]

            heads_i = num_heads[i]

            downsample = (i < self.num_layers - 1) and (not disable_patch_merging)

            # input resolution is dynamic; we store placeholder, blocks build masks on first forward

            stage = SwinStage(

                dim=in_dim, depth=depth_i, input_resolution=(0, 0),  # (0,0) placeholder

                num_heads=heads_i, window_size=window_size,

                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,

                drop=drop_rate, attn_drop=attn_drop_rate,

                drop_path=dpr[cursor: cursor + depth_i], downsample=downsample

            )

            self.stages.append(stage)

            dims.append(in_dim)

            cursor += depth_i

            in_dim = in_dim * 2 if downsample else in_dim  # update for next stage

        self.norm = nn.LayerNorm(in_dim)

        self.head = nn.Linear(in_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:

        x, H, W = self.patch_embed(x)  # (B, L, C0), H,W after patch stem

        # update stages' input_resolution lazily on first forward

        for st in self.stages:

            # set resolution placeholders if needed

            if st.blocks and (st.blocks[0].input_resolution == (0, 0)):

                for blk in st.blocks:
                    blk.input_resolution = (H, W)

            x, H, W, out_dim = st(x, H, W)

        x = self.norm(x)  # (B, L_last, C_last)

        # global pooling (no cls token)

        if self.global_pool in ("mean", "gap", "avg", "avgpool", "cls"):  # 'cls' mapped to mean

            x = x.mean(dim=1)

        else:

            x = x.mean(dim=1)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.forward_features(x)

        x = self.head(x)

        return x


# -----------------------------

# Factory functions

# -----------------------------

def swin_tiny(img_size=224, num_classes=1000, window_size=7,

              drop_path_rate=0.1, global_pool="mean",

              disable_patch_merging: bool = False):
    """

    Swin-Tiny default: embed_dim=96, depths=[2,2,6,2], heads=[3,6,12,24], patch_size=4

    """

    return SwinTransformer(

        img_size=img_size, patch_size=4, in_chans=3, num_classes=num_classes,

        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],

        window_size=window_size, mlp_ratio=4.0, qkv_bias=True,

        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=drop_path_rate,

        global_pool=global_pool, disable_patch_merging=disable_patch_merging

    )


def swin_tiny_nomerge(img_size=224, num_classes=1000, window_size=7,

                      drop_path_rate=0.1, global_pool="mean"):
    """

    Ablation: same channels across stages (no patch merging).

    Uses the same block depths as Swin-T.

    """

    # We emulate "no merging" by disabling it; dims stay at 96 across stages.

    return SwinTransformer(

        img_size=img_size, patch_size=4, in_chans=3, num_classes=num_classes,

        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 3, 3, 3],  # heads adjusted for constant dim

        window_size=window_size, mlp_ratio=4.0, qkv_bias=True,

        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=drop_path_rate,

        global_pool=global_pool, disable_patch_merging=True

    )
