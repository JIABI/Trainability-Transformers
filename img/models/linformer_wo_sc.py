import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

# -------------------------
# Residual gate (skip-connection ablation)
# -------------------------
class ResidualAlpha(torch.nn.Module):
    def __init__(self, fn, alpha: float = 1.0):
        super().__init__()
        self.fn = fn
        # buffer 便于无梯度切换、参与 to(device)、保存到 ckpt
        self.register_buffer("alpha", torch.tensor(float(alpha)))

    def forward(self, x, *args, **kwargs):
        return x + self.alpha * self.fn(x, *args, **kwargs)


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
# Patch Embedding (conv patchify, no ViT deps)
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

        # Conv2d patchify == unfold + linear proj
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):  # x: (B,3,H,W)
        x = self.proj(x)  # (B,C,H/P,W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x


# -------------------------
# Linformer Self-Attention
# -------------------------
class LinformerAttention(nn.Module):
    """
    Project K/V along sequence dimension N -> k using learnable E_k/E_v (head-shared).
    Shapes:
      q,k,v: (B, heads, N, head_dim)
      E_*:   (N, k) projects sequence length
    """

    def __init__(self, dim, num_heads, seq_len, k=None, share_kv: bool = False,
                 qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.seq_len = int(seq_len)

        # target projected length
        if k is None:
            # default: quarter of sequence length, capped
            k = min(256, max(64, self.seq_len // 4))
        self.k = int(min(self.seq_len, k))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Learnable projection matrices (sequence projection N->k). Head-shared, layer-specific.
        self.E_k = nn.Parameter(torch.empty(self.seq_len, self.k))
        if share_kv:
            self.E_v = self.E_k
        else:
            self.E_v = nn.Parameter(torch.empty(self.seq_len, self.k))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.E_k, std=0.02)
        if self.E_v is not self.E_k:
            nn.init.trunc_normal_(self.E_v, std=0.02)

    def forward(self, x):
        """
        x: (B, N, C) where N == self.seq_len
        """
        B, N, C = x.shape
        if N != self.seq_len:
            raise ValueError(f"LinformerAttention expects fixed seq_len={self.seq_len}, got N={N}.")

        qkv = self.qkv(x)  # (B, N, 3C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, N, head_dim)

        # ---------- 数值稳定：在 FP32 中做投影与注意力 ----------
        dtype_orig = q.dtype
        qf = q.float()
        kf = k.float()
        vf = v.float()
        Ek = self.E_k.float()
        Ev = self.E_v.float()

        # 序列维投影 N->k
        k_proj = torch.einsum('b h n d, n k -> b h k d', kf, Ek)  # (B, heads, k, d)
        v_proj = torch.einsum('b h n d, n k -> b h k d', vf, Ev)

        attn_logits = (qf @ k_proj.transpose(-2, -1)) * (self.scale)  # (B, heads, N, k)
        attn_logits = torch.clamp(attn_logits, min=-50.0, max=50.0)  # 保险

        attn = attn_logits.softmax(dim=-1)
        attn = self.attn_drop(attn)

        xf = attn @ v_proj  # (B, heads, N, d)
        x = xf.transpose(1, 2).reshape(B, N, C).to(dtype_orig)
        # ------------------------------------------------------

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# -------------------------
# Transformer Block (Linformer attn) — with residual gates
# -------------------------
class Block(nn.Module):
    def __init__(self, dim, num_heads, seq_len, k_lin=None, mlp_ratio=4.0, qkv_bias=True,
                 drop=0.0, attn_drop=0.0, drop_path=0.0, share_kv=False,
                 alpha_attn: float = 1.0, alpha_mlp: float = 1.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = LinformerAttention(
            dim, num_heads, seq_len, k=k_lin, share_kv=share_kv,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, mlp_ratio, drop)

        # 用 ResidualAlpha 包装两条分支（把 LN 和 DropPath 一起包进去）
        self.resid_attn = ResidualAlpha(
            fn=lambda x: self.drop_path(self.attn(self.norm1(x))),
            alpha=alpha_attn
        )
        self.resid_mlp = ResidualAlpha(
            fn=lambda x: self.drop_path(self.mlp(self.norm2(x))),
            alpha=alpha_mlp
        )

    def forward(self, x):
        x = self.resid_attn(x)   # x + alpha_attn * DropPath(Attn(LN(x)))
        x = self.resid_mlp(x)    # x + alpha_mlp * DropPath(MLP(LN(x)))
        return x


# -------------------------
# Linformer Classifier (image -> patches -> encoder)
# -------------------------
class LinformerClassifier(nn.Module):
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
            k_lin: Optional[int] = None,  # projected length k
            share_kv: bool = False,
            global_pool: str = "mean",  # "mean" or "cls"
            # 新增：残差门控
            alpha_attn: float = 1.0,
            alpha_mlp: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim
        self.global_pool = global_pool

        # Patchify
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # sequence length (with/without cls)
        if global_pool == "cls":
            self.has_cls = True
            self.seq_len = num_patches + 1
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.has_cls = False
            self.seq_len = num_patches

        # pos embedding matches seq_len exactly (Linformer不支持插值)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, seq_len=self.seq_len, k_lin=k_lin, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], share_kv=share_kv,
                  alpha_attn=alpha_attn, alpha_mlp=alpha_mlp)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.head = nn.Linear(embed_dim, num_classes)

        # init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.has_cls:
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

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, C)

        if self.has_cls:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)  # (B, 1+N, C)

        # Linformer：不支持位置编码插值，要求固定图像尺寸/patch数
        if x.shape[1] != self.seq_len:
            raise ValueError(f"Input sequence len {x.shape[1]} != model seq_len {self.seq_len}. "
                             f"Please keep image_size fixed.")

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.has_cls:
            feats = x[:, 0]
        else:
            feats = x.mean(dim=1)
        return feats

    def forward(self, x):
        feats = self.forward_features(x)
        logits = self.head(feats)
        return logits


def linformer_b16(
        img_size=384,
        num_classes=80,
        drop_path_rate=0.1,
        k_lin: Optional[int] = None,  # e.g., 256 for img_size=384; 128 for 224
        share_kv: bool = False,
        global_pool: str = "mean",  # "mean" or "cls"
        # 新增：残差门控
        alpha_attn: float = 1.0,
        alpha_mlp: float = 1.0,
):
    return LinformerClassifier(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=384,
        depth=8,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=drop_path_rate,
        k_lin=k_lin,
        share_kv=share_kv,
        global_pool=global_pool,
        alpha_attn=alpha_attn,
        alpha_mlp=alpha_mlp,
    )
