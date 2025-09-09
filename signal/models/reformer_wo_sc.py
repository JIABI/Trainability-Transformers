from __future__ import annotations
import math
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Utils ----------
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

# ---------- Residual gate (for skip ablation) ----------
class ResidualAlpha(nn.Module):
    """
    Wrap a sublayer F into: x + alpha * F(x, ...).
    alpha=1.0 -> normal residual; alpha=0.0 -> residual removed.
    """
    def __init__(self, fn, alpha: float = 1.0):
        super().__init__()
        self.fn = fn
        self.register_buffer("alpha", torch.tensor(float(alpha)))
    def forward(self, x, *args, **kwargs):
        return x + self.alpha * self.fn(x, *args, **kwargs)

# ---------- Blockwise (LSH) Attention ----------
def masked_softmax(logits: torch.Tensor, key_mask: torch.Tensor, dim: int = -1):
    # logits: (B,Bk,H,b,b), key_mask: (B,Bk,b) with True for valid
    large_neg = torch.finfo(logits.dtype).min / 4
    # expand mask to (..., 1, 1, 1, b) then broadcast on last dim
    mask_k = (~key_mask).unsqueeze(2).unsqueeze(2)  # (B,Bk,1,1,b), True where padding
    logits = logits.masked_fill(mask_k, large_neg)
    logits = logits - logits.amax(dim=dim, keepdim=True)
    return logits.softmax(dim=dim)

class LSHBlockAttention(nn.Module):
    """
    Approximate Reformer LSH attention:
      1) Hash tokens into n_buckets via random projection + argmax (content-based).
      2) Sort tokens by bucket id; split into equal blocks of size bucket_size.
      3) Self-attend within each block (O(#blocks * bucket_size^2)).
    """
    def __init__(self, dim: int, num_heads: int = 8, bucket_size: int = 64,
                 n_hashes: int = 1, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.bucket_size = int(bucket_size)
        self.n_hashes = int(max(1, n_hashes))

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.o = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.register_buffer("hash_proj", None, persistent=False)

    def _maybe_init_proj(self, C: int, n_buckets: int, device):
        if (self.hash_proj is None) or (self.hash_proj.shape != (C, n_buckets)) or (self.hash_proj.device != device):
            with torch.no_grad():
                self.hash_proj = torch.randn(C, n_buckets, device=device)  # (C, n_buckets)

    @staticmethod
    def _split_heads(t: torch.Tensor, num_heads: int) -> torch.Tensor:
        # (B,N,C) -> (B,N,H,Dh)
        B, N, C = t.shape
        Dh = C // num_heads
        return t.view(B, N, num_heads, Dh)

    @staticmethod
    def _merge_heads(t: torch.Tensor) -> torch.Tensor:
        # (B,N,H,Dh) -> (B,N,C)
        B, N, H, Dh = t.shape
        return t.contiguous().view(B, N, H * Dh)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, N, C)
        attn_mask: (B, N) with 1/True for valid, 0/False for pad (optional)
        """
        B, N, C = x.shape
        device = x.device

        # 1) Hashing (content-based buckets)
        x_norm = F.layer_norm(x, (C,))
        n_buckets = max(1, math.ceil(N / self.bucket_size))
        self._maybe_init_proj(C, n_buckets, device)
        hash_scores = x_norm @ self.hash_proj  # (B,N,n_buckets)
        bucket_ids = hash_scores.argmax(dim=-1)  # (B,N) in [0..n_buckets-1]

        # stable sort by (bucket_id, position)
        pos_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        sort_keys = bucket_ids.to(torch.int64) * (N + 1) + pos_idx
        perm = sort_keys.argsort(dim=-1)  # (B,N)
        inv_perm = torch.empty_like(perm)
        inv_perm.scatter_(1, perm, torch.arange(N, device=device).unsqueeze(0).expand_as(perm))

        # project q/k/v and permute
        q = self._split_heads(self.q(x), self.num_heads)
        k = self._split_heads(self.k(x), self.num_heads)
        v = self._split_heads(self.v(x), self.num_heads)
        idx = perm.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_heads, self.head_dim)
        q = q.gather(1, idx); k = k.gather(1, idx); v = v.gather(1, idx)

        # sort mask
        if attn_mask is None:
            attn_mask_sorted = torch.ones(B, N, device=device, dtype=torch.bool)
        else:
            attn_mask_sorted = attn_mask.gather(1, perm).bool()

        # 2) Pad to multiple of bucket_size and reshape into blocks
        bsz = self.bucket_size
        n_blocks = math.ceil(N / bsz)
        pad = n_blocks * bsz - N
        if pad > 0:
            pad_q = torch.zeros(B, pad, self.num_heads, self.head_dim, device=device, dtype=q.dtype)
            pad_mask = torch.zeros(B, pad, device=device, dtype=torch.bool)
            q = torch.cat([q, pad_q], dim=1)
            k = torch.cat([k, pad_q], dim=1)
            v = torch.cat([v, pad_q], dim=1)
            attn_mask_sorted = torch.cat([attn_mask_sorted, pad_mask], dim=1)

        q = q.view(B, n_blocks, bsz, self.num_heads, self.head_dim)  # (B,Bk,b,H,Dh)
        k = k.view(B, n_blocks, bsz, self.num_heads, self.head_dim)
        v = v.view(B, n_blocks, bsz, self.num_heads, self.head_dim)
        key_mask = attn_mask_sorted.view(B, n_blocks, bsz)  # (B,Bk,b)

        # 3) Within-block attention
        logits = torch.einsum('btqhd,btkhd->bthqk', q, k) / math.sqrt(self.head_dim)  # (B,Bk,H,b,b)
        attn = masked_softmax(logits, key_mask)  # (B,Bk,H,b,b)
        attn = self.attn_drop(attn)
        ctx = torch.einsum('bthqk,btkhd->btqhd', attn, v)  # (B,Bk,b,H,Dh)

        # merge, unpad, invert permutation
        ctx = ctx.contiguous().view(B, n_blocks * bsz, self.num_heads, self.head_dim)
        if pad > 0:
            ctx = ctx[:, :N, :, :]
        ctx = ctx.gather(1, inv_perm.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_heads, self.head_dim))
        out = self._merge_heads(ctx)  # (B,N,C)
        out = self.o(out)
        out = self.proj_drop(out)
        return out

# ---------- Reformer Block ----------
class ReformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, bucket_size: int, n_hashes: int,
                 mlp_ratio: float = 4.0, drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0, qkv_bias: bool = True,
                 # NEW: residual gates
                 alpha_attn: float = 1.0, alpha_mlp: float = 1.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LSHBlockAttention(dim, num_heads, bucket_size, n_hashes,
                                      qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio=mlp_ratio, drop=drop)

        # wrap with residual gates (match Pre-LN + DropPath)
        self.resid_attn = ResidualAlpha(
            fn=lambda y, mask=None: self.drop_path(self.attn(self.norm1(y), mask)),
            alpha=alpha_attn
        )
        self.resid_mlp = ResidualAlpha(
            fn=lambda y: self.drop_path(self.mlp(self.norm2(y))),
            alpha=alpha_mlp
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = self.resid_attn(x, attn_mask)  # x + alpha_attn * DropPath(Attn(LN(x)))
        x = self.resid_mlp(x)              # x + alpha_mlp  * DropPath(MLP(LN(x)))
        return x

# ---------- Text Reformer (Classifier) ----------
class TextReformer(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 num_classes: int = 2,
                 max_len: int = 1024,
                 embed_dim: int = 384,
                 depth: int = 8,
                 num_heads: int = 6,
                 bucket_size: int = 64,
                 n_hashes: int = 1,
                 mlp_ratio: float = 4.0,
                 drop_rate: float = 0.1,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.1,
                 use_cls_token: bool = True,
                 # NEW: residual gates
                 alpha_attn: float = 1.0,
                 alpha_mlp: float = 1.0):
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
            ReformerBlock(embed_dim, num_heads, bucket_size, n_hashes,
                          mlp_ratio=mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate,
                          drop_path=dpr[i], qkv_bias=True,
                          alpha_attn=alpha_attn, alpha_mlp=alpha_mlp)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        input_ids: (B,N); attention_mask: (B,N) with 1 for valid
        """
        B, N = input_ids.shape
        x = self.tok(input_ids)  # (B,N,C)
        if self.use_cls_token:
            cls = self.cls.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            pos = self.pos[:, :N + 1, :]
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask.new_ones(B, 1), attention_mask], dim=1)
        else:
            pos = self.pos[:, :N, :]
        x = x + pos

        # pass raw (B,N) mask down; blocks will permute & shape it as needed
        for blk in self.blocks:
            x = blk(x, attention_mask)

        x = self.norm(x)
        if self.use_cls_token:
            x = x[:, 0]
        else:
            if attention_mask is None:
                x = x.mean(dim=1)
            else:
                x = (x * attention_mask[:, :, None]).sum(dim=1) / (attention_mask.sum(dim=1, keepdim=True) + 1e-6)
        return self.head(x)

def reformer_text_small(vocab_size: int, num_classes: int = 2, max_len: int = 1024,
                        embed_dim=384, depth=8, num_heads=6, bucket_size=64, n_hashes=1,
                        mlp_ratio=4.0, drop_rate=0.1, drop_path_rate=0.1, use_cls_token=True,
                        # NEW: residual gates
                        alpha_attn: float = 1.0, alpha_mlp: float = 1.0):
    return TextReformer(
        vocab_size=vocab_size, num_classes=num_classes, max_len=max_len,
        embed_dim=embed_dim, depth=depth, num_heads=num_heads,
        bucket_size=bucket_size, n_hashes=n_hashes,
        mlp_ratio=mlp_ratio, drop_rate=drop_rate, attn_drop_rate=0.0,
        drop_path_rate=drop_path_rate, use_cls_token=use_cls_token,
        alpha_attn=alpha_attn, alpha_mlp=alpha_mlp
    )
