#!/usr/bin/env python3
# train_ablation.py
import os, math, argparse, random, datetime
from typing import List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision import datasets, transforms as T
from torchvision.transforms.functional import InterpolationMode

# ============== Import models (需要 _wo_sc 版本支持 alpha_attn/alpha_mlp) ==============
from models.vit_wo_sc import vit_b
from models.swin_wo_sc import swin_tiny, swin_tiny_nomerge
try:
    from models.linformer_wo_sc import linformer_b16
    HAS_LINF = True
except Exception:
    HAS_LINF = False
from models.cvt_wo_sc import cvt_13, cvt_13_nopyramid

# ============== Utils ==============
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def nowstamp():
    import datetime as _dt
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")

def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

# ============== Datasets ==============
class CIFAR10Dataset(Dataset):
    def __init__(self, root: str, train: bool, image_size: int):
        tf = T.Compose([
            T.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0)) if train else T.CenterCrop(image_size),
            T.RandomHorizontalFlip(p=0.5 if train else 0.0),
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        self.ds = datasets.CIFAR10(root=root, train=train, download=True, transform=tf)
        self.num_classes = 10

    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        x, y = self.ds[i]
        return x, torch.tensor(y, dtype=torch.long)

# ============== Metrics ==============
@torch.no_grad()
def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item()

# ============== Warmup + Cosine ==============
class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        e = self.last_epoch + 1
        if e <= self.warmup_epochs:
            return [base * e / max(1, self.warmup_epochs) for base in self.base_lrs]
        cos_e = (e - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
        cos = 0.5 * (1 + math.cos(math.pi * cos_e))
        return [self.min_lr + (base - self.min_lr) * cos for base in self.base_lrs]

# ============== Ablation mapping ==============
def ablation_to_alphas(ablation: str):
    """
    返回 (alpha_attn, alpha_mlp, suffix)
    alpha_* = 1.0 表示保留该分支 residual；0.0 表示移除该分支 residual
    """
    if ablation == "attn_off":
        return 0.0, 1.0, "wo_attsc"
    elif ablation == "mlp_off":
        return 1.0, 0.0, "wo_mlpsc"
    elif ablation == "all_off":
        return 0.0, 0.0, "wo_allsc"
    else:
        raise ValueError("ablation must be one of: attn_off, mlp_off, all_off")

# ============== Model Builder with Ablation ==============
def build_single_model(args, model_name: str, num_classes: int,
                       alpha_attn: float, alpha_mlp: float,
                       drop_path_rate: float):
    """
    构建模型。要求各 _wo_sc 版本的 Block 接受 alpha_attn/alpha_mlp 这两个 kwarg。
    """
    common_kwargs = dict(drop_path_rate=drop_path_rate, )
    if model_name == "vit":
        m = vit_b(
            patch_size=args.patch_size, img_size=args.image_size,
            num_classes=num_classes, global_pool=args.global_pool,
            # drop-path 替换为有效值
            drop_path_rate=drop_path_rate,
            # 残差开关
            alpha_attn=alpha_attn, alpha_mlp=alpha_mlp,
        )
    elif model_name == "swin":
        swin_ctor = swin_tiny_nomerge if args.disable_merge else swin_tiny
        m = swin_ctor(
            img_size=args.image_size, num_classes=num_classes,
            window_size=args.window_size, global_pool=args.global_pool,
            drop_path_rate=drop_path_rate,
            alpha_attn=alpha_attn, alpha_mlp=alpha_mlp,
        )
    elif model_name == "cvt":
        cvt_ctor = cvt_13_nopyramid if args.disable_pyramid else cvt_13
        m = cvt_ctor(
            img_size=args.image_size, num_classes=num_classes,
            global_pool=args.global_pool, drop_path_rate=drop_path_rate,
            alpha_attn=alpha_attn, alpha_mlp=alpha_mlp,
        )
    elif model_name == "linformer":
        assert HAS_LINF, "linformer_wo_sc.py not found or import failed"
        Np = (args.image_size // 16) * (args.image_size // 16)
        k_auto = int(round(Np * args.k_ratio)) if args.k_ratio is not None else None
        k_eff = args.k_lin if args.k_lin is not None else (
            k_auto if k_auto is not None else (128 if args.image_size == 224 else 256))
        print(f"[Linformer] N={Np}, k={k_eff}, pool={args.global_pool}")
        m = linformer_b16(
            img_size=args.image_size, num_classes=num_classes,
            drop_path_rate=drop_path_rate, k_lin=k_eff, global_pool=args.global_pool,
            alpha_attn=alpha_attn, alpha_mlp=alpha_mlp,
        )
    else:
        raise ValueError("Unknown model")
    return m

# ============== Train / Eval ==============
def train_one_epoch(model, loader, optimizer, scaler, device, loss_fn,
                    accum_steps=1, amp=True, amp_dtype="bf16",
                    max_norm=None):
    model.train()
    running = 0.0; n_samples = 0
    optimizer.zero_grad(set_to_none=True)
    autocast_dtype = (torch.bfloat16 if amp_dtype == "bf16" else torch.float16)

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype, enabled=amp):
            logits = model(images)
            loss = loss_fn(logits, targets)
        if scaler is not None:
            scaler.scale(loss / accum_steps).backward()
        else:
            (loss / accum_steps).backward()
        if (step + 1) % accum_steps == 0:
            if max_norm is not None:
                if scaler is not None: scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if scaler is not None:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        bs = images.size(0)
        running += float(loss.item()) * bs
        n_samples += bs
    return running / max(1, n_samples)

@torch.no_grad()
def evaluate(model, loader, device, amp=True, amp_dtype="bf16"):
    model.eval()
    all_logits, all_targets = [], []
    autocast_dtype = (torch.bfloat16 if amp_dtype == "bf16" else torch.float16)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype, enabled=amp):
            logits = model(images)
        all_logits.append(logits.detach().float().cpu())
        all_targets.append(targets.detach().cpu())
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    acc = top1_accuracy(logits, targets)
    return {"acc1": acc}

# ============== Runner ==============
def run_single_experiment(args, model_name: str, ablation: str,
                          train_loader, val_loader, num_classes, device, save_root):
    alpha_attn, alpha_mlp, suffix = ablation_to_alphas(ablation)

    # 去残差时可选把 drop-path 置 0，更稳
    eff_dpr = 0.0 if (args.zero_drop_path_when_ablate and (alpha_attn < 1.0 or alpha_mlp < 1.0)) else args.drop_path

    model = build_single_model(args, model_name, num_classes,
                               alpha_attn, alpha_mlp, eff_dpr).to(device)
    loss_fn  = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineLR(optimizer, warmup_epochs=args.warmup_epochs,
                               max_epochs=args.epochs, min_lr=1e-6)
    scaler = torch.cuda.amp.GradScaler() if args.amp and args.amp_dtype == "fp16" else None

    out_dir = os.path.join(save_root, model_name, suffix)
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, f"{model_name}_{suffix}.pt")

    best_metric = -1e9
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, loss_fn,
                                     accum_steps=args.accum_steps, amp=args.amp, amp_dtype=args.amp_dtype,
                                     max_norm=args.max_norm)
        metrics = evaluate(model, val_loader, device, amp=args.amp, amp_dtype=args.amp_dtype)
        cur = metrics["acc1"]
        print(f"[{model_name}/{ablation}] Epoch {epoch+1}/{args.epochs} | train {train_loss:.4f} | acc1 {cur:.4f}")

        if cur > best_metric:
            best_metric = cur
            torch.save(model.state_dict(), best_path)

        scheduler.step()
    print(f"[{model_name}/{ablation}] Done. Best metric = {best_metric:.4f}. Saved: {best_path}")
    return best_path, best_metric

# ============== Main ==============
def main():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10"])
    parser.add_argument("--data_root", type=str, default="./data/cifar10")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)

    # models & ablations
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        choices=["vit","swin","cvt","linformer"])
    parser.add_argument("--ablations", type=str, nargs="+", required=True,
                        choices=["attn_off","mlp_off","all_off"])

    # model knobs
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--global_pool", type=str, default="cls", choices=["cls","mean"])
    parser.add_argument("--drop_path", type=float, default=0.1)
    parser.add_argument("--k_lin", type=int, default=None)
    parser.add_argument("--k_ratio", type=float, default=None)
    parser.add_argument("--window_size", type=int, default=7)
    parser.add_argument("--disable_merge", action="store_true")
    parser.add_argument("--disable_pyramid", action="store_true")

    # train
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--accum_steps", type=int, default=1)

    # amp
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16","fp16"])

    # misc
    parser.add_argument("--save_dir", type=str, default="runs_ablate")
    parser.add_argument("--zero_drop_path_when_ablate", action="store_true",
                        help="在做任意消融(attn_off/mlp_off/all_off)时，把 drop-path 设为 0，更稳")

    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loaders
    train_ds = CIFAR10Dataset(args.data_root, train=True, image_size=args.image_size)
    val_ds   = CIFAR10Dataset(args.data_root, train=False, image_size=args.image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # 按传入顺序循环（如需固定顺序可自行排序）
    for model_name in args.models:
        for ablation in args.ablations:
            run_single_experiment(args, model_name, ablation,
                                  train_loader, val_loader, train_ds.num_classes, device, args.save_dir)

if __name__ == "__main__":
    main()

