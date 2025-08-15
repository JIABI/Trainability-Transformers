import os, json, math, time, argparse, random, datetime

from typing import Tuple, Dict, List, Optional
from torch.utils.data import Dataset

import torch

from torchvision import datasets, transforms as T

import numpy as np

from PIL import Image

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as T

from torchvision.transforms.functional import InterpolationMode

# =========================

# Models (your local files)

# =========================

from models.vit import vit_b  # must accept patch_size
from models.swin import swin_tiny, swin_tiny_nomerge

try:

    from models.linformer import linformer_b16

    HAS_LINF = True

except Exception:

    HAS_LINF = False


from functools import partial


def to_onehot(y: int, num_classes: int) -> torch.Tensor:
    # 单标签 -> onehot（如果你用 BCEWithLogitsLoss）
    return F.one_hot(torch.tensor(y), num_classes=num_classes).float()


class OneHot:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def __call__(self, y: int):
        return F.one_hot(torch.tensor(y), num_classes=self.num_classes).float()
# ============== Utils ==============

def set_seed(seed=42):
    random.seed(seed);
    np.random.seed(seed)

    torch.manual_seed(seed);
    torch.cuda.manual_seed_all(seed)


def nowstamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


IMAGENET_MEAN = (0.485, 0.456, 0.406)

IMAGENET_STD = (0.229, 0.224, 0.225)


# ============== Datasets ==============

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)

CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class CIFAR10Dataset(Dataset):

    def __init__(self, root: str, train: bool, image_size: int):
        tf = T.Compose([

            T.Resize(image_size, interpolation=InterpolationMode.BICUBIC),

            T.RandomResizedCrop(image_size, scale=(0.8, 1.0)) if train else T.CenterCrop(image_size),

            T.RandomHorizontalFlip(p=0.5 if train else 0.0),  # ✅ 替代原来的 Lambda

            T.ToTensor(),

            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),  # 或者用你之前的 IMAGENET_MEAN/STD

        ])

        self.ds = datasets.CIFAR10(root=root, train=train, download=True, transform=tf)

        self.num_classes = 10

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]  # y: int [0..9]

        return x, torch.tensor(y, dtype=torch.long)


class CocoMultiLabel(Dataset):
    """COCO multi-label (instances_*.json) -> 80-way 0/1 vector."""

    def __init__(self, root: str, split: str, image_size: int):

        from pycocotools.coco import COCO

        assert split in ["train", "val"]

        ann = os.path.join(root, "annotations",

                           f"instances_{'train2017' if split == 'train' else 'val2017'}.json")

        self.coco = COCO(ann)

        self.img_dir = os.path.join(root, f"{split}2017")

        cat_ids = sorted(self.coco.getCatIds())

        self.cat2idx = {cid: i for i, cid in enumerate(cat_ids)}

        self.num_classes = len(self.cat2idx)

        self.samples = []

        for img_id in self.coco.getImgIds():

            info = self.coco.loadImgs(img_id)[0]

            path = os.path.join(self.img_dir, info["file_name"])

            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)

            anns = self.coco.loadAnns(ann_ids)

            y = np.zeros(self.num_classes, dtype=np.float32)

            for a in anns:

                cid = a["category_id"]

                if cid in self.cat2idx: y[self.cat2idx[cid]] = 1.0

            self.samples.append((path, y))

        self.tf_train = T.Compose([

            T.Resize(image_size, interpolation=InterpolationMode.BICUBIC),

            T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),

            T.RandomHorizontalFlip(),

            T.ToTensor(),

            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),

        ])

        self.tf_val = T.Compose([

            T.Resize(image_size, interpolation=InterpolationMode.BICUBIC),

            T.CenterCrop(image_size),

            T.ToTensor(),

            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),

        ])

        self.train = (split == "train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):

        p, y = self.samples[i]

        img = Image.open(p).convert("RGB")

        tf = self.tf_train if self.train else self.tf_val

        return tf(img), torch.from_numpy(y)


# ============== Metrics ==============

@torch.no_grad()
def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)

    return (pred == targets).float().mean().item()


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # y_true: [N], {0,1}; y_score: [N] real

    # sort by score desc

    order = np.argsort(-y_score, kind="mergesort")

    y_true = y_true[order]

    # precision at each positive

    tp = 0

    precisions = []

    for i, t in enumerate(y_true, start=1):

        if t > 0:
            tp += 1

            precisions.append(tp / i)

    if tp == 0:
        return np.nan

    return float(np.mean(precisions))


@torch.no_grad()
def multilabel_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold=0.5):
    # logits/targets: [N,C]

    probs = torch.sigmoid(logits).cpu().numpy()

    y = targets.cpu().numpy()

    # mAP

    C = y.shape[1]

    APs = []

    for c in range(C):

        ap = _average_precision(y[:, c].astype(np.int32), probs[:, c])

        if not np.isnan(ap): APs.append(ap)

    mAP = float(np.mean(APs)) if APs else 0.0

    # F1 (micro/macro)

    preds = (probs >= threshold).astype(np.float32)

    tp = (preds * y).sum()

    fp = (preds * (1 - y)).sum()

    fn = ((1 - preds) * y).sum()

    prec = tp / max(1.0, tp + fp)

    rec = tp / max(1.0, tp + fn)

    f1_micro = 2 * prec * rec / max(1e-12, (prec + rec))

    f1s = []

    for c in range(C):
        tp = (preds[:, c] * y[:, c]).sum()

        fp = (preds[:, c] * (1 - y[:, c])).sum()

        fn = ((1 - preds[:, c]) * y[:, c]).sum()

        p = tp / max(1.0, tp + fp);
        r = tp / max(1.0, tp + fn)

        f1s.append(0.0 if (p + r) == 0 else 2 * p * r / (p + r))

    f1_macro = float(np.mean(f1s))

    return {"mAP": float(mAP), "f1_micro": float(f1_micro), "f1_macro": float(f1_macro)}


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

        # cosine

        cos_e = (e - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)

        cos = 0.5 * (1 + math.cos(math.pi * cos_e))

        return [self.min_lr + (base - self.min_lr) * cos for base in self.base_lrs]


# ============== Train / Eval ==============

def build_model(args, num_classes: int):
    if args.model == "vit":

        m = vit_b(patch_size=args.patch_size, img_size=args.image_size,

                  num_classes=num_classes, drop_path_rate=args.drop_path,

                  global_pool=args.global_pool)

    elif args.model == "linformer":

        assert HAS_LINF, "linformer.py not found"

        Np = (args.image_size // 16) * (args.image_size // 16)

        k_auto = int(round(Np * args.k_ratio)) if args.k_ratio is not None else None

        k_eff = args.k_lin if args.k_lin is not None else (
            k_auto if k_auto is not None else (128 if args.image_size == 224 else 256))

        print(f"[Linformer] N={Np}, k={k_eff}, pool={args.global_pool}")

        m = linformer_b16(img_size=args.image_size, num_classes=num_classes,

                          drop_path_rate=args.drop_path, k_lin=k_eff, global_pool=args.global_pool)
    elif args.model == "swin":
        if args.disable_merge:
            m = swin_tiny_nomerge(img_size=args.image_size, num_classes=num_classes,
                                      window_size=args.window_size, drop_path_rate=args.drop_path,
                                      global_pool=args.global_pool)
        else:
            m = swin_tiny(img_size=args.image_size, num_classes=num_classes,
                              window_size=args.window_size, drop_path_rate=args.drop_path,
                              global_pool=args.global_pool)
    elif args.model == "cvt":
        from models.cvt import cvt_13, cvt_13_nopyramid
        if args.disable_pyramid:
            m = cvt_13_nopyramid(img_size=args.image_size, num_classes=num_classes,
                                     drop_path_rate=args.drop_path, global_pool=args.global_pool)
        else:
            m = cvt_13(img_size=args.image_size, num_classes=num_classes,
                           drop_path_rate=args.drop_path, global_pool=args.global_pool)
    else:

        raise ValueError("Unknown model")

    return m


def get_loaders(args):
    if args.dataset == "cifar10":

        root = args.data_root or "./data/cifar10"

        train_ds = CIFAR10Dataset(root, train=True, image_size=args.image_size)

        val_ds = CIFAR10Dataset(root, train=False, image_size=args.image_size)

        num_classes = 10

    elif args.dataset == "coco":

        assert args.data_root, "--data_root required for COCO"

        train_ds = CocoMultiLabel(args.data_root, "train", args.image_size)

        val_ds = CocoMultiLabel(args.data_root, "val", args.image_size)

        num_classes = train_ds.num_classes

        # 存类映射，便于后续分析

        if args.save_dir:
            with open(os.path.join(args.save_dir, "cat2idx.json"), "w") as f:
                json.dump(train_ds.cat2idx, f)

    else:

        raise ValueError("dataset must be cifar10 or coco")

    import multiprocessing as mp

    ctx = mp.get_context("spawn")

    pin = args.pin_memory and args.num_workers > 0

    persist = (not args.no_persist) and args.num_workers > 0

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,

                              num_workers=args.num_workers, pin_memory=pin,

                              persistent_workers=persist, drop_last=False,

                              multiprocessing_context=ctx)

    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False,

                            num_workers=args.num_workers, pin_memory=pin,

                            persistent_workers=persist, drop_last=False,

                            multiprocessing_context=ctx)

    return train_loader, val_loader, num_classes


def train_one_epoch(model, loader, optimizer, scaler, device, loss_fn,
                    dataset_kind: str, accum_steps=1, amp=True, amp_dtype="bf16",
                    max_norm=None, progress=True, log_interval=50):
    model.train()
    running = 0.0;
    n_samples = 0
    optimizer.zero_grad(set_to_none=True)

    iterator = enumerate(loader)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=len(loader), desc="Train", dynamic_ncols=True)
        except Exception:
            pass

    autocast_dtype = (torch.bfloat16 if amp_dtype == "bf16" else torch.float16)
    for step, (images, targets) in iterator:
        images = images.to(device, non_blocking=True)
        if dataset_kind == "cifar10":  # CE (long labels)
            targets = targets.to(device, non_blocking=True)
        else:  # coco: BCE multi-hot
            targets = targets.to(device, non_blocking=True).float()

        with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype, enabled=amp):
            logits = model(images)
            loss = loss_fn(logits, targets)

        if scaler is not None:  # fp16 scale
            scaler.scale(loss / accum_steps).backward()
        else:
            (loss / accum_steps).backward()

        if (step + 1) % accum_steps == 0:
            if max_norm is not None:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if scaler is not None:
                scaler.step(optimizer);
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        bs = images.size(0)
        running += float(loss.item()) * bs
        n_samples += bs

        if progress and hasattr(iterator, "set_postfix"):
            mem = (torch.cuda.memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0.0
            iterator.set_postfix(loss=f"{loss.item():.4f}",
                                 avg=f"{running / max(1, n_samples):.4f}",
                                 lr=f"{get_lr(optimizer):.2e}",
                                 mem=f"{mem:.1f}G")

        elif (step + 1) % log_interval == 0:
            print(f"Train | step {step + 1}/{len(loader)} | loss {loss.item():.4f} | "
                  f"avg {running / max(1, n_samples):.4f} | lr {get_lr(optimizer):.2e}", flush=True)

    return running / max(1, n_samples)


@torch.no_grad()
def evaluate(model, loader, device, dataset_kind: str, amp=True, amp_dtype="bf16", progress=True):
    model.eval()
    all_logits, all_targets = [], []

    iterator = loader
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(loader, total=len(loader), desc="Eval", dynamic_ncols=True)
        except Exception:
            pass

    autocast_dtype = (torch.bfloat16 if amp_dtype == "bf16" else torch.float16)
    for images, targets in iterator:
        images = images.to(device, non_blocking=True)
        if dataset_kind == "cifar10":
            targets = targets.to(device, non_blocking=True)
        else:
            targets = targets.to(device, non_blocking=True).float()
        with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype, enabled=amp):
            logits = model(images)

        all_logits.append(logits.detach().float().cpu())
        all_targets.append(targets.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    if dataset_kind == "cifar10":
        acc = top1_accuracy(logits, targets)
        return {"acc1": acc}
    else:
        return multilabel_metrics(logits, targets)


# ============== Main ==============
def main():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--dataset", type=str, default="coco", choices=["coco", "cifar10"])
    parser.add_argument("--data_root", type=str, default=None, help="COCO root or CIFAR download dir")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--no_persist", action="store_true")

    # model
    parser.add_argument("--model", type=str, choices=["vit", "linformer", "swin", "cvt"], default="vit")
    parser.add_argument("--disable_pyramid", action="store_true")
    parser.add_argument("--patch_size", type=int, default=16, help="for ViT")
    parser.add_argument("--global_pool", type=str, default="cls", choices=["cls", "mean"])
    parser.add_argument("--drop_path", type=float, default=0.1)
    parser.add_argument("--k_lin", type=int, default=None, help="for Linformer")
    parser.add_argument("--k_ratio", type=float, default=None, help="e.g., 0.125 for N/8")
    parser.add_argument("--window_size", type=int, default=7)
    parser.add_argument("--disable_merge", action="store_true")
    # opt
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--accum_steps", type=int, default=1)

    # amp
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    # misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)

    args = parser.parse_args()
    set_seed(args.seed)

    # perf toggles
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # out dir
    tag = args.exp_name or (f"{args.dataset}_{args.model}"
                            + (f"_p{args.patch_size}" if args.model == "vit" else "")
                            + (f"_k{args.k_lin}" if args.model == "linformer" and args.k_lin is not None else "")
                            + f"_img{args.image_size}")
    out_dir = args.save_dir or os.path.join("runs", f"{tag}_{nowstamp()}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[Save to] {out_dir}")

    # data & model
    train_loader, val_loader, num_classes = get_loaders(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, num_classes).to(device)

    # loss
    if args.dataset == "cifar10":
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    # optimizer & sched
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineLR(optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.epochs, min_lr=1e-6)

    # AMP scaler: only for fp16
    scaler = None
    if args.amp and args.amp_dtype == "fp16":
        scaler = torch.cuda.amp.GradScaler()

    print("Start training...")
    best_metric = -1e9
    best_path = os.path.join(out_dir, "best_weights.pt")

    for epoch in range(args.epochs):
        # train
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, loss_fn,
                                     dataset_kind=args.dataset, accum_steps=args.accum_steps,
                                     amp=args.amp, amp_dtype=args.amp_dtype, max_norm=args.max_norm,
                                     progress=args.progress)

        # eval
        metrics = evaluate(model, val_loader, device, dataset_kind=args.dataset,
                           amp=args.amp, amp_dtype=args.amp_dtype, progress=args.progress)
        if args.dataset == "cifar10":
            cur = metrics["acc1"]
            print(
                f"Epoch {epoch + 1}/{args.epochs} | train {train_loss:.4f} | acc1 {cur:.4f} | lr {get_lr(optimizer):.2e}")
        else:
            cur = metrics["mAP"]
            print(f"Epoch {epoch + 1}/{args.epochs} | train {train_loss:.4f} | "
                  f"mAP {metrics['mAP']:.4f} | f1μ {metrics['f1_micro']:.4f} | f1M {metrics['f1_macro']:.4f} | "
                  f"lr {get_lr(optimizer):.2e}")

        # save best
        if cur > best_metric:
            best_metric = cur
            torch.save(model.state_dict(), best_path)

        # save last checkpoint (for resume)
        ckpt = {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "sched": scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "args": vars(args),
        }
        if scaler is not None:
            ckpt["scaler"] = scaler.state_dict()
        torch.save(ckpt, os.path.join(out_dir, "last.pt"))

        scheduler.step()

    print(f"Done. Best metric = {best_metric:.4f}. Saved to: {best_path}")


if __name__ == "__main__":
    # spawn for safety in some envs
    try:
        import multiprocessing as mp

        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()