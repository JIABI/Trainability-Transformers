import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from pycocotools.coco import COCO

from models.vit import vit_b16
from models.linformer import linformer_b16

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # speed
    torch.backends.cudnn.deterministic = False


# ----------------------------
# COCO Multi-Label Dataset
# ----------------------------
class CocoMultiLabel(Dataset):
    def __init__(
            self,
            data_root: str,
            split: str,
            image_size: int,
            cat2idx: Dict[int, int] = None,
            build_mapping_from_ann: str = None,
            keep_empty: bool = True,
            transform=None,
    ):
        """
        data_root:
            coco root with structure:
              - train2017/
              - val2017/
              - annotations/instances_train2017.json
              - annotations/instances_val2017.json
        split: "train" or "val"
        """
        super().__init__()
        assert split in ["train", "val"]
        self.split = split
        self.data_root = data_root
        self.image_size = image_size
        self.transform = transform
        ann_file = os.path.join(
            data_root, "annotations", f"instances_{'train2017' if split == 'train' else 'val2017'}.json"
        )
        self.coco = COCO(ann_file)
        self.img_dir = os.path.join(data_root, f"{split}2017")

        # build or use provided mapping
        if cat2idx is None:
            assert build_mapping_from_ann is not None, "Need cat2idx or build_mapping_from_ann='train'/'val'"
            cat_ids = self.coco.getCatIds()
            cat_ids = sorted(cat_ids)  # stable
            self.cat2idx = {cid: i for i, cid in enumerate(cat_ids)}
        else:
            self.cat2idx = cat2idx
        self.num_classes = len(self.cat2idx)

        self.samples = []  # list of (img_path, multi_hot np.array[num_classes])
        img_ids = self.coco.getImgIds()

        for img_id in img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            file_name = img_info["file_name"]
            img_path = os.path.join(self.img_dir, file_name)

            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)

            labels = np.zeros(self.num_classes, dtype=np.float32)
            for ann in anns:
                cid = ann["category_id"]
                if cid in self.cat2idx:
                    labels[self.cat2idx[cid]] = 1.0

            if keep_empty or labels.sum() > 0:
                self.samples.append((img_path, labels))

        # basic transforms if not provided
        if self.transform is None:
            if split == "train":
                self.transform = T.Compose([
                    T.RandomResizedCrop(image_size, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ])
            else:
                self.transform = T.Compose([
                    T.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                    T.CenterCrop(image_size),
                    T.ToTensor(),
                    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, labels = self.samples[idx]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            im = self.transform(im)
        return im, torch.from_numpy(labels)


# ----------------------------
# Warmup + Cosine LR
# ----------------------------
class WarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # linear warmup to base lr
            warmup_factor = float(self.last_epoch + 1) / float(max(1, self.warmup_epochs))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        # cosine
        cos_epoch = self.last_epoch - self.warmup_epochs
        cos_total = max(1, self.max_epochs - self.warmup_epochs)
        cos_out = 0.5 * (1.0 + math.cos(math.pi * cos_epoch / cos_total))
        return [self.min_lr + (base_lr - self.min_lr) * cos_out for base_lr in self.base_lrs]


# ----------------------------
# Metrics
# ----------------------------
def sigmoid_np(x):
    return 1 / (1 + np.exp(-x))


def f1_scores(y_true: np.ndarray, y_pred_bin: np.ndarray) -> Tuple[float, float]:
    # y_* shape: (N, C)
    eps = 1e-8
    # micro
    tp = (y_true * y_pred_bin).sum()
    fp = ((1 - y_true) * y_pred_bin).sum()
    fn = (y_true * (1 - y_pred_bin)).sum()
    prec_micro = tp / (tp + fp + eps)
    rec_micro = tp / (tp + fn + eps)
    f1_micro = 2 * prec_micro * rec_micro / (prec_micro + rec_micro + eps)

    # macro
    tp_c = (y_true * y_pred_bin).sum(axis=0)
    fp_c = ((1 - y_true) * y_pred_bin).sum(axis=0)
    fn_c = (y_true * (1 - y_pred_bin)).sum(axis=0)
    prec_c = tp_c / (tp_c + fp_c + eps)
    rec_c = tp_c / (tp_c + fn_c + eps)
    f1_c = 2 * prec_c * rec_c / (prec_c + rec_c + eps)
    f1_macro = np.nanmean(f1_c)
    return float(f1_micro), float(f1_macro)


def average_precision_score_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AP for a single class, no sklearn needed. y_true in {0,1}, same length as y_score."""
    assert y_true.ndim == 1 and y_score.ndim == 1 and len(y_true) == len(y_score)
    # handle no positives
    P = y_true.sum()
    if P <= 0:
        return np.nan

    order = np.argsort(-y_score)  # descending scores
    y_true = y_true[order]

    tp = np.cumsum(y_true == 1)
    fp = np.cumsum(y_true == 0)

    recall = tp / (P + 1e-8)
    precision = tp / (tp + fp + 1e-8)

    # prepend (0,1)
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))

    # compute area under PR curve (trapezoidal)
    ap = np.trapz(precision, recall)
    return float(ap)


def mean_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # y_* shape: (N, C)
    aps = []
    for c in range(y_true.shape[1]):
        ap = average_precision_score_binary(y_true[:, c], y_score[:, c])
        if not np.isnan(ap):
            aps.append(ap)
    if len(aps) == 0:
        return float("nan")
    return float(np.mean(aps))


# ----------------------------
# Train/Eval loops
# ----------------------------
def compute_pos_weight(train_labels: np.ndarray) -> torch.Tensor:
    # pos_weight = (N - pos) / pos per class
    N = train_labels.shape[0]
    pos = train_labels.sum(axis=0)
    pos = np.clip(pos, 1.0, None)  # avoid div by zero
    pw = (N - pos) / pos
    return torch.tensor(pw, dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, scaler, device, loss_fn, accum_steps=1, amp=True, max_norm=None):
    model.train()
    running = 0.0
    n_samples = 0

    optimizer.zero_grad(set_to_none=True)
    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(images)
            loss = loss_fn(logits, targets)

        scaler.scale(loss / accum_steps).backward()

        if (step + 1) % accum_steps == 0:
            if max_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        bs = images.size(0)
        running += loss.item() * bs
        n_samples += bs

    return running / max(1, n_samples)


@torch.no_grad()
def evaluate(model, loader, device, amp=True, threshold=0.5):
    model.eval()
    all_logits = []
    all_targets = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(images)
        all_logits.append(logits.float().cpu())
        all_targets.append(targets.float().cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    probs = sigmoid_np(logits)
    preds_bin = (probs >= threshold).astype(np.float32)

    f1_micro, f1_macro = f1_scores(targets, preds_bin)
    mAP = mean_average_precision(targets, probs)
    return {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "mAP": mAP,
    }


# ----------------------------

# Main

# ----------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True, help="Path to COCO2017 root")

    parser.add_argument("--image_size", type=int, default=384)

    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--val_batch_size", type=int, default=128)

    parser.add_argument("--epochs", type=int, default=50)

    parser.add_argument("--warmup_epochs", type=int, default=5)

    parser.add_argument("--lr", type=float, default=5e-4)

    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--drop_path", type=float, default=0.1)

    parser.add_argument("--amp", action="store_true",default="bf16")

    parser.add_argument("--accum_steps", type=int, default=1)

    parser.add_argument("--max_norm", type=float, default=None, help="Grad clip max norm")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--out_dir", type=str, default="runs/linformer_b16_coco")

    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--save_metric", type=str, default="f1_macro", choices=["f1_macro", "f1_micro", "mAP"])

    parser.add_argument("--global_pool", type=str, default="cls", choices=["cls", "mean"])
    parser.add_argument("--progress", action="store_true", help="Show progress bar")
    parser.add_argument("--log_interval", type=int, default=50, help="Log interval")


    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Build train dataset first to get mapping & class imbalance

    train_ds_tmp = CocoMultiLabel(

        data_root=args.data_root,

        split="train",

        image_size=args.image_size,

        cat2idx=None,

        build_mapping_from_ann="train",

        keep_empty=True,

        transform=None,  # will setup internally

    )

    cat2idx = train_ds_tmp.cat2idx

    num_classes = train_ds_tmp.num_classes

    # Persist mapping to reuse

    with open(os.path.join(args.out_dir, "cat2idx.json"), "w") as f:

        json.dump(cat2idx, f, indent=2)

    # Build datasets with same mapping

    train_ds = train_ds_tmp  # already built

    val_ds = CocoMultiLabel(

        data_root=args.data_root,

        split="val",

        image_size=args.image_size,

        cat2idx=cat2idx,

        keep_empty=True,

        transform=None,

    )

    # Precompute pos_weight for BCE (class imbalance)

    train_labels = np.stack([lbl for _, lbl in train_ds.samples], axis=0)

    pos_weight = compute_pos_weight(train_labels)

    train_loader = DataLoader(

        train_ds,

        batch_size=args.batch_size,

        shuffle=True,

        num_workers=min(8, os.cpu_count() or 4),

        pin_memory=True,

        drop_last=False,

        persistent_workers=True,

    )

    val_loader = DataLoader(

        val_ds,

        batch_size=args.val_batch_size,

        shuffle=False,

        num_workers=min(8, os.cpu_count() or 4),

        pin_memory=True,

        drop_last=False,

        persistent_workers=True,

    )

    # Model

    #model = vit_b16(img_size=args.image_size, num_classes=num_classes, drop_path_rate=args.drop_path,

    #                global_pool=args.global_pool)
    model = linformer_b16(img_size=args.image_size, num_classes=num_classes,
                          drop_path_rate=args.drop_path,
                          k_lin=(256 if args.image_size >= 384 else 128),
                          global_pool=args.global_pool).to(device)

    model.to(device)

    # Optimizer & LR

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = WarmupCosineLR(optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.epochs, min_lr=1e-6)

    # Loss

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    # AMP scaler

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch = 0

    best_metric = -1.0

    best_path = os.path.join(args.out_dir, "best.pt")

    last_path = os.path.join(args.out_dir, "last.pt")

    # Resume

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")

        model.load_state_dict(ckpt["model"])

        optimizer.load_state_dict(ckpt["optim"])

        scheduler.load_state_dict(ckpt["sched"])

        scaler.load_state_dict(ckpt["scaler"])

        start_epoch = ckpt.get("epoch", 0) + 1

        best_metric = ckpt.get("best_metric", -1.0)

        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # Train

    log_file = os.path.join(args.out_dir, "log.txt")

    with open(log_file, "a") as lf:

        print("Start training...")

        for epoch in range(start_epoch, args.epochs):

            t0 = time.time()

            train_loss = train_one_epoch(

                model, train_loader, optimizer, scaler, device, loss_fn,

                accum_steps=args.accum_steps, amp=args.amp, max_norm=args.max_norm

            )

            scheduler.step()

            metrics = evaluate(model, val_loader, device, amp=args.amp, threshold=0.5)

            elapsed = time.time() - t0

            metric_now = metrics[args.save_metric]

            is_best = metric_now > best_metric

            if is_best:
                best_metric = metric_now

                torch.save(model.state_dict(), best_path.replace(".pt", "_weights.pt"))

            # save full checkpoint (for resume)

            ckpt = {

                "epoch": epoch,

                "model": model.state_dict(),

                "optim": optimizer.state_dict(),

                "sched": scheduler.state_dict(),

                "scaler": scaler.state_dict(),

                "best_metric": best_metric,

                "args": vars(args),

                "cat2idx": cat2idx,

            }

            torch.save(ckpt, last_path)

            log_str = (f"Epoch {epoch:03d} | TrainLoss {train_loss:.4f} | "

                       f"F1_micro {metrics['f1_micro']:.4f} | F1_macro {metrics['f1_macro']:.4f} | "

                       f"mAP {metrics['mAP']:.4f} | Time {elapsed / 60:.1f} min | Best({args.save_metric}) {best_metric:.4f}")

            print(log_str)

            lf.write(log_str + "\n")

            lf.flush()

    print(f"Done. Best {args.save_metric}: {best_metric:.4f}")

    print(f"Best weights: {best_path.replace('.pt', '_weights.pt')}")

    print(f"Last checkpoint: {last_path}")


if __name__ == "__main__":
    main()
