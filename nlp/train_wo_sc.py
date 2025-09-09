import os, argparse, math, random, numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# ====== models (make sure paths match your repo) ======
from models.performer_wo_sc import performer_text_small
from models.reformer_wo_sc import reformer_text_small


def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, tokenizer, max_len: int = 1024):
        ds = load_dataset("imdb", split=split)
        self.labels = ds["label"]
        self.texts = ds["text"]
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        enc = self.tok(
            self.texts[i],
            truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt"
        )
        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        label = torch.tensor(self.labels[i], dtype=torch.long)
        return input_ids, attention_mask, label


def get_loader(tokenizer, split, max_len, batch_size, workers):
    ds = IMDBDataset(split, tokenizer, max_len)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=(split == "train"),
        num_workers=workers, pin_memory=True
    )


# ---------------- Build model with ablation ----------------
def map_ablation(ablation: str):
    """
    返回: alpha_attn, alpha_mlp, suffix
    alpha_* = 1.0 表示保留该分支 residual; 0.0 表示移除该分支 residual
    """
    if ablation == "attn_off":
        return 0.0, 1.0, "wo_attsc"
    elif ablation == "mlp_off":
        return 1.0, 0.0, "wo_mlpsc"
    elif ablation == "all_off":
        return 0.0, 0.0, "wo_allsc"
    else:
        raise ValueError("ablation must be one of: attn_off, mlp_off, all_off")


def build_model(
    model_name: str,
    vocab_size: int,
    num_classes: int,
    max_len: int,
    embed_dim: int,
    depth: int,
    heads: int,
    m_features: int,
    bucket_size: int,
    n_hashes: int,
    drop_rate: float,
    drop_path_rate: float,
    ablation: str,
):
    alpha_attn, alpha_mlp, suffix = map_ablation(ablation)

    # 去残差时，建议把 DropPath 关掉更稳
    dp_rate = 0.0 if (alpha_attn < 1.0 or alpha_mlp < 1.0) else drop_path_rate

    if model_name == "performer":
        model = performer_text_small(
            vocab_size=vocab_size, num_classes=num_classes, max_len=max_len,
            embed_dim=embed_dim, depth=depth, num_heads=heads, m_features=m_features,
            mlp_ratio=4.0, drop_rate=drop_rate, drop_path_rate=dp_rate, use_cls_token=True,
            alpha_attn=alpha_attn, alpha_mlp=alpha_mlp,
        )
    elif model_name == "reformer":
        model = reformer_text_small(
            vocab_size=vocab_size, num_classes=num_classes, max_len=max_len,
            embed_dim=embed_dim, depth=depth, num_heads=heads,
            bucket_size=bucket_size, n_hashes=n_hashes,
            mlp_ratio=4.0, drop_rate=drop_rate, drop_path_rate=dp_rate, use_cls_token=True,
            alpha_attn=alpha_attn, alpha_mlp=alpha_mlp,
        )
    else:
        raise ValueError("model must be performer or reformer")

    return model, suffix


# ---------------- Train / Eval ----------------
def train_one_epoch(model, loader, optimizer, scaler, device, loss_fn, autocast_ctx):
    model.train()
    running = 0.0; correct = 0; total = 0
    pbar = tqdm(loader, desc="Train", dynamic_ncols=True)
    for input_ids, attn_mask, labels in pbar:
        input_ids = input_ids.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            logits = model(input_ids, attn_mask)
            loss = loss_fn(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()

        running += float(loss.item()) * input_ids.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
        pbar.set_postfix(loss=loss.item(), acc=correct / max(1, total))
    return running / max(1, total)


@torch.no_grad()
def evaluate(model, loader, device, autocast_ctx):
    model.eval()
    correct = 0; total = 0
    for input_ids, attn_mask, labels in tqdm(loader, desc="Eval", dynamic_ncols=True):
        input_ids = input_ids.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast_ctx:
            logits = model(input_ids, attn_mask)
        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
    return correct / max(1, total)


def run_single(args, model_name, ablation, tokenizer, device):
    # data
    train_loader = get_loader(tokenizer, "train", args.max_len, args.batch_size, args.num_workers)
    test_loader  = get_loader(tokenizer, "test",  args.max_len, args.batch_size * 2, args.num_workers)

    # model
    model, suffix = build_model(
        model_name=model_name,
        vocab_size=tokenizer.vocab_size,
        num_classes=2,
        max_len=args.max_len,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads,
        m_features=args.m_features,
        bucket_size=args.bucket_size,
        n_hashes=args.n_hashes,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
        ablation=ablation,
    )
    model = model.to(device)

    # opt & amp
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    use_amp = (args.amp_dtype != "fp32") and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp)

    # saving
    out_dir = os.path.join(args.save_dir, model_name, suffix)
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, f"{model_name}_{suffix}.pt")

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, scaler, device, loss_fn, autocast_ctx)
        acc = evaluate(model, test_loader, device, autocast_ctx)
        print(f"[{model_name}/{suffix}] Epoch {epoch}/{args.epochs} | train_loss={tr_loss:.4f} | test_acc={acc:.4f}")
        if acc > best:
            best = acc
            torch.save(
                {"model": model.state_dict(), "tokenizer": "bert-base-uncased"},
                best_path
            )
            print(f"  ✅ Saved: {best_path} (acc={best:.4f})")
    return best_path, best


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", type=str, default="runs_imdb_ablate")

    # models & ablations
    ap.add_argument("--models", type=str, nargs="+", required=True,
                    choices=["performer", "reformer"])
    ap.add_argument("--ablations", type=str, nargs="+", required=True,
                    choices=["attn_off", "mlp_off", "all_off"])

    # model hyperparams
    ap.add_argument("--embed_dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--m_features", type=int, default=128)     # Performer
    ap.add_argument("--bucket_size", type=int, default=64)     # Reformer
    ap.add_argument("--n_hashes", type=int, default=1)         # Reformer
    ap.add_argument("--drop_rate", type=float, default=0.1)
    ap.add_argument("--drop_path_rate", type=float, default=0.1)

    # AMP
    ap.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

    # 如果你想固定顺序，也可以用下面这行；默认就按传入顺序循环
    # ablations_ordered = [a for a in ["attn_off", "mlp_off", "all_off"] if a in args.ablations]
    ablations_ordered = args.ablations

    for model_name in args.models:
        for ablation in ablations_ordered:
            run_single(args, model_name, ablation, tokenizer, device)


if __name__ == "__main__":
    main()
