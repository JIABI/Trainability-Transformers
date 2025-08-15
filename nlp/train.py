# train_imdb_text.py
import os, argparse, math, random, numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# models
from models.performer import performer_text_small
from models.reformer import reformer_text_small


def set_seed(s=42):
    random.seed(s);
    np.random.seed(s);
    torch.manual_seed(s);
    torch.cuda.manual_seed_all(s)


class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, tokenizer, max_len: int = 1024):
        ds = load_dataset("imdb", split=split)
        self.labels = ds["label"]
        self.texts = ds["text"]
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, padding="max_length",
                       max_length=self.max_len, return_tensors="pt")
        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        label = torch.tensor(self.labels[i], dtype=torch.long)
        return input_ids, attention_mask, label


def get_loader(tokenizer, split, max_len, batch_size, workers):
    ds = IMDBDataset(split, tokenizer, max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                      num_workers=workers, pin_memory=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, choices=["performer", "reformer"], required=True)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", type=str, default="runs/imdb_model")
    ap.add_argument("--num_workers", type=int, default=4)

    # Performer/Reformer specific
    ap.add_argument("--embed_dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--m_features", type=int, default=128, help="Performer random feature dim")
    ap.add_argument("--bucket_size", type=int, default=64, help="Reformer LSH block size")
    ap.add_argument("--n_hashes", type=int, default=1, help="Reformer number of hashes (approx)")
    ap.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    vocab_size = tokenizer.vocab_size

    train_loader = get_loader(tokenizer, "train", args.max_len, args.batch_size, args.num_workers)
    test_loader = get_loader(tokenizer, "test", args.max_len, args.batch_size * 2, args.num_workers)

    if args.model == "performer":
        model = performer_text_small(
            vocab_size=vocab_size, num_classes=2, max_len=args.max_len,
            embed_dim=args.embed_dim, depth=args.depth, num_heads=args.heads,
            m_features=args.m_features, drop_rate=0.1, drop_path_rate=0.1
        )
    else:  # reformer
        model = reformer_text_small(
            vocab_size=vocab_size, num_classes=2, max_len=args.max_len,
            embed_dim=args.embed_dim, depth=args.depth, num_heads=args.heads,
            bucket_size=args.bucket_size, n_hashes=args.n_hashes,
            drop_rate=0.1, drop_path_rate=0.1
        )
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    use_amp = (args.amp_dtype != "fp32") and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp)

    os.makedirs(args.save_dir, exist_ok=True)
    best = 0.0

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        pbar = tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}", dynamic_ncols=True)
        running = 0.0;
        correct = total = 0
        for input_ids, attn_mask, labels in pbar:
            input_ids = input_ids.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx:
                logits = model(input_ids, attn_mask)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(opt);
            scaler.update()

            running += float(loss.item()) * input_ids.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == labels).sum().item())
            total += int(labels.numel())
            pbar.set_postfix(loss=loss.item(), acc=correct / max(1, total))
        train_loss = running / max(1, total)

        # ---- eval ----
        model.eval();
        correct = total = 0
        with torch.no_grad():
            for input_ids, attn_mask, labels in tqdm(test_loader, desc="Eval", dynamic_ncols=True):
                input_ids = input_ids.to(device);
                attn_mask = attn_mask.to(device);
                labels = labels.to(device)
                with autocast_ctx:
                    logits = model(input_ids, attn_mask)
                pred = logits.argmax(dim=1)
                correct += int((pred == labels).sum().item())
                total += int(labels.numel())
        acc = correct / max(1, total)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, test_acc={acc:.4f}")

        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "tokenizer": "bert-base-uncased"},
                       os.path.join(args.save_dir, f"best_{args.model}.pt"))
            print(f"[Saved] {args.save_dir}/best_{args.model}.pt (acc={best:.4f})")

    print(f"Done. Best {args.model} acc: {best:.4f}")


if __name__ == "__main__":
    main()