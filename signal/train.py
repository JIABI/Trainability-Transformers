# train_ecg.py
# Train Conformer or S4D on ECG5000 (UCR). Auto-download if files missing.
import os, io, zipfile, argparse, random, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# === models ===
from models.conformer import conformer1d_small
from models.s4 import s4d1d_small


# -----------------------------
# Utils
# -----------------------------
def set_seed(s=42):
    random.seed(s);
    np.random.seed(s);
    torch.manual_seed(s);
    torch.cuda.manual_seed_all(s)


def maybe_download_ecg5000(root: str):
    """
    Download ECG5000 from UCR if not exists.
    Files expected:
      {root}/ECG5000/ECG5000_TRAIN.txt
      {root}/ECG5000/ECG5000_TEST.txt
    """
    d = os.path.join(root, "ECG5000")
    train_p = os.path.join(d, "ECG5000_TRAIN.txt")
    test_p = os.path.join(d, "ECG5000_TEST.txt")
    if os.path.exists(train_p) and os.path.exists(test_p):
        return d
    os.makedirs(d, exist_ok=True)
    url = "http://www.timeseriesclassification.com/Downloads/ECG5000.zip"
    try:
        import urllib.request
        print(f"[Download] ECG5000 from {url}")
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                if name.endswith("_TRAIN.txt") or name.endswith("_TEST.txt"):
                    with zf.open(name) as fsrc, open(os.path.join(d, os.path.basename(name)), "wb") as fdst:
                        fdst.write(fsrc.read())
        print("[Download] ECG5000 ready at:", d)
    except Exception as e:
        print("[Warn] Auto-download failed:", e)
        print("Please manually place ECG5000_TRAIN.txt and ECG5000_TEST.txt under", d)
    return d


def _load_txt(path):
    # robust loader for UCR .txt (space or comma separated)
    try:
        arr = np.loadtxt(path, delimiter=",")
    except Exception:
        arr = np.loadtxt(path)
    return arr


class ECG5000(Dataset):
    """
    Each row: label (1..5), followed by 140 values.
    Returns (x, y): x (1, 140), y in [0..4]
    """

    def __init__(self, root: str, split: str = "train", z_norm: bool = True):
        super().__init__()
        base = maybe_download_ecg5000(root)
        fn = os.path.join(base, f"ECG5000_{'TRAIN' if split == 'train' else 'TEST'}.txt")
        data = _load_txt(fn).astype(np.float32)
        y = data[:, 0].astype(np.int64) - 1  # 0..4
        x = data[:, 1:]  # (N, 140)
        if z_norm:
            m = x.mean(axis=1, keepdims=True)
            s = x.std(axis=1, keepdims=True) + 1e-6
            x = (x - m) / s
        self.x = torch.from_numpy(x)[:, None, :]  # (N,1,140)
        self.y = torch.from_numpy(y)

    def __len__(self): return self.x.shape[0]

    def __getitem__(self, i): return self.x[i], self.y[i]


# -----------------------------
# Train / Eval
# -----------------------------
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, choices=["conformer", "s4"], default="conformer")
    ap.add_argument("--data_root", type=str, default="/home/ubuntu/PycharmProjects/transformer/signal/data")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", type=str, default="runs/ecg_conformer")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    # Conformer knobs
    ap.add_argument("--conf_d_model", type=int, default=144)
    ap.add_argument("--conf_depth", type=int, default=6)
    ap.add_argument("--conf_heads", type=int, default=6)
    ap.add_argument("--conf_subsample", type=int, default=2)

    # S4 knobs
    ap.add_argument("--s4_d_model", type=int, default=144)
    ap.add_argument("--s4_depth", type=int, default=6)
    ap.add_argument("--s4_state_dim", type=int, default=64)
    ap.add_argument("--s4_subsample", type=int, default=1)

    args = ap.parse_args()
    set_seed(args.seed)

    # Data
    train_ds = ECG5000(args.data_root, split="train", z_norm=True)
    test_ds = ECG5000(args.data_root, split="test", z_norm=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=max(256, args.batch_size), shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    num_classes = 5
    if args.model == "conformer":
        model = conformer1d_small(num_classes=num_classes, in_chans=1,
                                  d_model=args.conf_d_model, depth=args.conf_depth,
                                  num_heads=args.conf_heads, subsample_factor=args.conf_subsample)
    else:
        model = s4d1d_small(num_classes=num_classes, in_chans=1,
                            d_model=args.s4_d_model, depth=args.s4_depth,
                            state_dim=args.s4_state_dim, subsample_factor=args.s4_subsample,
                            use_posenc=False)
    model = model.to(device)

    # Optim / AMP
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    use_amp = (args.amp_dtype != "fp32") and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    os.makedirs(args.save_dir, exist_ok=True)
    best = 0.0

    print("Start training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}", dynamic_ncols=True)
        run_loss = 0.0;
        seen = 0
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt);
            scaler.update()

            bs = x.size(0)
            run_loss += float(loss.item()) * bs
            seen += bs
            pbar.set_postfix(loss=loss.item())
        train_loss = run_loss / max(1, seen)

        # eval
        model.eval()
        correct = tot = 0
        with torch.no_grad():
            for x, y in tqdm(test_loader, desc="Eval", dynamic_ncols=True):
                x = x.to(device);
                y = y.to(device)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    logits = model(x)
                pred = logits.argmax(dim=1)
                correct += int((pred == y).sum().item());
                tot += int(y.numel())
        acc = correct / max(1, tot)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, test_acc={acc:.4f}")

        if acc > best:
            best = acc
            fn = os.path.join(args.save_dir, f"best_{args.model}.pt")
            torch.save({"model": model.state_dict()}, fn)
            print(f"[Saved] {fn} (acc={best:.4f})")

    print("Done. Best acc:", best)


if __name__ == "__main__":
    main()