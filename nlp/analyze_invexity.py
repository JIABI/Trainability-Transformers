import os, re, csv, time, random, argparse, sys
from typing import List, Tuple, Dict

# -------- 依赖导入（若缺会报清晰错误）--------
try:
    import numpy as np
except Exception as e:
    print(f"[ImportError] 需要 numpy: {e}", flush=True);
    sys.exit(2)
try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except Exception as e:
    print(f"[ImportError] 需要 PyTorch: {e}", flush=True);
    sys.exit(2)


def log(msg: str): print(msg, flush=True)


def set_seed(s=42):
    random.seed(s);
    np.random.seed(s)
    torch.manual_seed(s);
    torch.cuda.manual_seed_all(s)


# ----------------- 准备(在线下载) -----------------
def prepare_online():
    """
    一键在线下载 IMDB 数据与 bert-base-uncased 分词器到本地缓存。
    """
    try:
        from datasets import load_dataset
        log("[Prepare] 下载 IMDB(train/test) ...")
        load_dataset("imdb", split="train")
        load_dataset("imdb", split="test")
        log("[Prepare] IMDB OK")
    except Exception as e:
        raise RuntimeError("下载 IMDB 失败。请检查网络或代理。") from e

    try:
        from transformers import AutoTokenizer
        log("[Prepare] 下载 bert-base-uncased 分词器 ...")
        AutoTokenizer.from_pretrained("bert-base-uncased")
        log("[Prepare] Tokenizer OK")
    except Exception as e:
        raise RuntimeError("下载分词器失败。请检查网络或代理。") from e


# ----------------- 数据集 -----------------
class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, tokenizer, max_len: int = 1024):
        from datasets import load_dataset  # 延迟导入，便于 --prepare 时先安装/下载
        log(f"[Dataset] 加载 IMDB split={split}, max_len={max_len}")
        ds = load_dataset("imdb", split=split)
        self.labels = ds["label"]
        self.texts = ds["text"]
        self.tok = tokenizer
        self.max_len = max_len
        self.num_classes = 2

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, padding="max_length",
                       max_length=self.max_len, return_tensors="pt")
        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        label = torch.tensor(self.labels[i], dtype=torch.long)
        return (input_ids, attention_mask), label


# ------------- 从文件夹名解析配置 -------------
def parse_cfg_from_folder(name: str, default_max_len: int = 1024):
    """
    支持：
      imdb_perf_m64    -> performer, m_features=64
      imdb_perf_m128   -> performer, m_features=128
      imdb_perf_m256   -> performer, m_features=256
      imdb_reformer_b32  -> reformer, bucket_size=32
      imdb_reformer_b64  -> reformer, bucket_size=64
      imdb_reformer_b128 -> reformer, bucket_size=128
    """
    n = name.lower()
    if "perf" in n or "performer" in n:
        m = re.search(r'_m(\d+)', n)
        m_features = int(m.group(1)) if m else 128
        return dict(model="performer", m_features=m_features, max_len=default_max_len)

    if "reformer" in n:
        b = re.search(r'_b(\d+)', n)
        bucket_size = int(b.group(1)) if b else 64
        return dict(model="reformer", bucket_size=bucket_size, n_hashes=1, max_len=default_max_len)

    raise ValueError(f"无法从文件夹名解析模型类型/超参: {name}")


# ----------------- 构建模型 + 加载权重 -----------------
def build_model(cfg: dict, vocab_size: int, num_classes: int, device: torch.device, weights_dir: str):
    """
    与你的 train_imdb_text.py 对齐：
      - performer_text_small / reformer_text_small
      - embed_dim=384, depth=8, num_heads=6
      - Performer: m_features 从文件夹名解析
      - Reformer : bucket_size/n_hashes 从文件夹名解析
    自动寻找权重名：best_performer.pt / best_reformer.pt / best.pt
    """
    model_name = cfg["model"]
    if model_name == "performer":
        from models.performer import performer_text_small
        model = performer_text_small(
            vocab_size=vocab_size, num_classes=num_classes, max_len=cfg["max_len"],
            embed_dim=384, depth=8, num_heads=6,
            m_features=cfg["m_features"], drop_rate=0.1, drop_path_rate=0.1
        )
        fname = "best_performer.pt"
    elif model_name == "reformer":
        from models.reformer import reformer_text_small
        model = reformer_text_small(
            vocab_size=vocab_size, num_classes=num_classes, max_len=cfg["max_len"],
            embed_dim=384, depth=8, num_heads=6,
            bucket_size=cfg["bucket_size"], n_hashes=cfg.get("n_hashes", 1),
            drop_rate=0.1, drop_path_rate=0.1
        )
        fname = "best_reformer.pt"
    else:
        raise ValueError(f"未知模型: {model_name}")

    p1 = os.path.join(weights_dir, fname)
    p2 = os.path.join(weights_dir, "best.pt")
    ckpt_path = p1 if os.path.exists(p1) else p2 if os.path.exists(p2) else None
    if ckpt_path is None:
        raise FileNotFoundError(f"未找到权重: {p1} 或 {p2}")

    log(f"  [ckpt] loading {ckpt_path}")
    sd_raw = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd_raw, dict) and "state_dict" in sd_raw:
        sd = sd_raw["state_dict"]
    elif isinstance(sd_raw, dict) and "model" in sd_raw:
        sd = sd_raw["model"]
    else:
        sd = sd_raw

    # strip 'module.'
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=False)
    if getattr(msg, "missing_keys", None):
        log(f"  [load_state_dict] Missing {len(msg.missing_keys)} keys (show <=8): {msg.missing_keys[:8]}")
    if getattr(msg, "unexpected_keys", None):
        log(f"  [load_state_dict] Unexpected {len(msg.unexpected_keys)} keys (show <=8): {msg.unexpected_keys[:8]}")

    return model.to(device).eval()


# ----------------- Invexity 组件 -----------------
def pick_classes(logits: torch.Tensor, mode: str, k: int, num_classes: int, rng: np.random.Generator) -> List[int]:
    if mode == "all" or (k is None) or (k < 0) or (k >= num_classes):
        return list(range(num_classes))
    if mode == "topk":
        probs = torch.softmax(logits, dim=-1)
        return probs.topk(k=k).indices.tolist()
    if mode == "random":
        return rng.choice(num_classes, size=min(k, num_classes), replace=False).tolist()
    raise ValueError("class_pick must be topk | random | all")


def rank_full(G: torch.Tensor, rtol: float = 1e-4, atol: float = 1e-8) -> Tuple[bool, float]:
    S = torch.linalg.svdvals(G.double())
    if S.numel() == 0: return False, 0.0
    smax = float(S.max());
    thresh = max(atol, rtol * smax)
    r = int((S > thresh).sum());
    smin = float(S.min())
    return (r == G.shape[0]), smin


def neg_independent(G: torch.Tensor, steps: int = 200, lr: float = 0.1, delta_scale: float = 1e-4) -> Tuple[
    bool, float]:
    m = G.shape[0];
    GT = G.t().contiguous()
    mu = torch.rand(m);
    mu = torch.relu(mu);
    mu = mu / (mu.norm() + 1e-12);
    mu.requires_grad_(True)
    opt = torch.optim.SGD([mu], lr=lr, momentum=0.0)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        r = GT @ mu
        loss = 0.5 * (r @ r)
        loss.backward();
        opt.step()
        with torch.no_grad():
            mu.clamp_(min=0.0);
            nrm = mu.norm()
            if nrm > 0: mu.div_(nrm)
    with torch.no_grad():
        res = (GT @ mu).norm().item()
        row_norm = torch.norm(G, dim=1).mean().item() + 1e-12
        delta = delta_scale * row_norm
        return (res > delta), res


def compute_G_rows_text_on_embeddings(model: nn.Module,
                                      input_ids: torch.Tensor,
                                      attention_mask: torch.Tensor,
                                      class_indices: List[int]) -> torch.Tensor:
    # 1) 找 embedding 层
    emb_layer = None
    for name in ["tok", "embeddings", "embed", "wte"]:
        if hasattr(model, name) and isinstance(getattr(model, name), nn.Embedding):
            emb_layer = getattr(model, name);
            break
    if emb_layer is None:
        for m in model.modules():
            if isinstance(m, nn.Embedding):
                emb_layer = m;
                break
    if emb_layer is None:
        raise RuntimeError("未找到 nn.Embedding 层")

    # 2) hook 捕获 E
    captured: Dict[str, torch.Tensor] = {}

    def hook(_m, _inp, out):
        captured["E"] = out

    h = emb_layer.register_forward_hook(hook)

    logits = model(input_ids, attention_mask).squeeze(0)  # (C,)
    E = captured.get("E", None);
    h.remove()
    if E is None:
        raise RuntimeError("未捕获到 embedding 输出 E")

    rows = []
    for i in class_indices:
        model.zero_grad(set_to_none=True)
        if E.grad is not None: E.grad.zero_()
        g, = torch.autograd.grad(logits[i], E, retain_graph=True, create_graph=False, allow_unused=False)
        rows.append(g.detach().flatten().cpu())
    return torch.stack(rows, dim=0)


# ----------------- 主流程 -----------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--prepare", action="store_true",

                    help="在线下载 IMDB 与 bert-base-uncased 到本地缓存，然后退出。")

    ap.add_argument("--runs_root", type=str, default="runs",

                    help="训练输出根目录（包含 imdb_perf_* / imdb_reformer_* 子目录）")

    ap.add_argument("--folders", type=str, nargs="+",

                    help="要分析的子目录名，例如 imdb_perf_m64 imdb_reformer_b32")

    ap.add_argument("--max_len", type=int, default=1024)

    ap.add_argument("--samples", type=int, default=200)

    ap.add_argument("--topk", type=int, default=2)

    ap.add_argument("--class_pick", type=str, default="topk", choices=["topk", "random", "all"])

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--out_csv", type=str, default="imdb_invexity_metrics.csv")

    args = ap.parse_args()

    if args.prepare:
        log("[Mode] 准备在线下载依赖和数据 ...")

        prepare_online()

        log("[Done] 已完成下载。可以移除 --prepare 开始正式分析。")

        return

    # 运行分析

    if not args.folders:
        print("错误：未提供 --folders。例：--folders imdb_perf_m64 imdb_reformer_b32", flush=True)

        sys.exit(2)

    # 设备

    device = torch.device("cuda" if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    set_seed(args.seed)

    log(f"[Env] device={device} | runs_root={args.runs_root} | folders={len(args.folders)}")

    # 依赖：tokenizer + dataset（若首次运行未缓存，会在线下载）

    try:

        from transformers import AutoTokenizer

        log("[Tokenizer] 加载 bert-base-uncased（必要时会在线下载）")

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

    except Exception as e:

        raise RuntimeError("加载分词器失败：请先执行 `python analyze_nlp_invexity.py --prepare`。") from e

    try:

        ds = IMDBDataset("test", tokenizer, max_len=args.max_len)

    except Exception as e:

        raise RuntimeError("加载 IMDB 失败：请先执行 `python analyze_nlp_invexity.py --prepare`。") from e

    num_classes = ds.num_classes

    total = len(ds)

    sel = min(args.samples, total)

    rng = np.random.default_rng(args.seed)

    idxs = rng.choice(total, size=sel, replace=False)

    log(f"[Dataset] total={total} | selected={sel}")

    vocab_size = tokenizer.vocab_size

    # 预检 runs/ckpt

    found_any = False

    for f in args.folders:

        d = os.path.join(args.runs_root, f)

        c1 = os.path.exists(os.path.join(d, "best_performer.pt"))

        c2 = os.path.exists(os.path.join(d, "best_reformer.pt"))

        c3 = os.path.exists(os.path.join(d, "best.pt"))

        log(f"[Precheck] {f:>20} | dir={os.path.isdir(d)} | perf={c1} ref={c2} best={c3}")

        if (os.path.isdir(d) and (c1 or c2 or c3)): found_any = True

    if not found_any:
        log("[Abort] 没找到任何有效的 checkpoint。请检查 --runs_root 与 --folders。")

        sys.exit(2)

    # CSV

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    fout = open(args.out_csv, "w", newline="")

    writer = csv.writer(fout)

    writer.writerow(["model", "index", "num_classes_used", "p_LI", "p_NI", "sigma_min", "residual"])

    wrote_rows = 0

    # 主循环

    for folder in args.folders:

        run_dir = os.path.join(args.runs_root, folder)

        if not os.path.isdir(run_dir):
            log(f"[Skip] 目录不存在: {run_dir}")

            continue

        try:

            cfg = parse_cfg_from_folder(folder, default_max_len=args.max_len)

        except Exception as e:

            log(f"[Skip] 解析失败 {folder}: {e}")

            continue

        log(f"\n[Run] {folder} | cfg={cfg}")

        try:

            model = build_model(cfg, vocab_size=vocab_size, num_classes=num_classes,

                                device=device, weights_dir=run_dir)

        except Exception as e:

            log(f"[Skip] 构建/加载失败: {e}")

            continue

        li_ok = ni_ok = 0

        smins, ress = [], []

        t0 = time.time()

        for c, idx in enumerate(idxs, 1):

            (input_ids, attention_mask), _ = ds[int(idx)]

            input_ids = input_ids.unsqueeze(0).to(device)

            attention_mask = attention_mask.unsqueeze(0).to(device)

            with torch.no_grad():

                logits = model(input_ids, attention_mask).squeeze(0).detach().cpu()

            class_indices = pick_classes(logits, args.class_pick, args.topk, num_classes, rng)

            G = compute_G_rows_text_on_embeddings(model, input_ids, attention_mask, class_indices)

            li, smin = rank_full(G, rtol=1e-4, atol=1e-8)

            ni, res = neg_independent(G, steps=150, lr=0.2, delta_scale=1e-4)

            li_ok += int(li);
            ni_ok += int(ni)

            smins.append(smin);
            ress.append(res)

            writer.writerow([folder, int(idx), len(class_indices), int(li), int(ni), smin, res])

            wrote_rows += 1

            if c % 10 == 0 or c == sel:
                log(f"  [{c}/{sel}] p_LI={li_ok / c:.3f} | p_NI={ni_ok / c:.3f} | "

                    f"smin~{np.median(smins):.3e} | res~{np.median(ress):.3e}")

        dt = (time.time() - t0) / 60

        if smins:

            log(f"[Done] {folder}: LI={li_ok / sel:.4f}  NI={ni_ok / sel:.4f}  "

                f"median smin={np.median(smins):.3e}  median res={np.median(ress):.3e}  time={dt:.1f}m")

        else:

            log(f"[Done] {folder}: 没有样本被处理（上一步可能已报错）")

    fout.close()

    log(f"\nAll done. Saved: {args.out_csv} (rows={wrote_rows})")


if __name__ == "__main__":
    main()
