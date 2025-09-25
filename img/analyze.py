import argparse, os, re, math, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
import torchvision.datasets as tvd

# ====== 你的模型导入 ======
from models.vit import vit_b
from models.swin import swin_tiny, swin_tiny_nomerge
from models.cvt import cvt_13, cvt_13_nopyramid
from models.linformer import linformer_b16


# ---------------- Utils ----------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset(img_size, data_root):
    tfm = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    ds = tvd.CIFAR10(root=data_root, train=False, download=True, transform=tfm)
    return ds, 10


# ---------------- Parse 模型名字 ----------------
def parse_model_from_folder(name: str):
    n = name.lower()
    m = re.search(r'_(\d{3})$', n)  # 获取 image_size
    img_size = int(m.group(1)) if m else 224

    if "vit_p16" in n:
        return dict(model="vit", patch_size=16, img_size=img_size)
    if "vit_p32" in n:
        return dict(model="vit", patch_size=32, img_size=img_size)
    if "swin_t_nomerge" in n:
        return dict(model="swin_nomerge", window_size=7, img_size=img_size)
    if "swin_t" in n:
        return dict(model="swin", window_size=7, img_size=img_size)
    if "cvt13_nopyramid" in n:
        return dict(model="cvt_nopyramid", img_size=img_size)
    if "cvt13" in n:
        return dict(model="cvt", img_size=img_size)
    if "linf_k" in n:
        k = int(re.search(r"linf_k(\d+)_", n).group(1))
        return dict(model="linformer", k_lin=k, img_size=img_size)

    raise ValueError(f"Unrecognized folder: {name}")


# ---------------- 构建模型 ----------------
def build_model(cfg, num_classes, weights, device):
    model_type = cfg["model"]

    if model_type == "vit":
        model = vit_b(
            patch_size=cfg["patch_size"],
            img_size=cfg["img_size"],
            num_classes=num_classes,
            drop_path_rate=0.1,
            global_pool="cls"
        )

    elif model_type == "swin":
        model = swin_tiny(
            img_size=cfg["img_size"],
            num_classes=num_classes,
            window_size=cfg["window_size"],
            drop_path_rate=0.1,
            global_pool="mean"
        )

    elif model_type == "swin_nomerge":
        model = swin_tiny_nomerge(
            img_size=cfg["img_size"],
            num_classes=num_classes,
            window_size=cfg["window_size"],
            drop_path_rate=0.1,
            global_pool="mean"
        )

    elif model_type == "cvt":
        model = cvt_13(
            img_size=cfg["img_size"],
            num_classes=num_classes,
            drop_path_rate=0.1,
            global_pool="mean"
        )

    elif model_type == "cvt_nopyramid":
        model = cvt_13_nopyramid(
            img_size=cfg["img_size"],
            num_classes=num_classes,
            drop_path_rate=0.1,
            global_pool="mean"
        )

    elif model_type == "linformer":
        model = linformer_b16(
            img_size=cfg["img_size"],
            num_classes=num_classes,
            drop_path_rate=0.1,
            k_lin=cfg["k_lin"]
        )
    else:
        raise ValueError(f"Unsupported model {model_type}")

    # 加载权重
    state = torch.load(weights, map_location="cpu")
    sd = state.get("state_dict", state)
    model.load_state_dict(sd, strict=False)

    return model.to(device).eval()


# ---------------- Invex Metrics ----------------
def jl_project(vecs: torch.Tensor, out_dim: int, seed: int = 1234):
    m, n = vecs.shape
    g = torch.Generator(device=vecs.device).manual_seed(seed)
    R = torch.randn(n, out_dim, generator=g, device=vecs.device) / math.sqrt(out_dim)
    return vecs @ R


def cos_max_offdiag(J: torch.Tensor):
    G = J @ J.t()
    m = G.shape[0]
    mask = ~torch.eye(m, dtype=torch.bool, device=G.device)
    vals = torch.abs(G[mask])
    return float(vals.max().item()) if vals.numel() > 0 else 0.0


def nnls_projected_grad(A: torch.Tensor, b: torch.Tensor, iters=200, lr=1e-1, reg=1e-6):
    k = A.shape[1]
    mu = torch.zeros(k, device=A.device)
    AT = A.t()
    for _ in range(iters):
        grad = 2 * (AT @ (A @ mu + b)) + 2 * reg * mu
        mu = torch.clamp(mu - lr * grad, min=0.0)
    return torch.norm(A @ mu + b, p=2).item()


@torch.no_grad()
def sv_min_from_gram(J: torch.Tensor) -> float:
    """
    返回未归一化 J 的最小奇异值：sqrt(最小特征值(J J^T))。
    仅依赖 CxD 的 Gram，稳定且与输入维度无关。
    """
    G = J @ J.t()  # (C,C)
    evals = torch.linalg.eigvalsh(G)  # 升序
    smin = torch.clamp(evals[0], min=0).sqrt().item()
    return float(smin)


def compute_metrics_for_batch(model, x, num_classes, jl_dim=None, ni_eps=1e-3, device="cuda"):
    """
    返回：
      LI_flags, NI_flags, NI_margins, COSMAX, SIGMIN_RAW, SIGMIN_NORM
    """
    model.eval()
    x = x.to(device)
    x.requires_grad_(True)
    logits = model(x)
    if logits.shape[1] != num_classes:
        num_classes = logits.shape[1]
    B, C = logits.shape

    LI_flags, NI_flags, NI_margins, COSMAX = [], [], [], []
    SIGMIN_RAW, SIGMIN_NORM = [], []

    for b in range(B):
        rows = []
        for i in range(C):
            gi = torch.autograd.grad(logits[b, i], x, retain_graph=True)[0][b].reshape(-1)
            rows.append(gi)
        J_raw = torch.stack(rows, dim=0)  # (C,D)
        # σ_min（未归一）
        smin_raw = sv_min_from_gram(J_raw)
        SIGMIN_RAW.append(smin_raw)

        # 行归一（仅用于辅助统计）
        J = J_raw.detach() / (J_raw.detach().norm(dim=1, keepdim=True) + 1e-12)
        smin_norm = sv_min_from_gram(J)
        SIGMIN_NORM.append(smin_norm)

        Jp = jl_project(J, jl_dim) if jl_dim else J

        # LI：基于奇异值比
        s = torch.linalg.svdvals(Jp)
        LI_flags.append(bool(s[-1] / (s[0] + 1e-12) > 1e-3))

        # NI：非负组合残差
        NI_ok, mres = True, float("inf")
        for i in range(C):
            A_ = torch.cat([Jp[:i], Jp[i + 1:]], dim=0).t()
            bvec = Jp[i]
            res = nnls_projected_grad(A_, -bvec)
            mres = min(mres, res)
            if res <= ni_eps:
                NI_ok = False

        NI_flags.append(NI_ok)
        NI_margins.append(mres)
        COSMAX.append(cos_max_offdiag(J))

    return (
        np.array(LI_flags, dtype=np.float32),
        np.array(NI_flags, dtype=np.float32),
        np.array(NI_margins, dtype=np.float32),
        np.array(COSMAX, dtype=np.float32),
        np.array(SIGMIN_RAW, dtype=np.float32),
        np.array(SIGMIN_NORM, dtype=np.float32),
    )


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", type=str, default="runs")
    parser.add_argument("--folders", type=str, nargs="+", required=True)
    parser.add_argument("--data_root", type=str, default="data/cifar10")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--jl_dim", type=int, default=None)
    parser.add_argument("--ni_eps", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_csv", type=str, default="image_models_metrics.csv")
    # 目录式导出（可选）
    parser.add_argument("--dump_sigma_dir", type=str, default=None,
                        help="若指定，将为每个模型输出 per-sample σ_min 到该目录。")
    parser.add_argument("--dump_all_metrics_dir", type=str, default=None,
                        help="若指定，将为每个模型输出所有 per-sample 指标到该目录。")
    parser.add_argument("--dump_imdb_like_dir", type=str, default=None,
                        help="若指定，按列(model,index,num_classes_used,p_LI,p_NI,sigma_min,residual) 为每个模型各输出一个 CSV。")
    # **方案 B：单文件合并导出**
    parser.add_argument("--dump_imdb_like_file", type=str, default=None,
                        help="若指定，把所有模型的逐样本结果合并写入同一个 CSV 文件（列：model,index,num_classes_used,p_LI,p_NI,sigma_min,residual）。")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    rows = []

    if args.dump_sigma_dir:
        os.makedirs(args.dump_sigma_dir, exist_ok=True)
    if args.dump_all_metrics_dir:
        os.makedirs(args.dump_all_metrics_dir, exist_ok=True)
    if args.dump_imdb_like_dir:
        os.makedirs(args.dump_imdb_like_dir, exist_ok=True)

    # —— 方案 B：所有模型的 imdb-like 行累积到一个列表，最后一次性写文件
    imdb_like_rows_all = [] if args.dump_imdb_like_file else None

    for folder in args.folders:
        print(f"Processing {folder} ...")
        try:
            cfg = parse_model_from_folder(folder)
            weights_path = os.path.join(args.runs_root, folder, "best_weights.pt")
            if not os.path.exists(weights_path):
                print(f"[Skip] 权重文件不存在: {weights_path}")
                continue

            ds, num_classes = build_dataset(cfg["img_size"], args.data_root)
            rng = np.random.RandomState(args.seed)
            sel = min(args.samples, len(ds))
            idx = rng.choice(len(ds), size=sel, replace=False)
            dl = DataLoader(Subset(ds, idx), batch_size=args.batch_size, shuffle=False)

            model = build_model(cfg, num_classes=num_classes, weights=weights_path, device=device)

            LI_list, NI_list, MG_list, CS_list = [], [], [], []
            SRAW_list, SNORM_list = [], []

            # 逐样本记录
            per_sample_rows = []
            offset = 0

            for x, _ in dl:
                LI, NI, MG, CS, SRAW, SNORM = compute_metrics_for_batch(
                    model, x, num_classes=num_classes, jl_dim=args.jl_dim, ni_eps=args.ni_eps, device=device
                )

                LI_list.append(LI)
                NI_list.append(NI)
                MG_list.append(MG)
                CS_list.append(CS)
                SRAW_list.append(SRAW)
                SNORM_list.append(SNORM)

                bs = len(LI)
                for j in range(bs):
                    per_sample_rows.append(dict(
                        model=folder,
                        parsed_model=cfg["model"],
                        img_size=cfg["img_size"],
                        sample_global_index=int(idx[offset + j]),
                        sample_batch_index=int(offset + j),
                        LI=int(LI[j]),
                        NI=int(NI[j]),
                        NI_margin=float(MG[j]),
                        cosmax=float(CS[j]),
                        sigma_min=float(SRAW[j]),
                        sigma_min_norm=float(SNORM[j])
                    ))
                offset += bs

            LI_all = np.concatenate(LI_list)
            NI_all = np.concatenate(NI_list)
            MG_all = np.concatenate(MG_list)
            CS_all = np.concatenate(CS_list)
            SR_all = np.concatenate(SRAW_list)  # σ_min (raw)
            SN_all = np.concatenate(SNORM_list)  # σ_min (normed)

            # 可选：导出 per-sample σ_min
            if args.dump_sigma_dir:
                df_sigma = pd.DataFrame({
                    "index": idx[:len(SR_all)],
                    "sigma_min": SR_all,
                    "sigma_min_norm": SN_all,
                })
                out_sigma = os.path.join(args.dump_sigma_dir, f"{folder}_sigma.csv")
                df_sigma.to_csv(out_sigma, index=False)
                print(f"[Dump] per-sample sigma -> {out_sigma}")

            # 可选：导出 per-sample 全指标
            if args.dump_all_metrics_dir:
                df_all = pd.DataFrame(per_sample_rows)
                out_all = os.path.join(args.dump_all_metrics_dir, f"{folder}_per_sample_metrics.csv")
                df_all.to_csv(out_all, index=False)
                print(f"[Dump] per-sample metrics -> {out_all}")

            # imdb-like（每模型各一份）
            # 先探测 num_classes_used
            try:
                with torch.no_grad():
                    dummy = torch.zeros(1, 3, cfg["img_size"], cfg["img_size"], device=device)
                    num_classes_used = int(model(dummy).shape[1])
            except Exception:
                num_classes_used = int(num_classes)

            imdb_like_rows = []
            for r in per_sample_rows:
                imdb_like_rows.append({
                    "model": r["model"],
                    "index": int(r["sample_global_index"]),
                    "num_classes_used": num_classes_used,
                    "p_LI": int(r["LI"]),
                    "p_NI": int(r["NI"]),
                    "sigma_min": float(r["sigma_min"]),
                    "residual": float(r["NI_margin"]),
                })

            if args.dump_imdb_like_dir:
                df_imdb = pd.DataFrame(imdb_like_rows)
                out_imdb = os.path.join(args.dump_imdb_like_dir, f"{folder}_per_sample_imdb_like.csv")
                df_imdb.to_csv(out_imdb, index=False)
                print(f"[Dump] imdb-like per-sample metrics -> {out_imdb}")

            if imdb_like_rows_all is not None:
                imdb_like_rows_all.extend(imdb_like_rows)

            # 汇总行
            row = dict(
                model=folder,
                parsed_model=cfg["model"],
                img_size=cfg["img_size"],
                samples=int(LI_all.size),
                LI_rate=float(LI_all.mean()),
                NI_rate=float(NI_all.mean()),
                NI_margin_mean=float(MG_all.mean()),
                NI_margin_median=float(np.median(MG_all)),
                cosmax_mean=float(CS_all.mean()),
                sigma_min_mean=float(SR_all.mean()),
                sigma_min_median=float(np.median(SR_all)),
                sigma_min_norm_mean=float(SN_all.mean()),
                sigma_min_norm_median=float(np.median(SN_all)),
            )
            rows.append(row)

            print(f"[{folder}] LI={row['LI_rate']:.3f} NI={row['NI_rate']:.3f} "
                  f"margin={row['NI_margin_mean']:.4f} "
                  f"smin={row['sigma_min_mean']:.4f} (raw)")

        except Exception as e:
            print(f"[Skip] 构建模型失败 {folder}: {e}")

    # 写汇总
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"Saved CSV to: {args.out_csv}")

    # 方案 B：写单个合并文件
    if imdb_like_rows_all is not None:
        df_all_imdb = pd.DataFrame(imdb_like_rows_all)
        out_file = args.dump_imdb_like_file
        # 确保目录存在
        out_dir = os.path.dirname(out_file)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        df_all_imdb.to_csv(out_file, index=False)
        print(f"[Dump] imdb-like ALL models per-sample metrics -> {out_file}")


if __name__ == "__main__":
    main()