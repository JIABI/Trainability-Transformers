# analyze_signal_invexity.py

# analyze_signal.py

import os, re, argparse

import torch

from torch.utils.data import DataLoader

import numpy as np

import pandas as pd

from tqdm import tqdm

from models.conformer import conformer1d_small

from models.s4 import s4d1d_small

from train import ECG5000, set_seed


def parse_group(folder: str):
    """

    根据 folder 名字解析超参数:

    - Conformer: conf_k15_h4 -> kernel=15, heads=4

    - S4: s4d_n64 -> state_dim=64

    """

    if "conf" in folder.lower():

        m_k = re.search(r"k(\d+)", folder)

        m_h = re.search(r"h(\d+)", folder)

        kernel = int(m_k.group(1)) if m_k else None

        heads = int(m_h.group(1)) if m_h else None

        return "conformer", kernel, heads, None

    elif "s4" in folder.lower():

        m_n = re.search(r"n(\d+)", folder)

        state_dim = int(m_n.group(1)) if m_n else None

        return "s4", None, None, state_dim

    else:

        raise ValueError(f"无法识别模型类型: {folder}")


def build_model(model_name: str, num_classes: int, kernel=None, heads=None, state_dim=None):
    if model_name == "conformer":

        return conformer1d_small(

            num_classes=num_classes, in_chans=1,

            d_model=144, depth=6,

            num_heads=heads or 6, subsample_factor=2

        )

    elif model_name == "s4":

        return s4d1d_small(

            num_classes=num_classes, in_chans=1,

            d_model=144, depth=6,

            state_dim=state_dim or 64, subsample_factor=1,

            use_posenc=False

        )

    else:

        raise ValueError(f"Unknown model {model_name}")


def analyze(args):
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    test_ds = ECG5000(args.data_root, split="test", z_norm=True)

    test_loader = DataLoader(test_ds, batch_size=args.batch_size,

                             shuffle=False, num_workers=2, pin_memory=True)

    num_classes = 5

    all_records = []

    for folder in args.folders:

        print(f"\n[Process] {folder}")

        model_name, kernel, heads, state_dim = parse_group(folder)

        # 构建模型

        model = build_model(model_name, num_classes, kernel, heads, state_dim).to(device)

        # 加载权重

        if model_name == "conformer":

            weight_file = os.path.join(args.runs_root, folder, "best_conformer.pt")

        else:

            weight_file = os.path.join(args.runs_root, folder, "best_s4.pt")

        if not os.path.exists(weight_file):
            print(f"[Skip] 权重不存在 {weight_file}")

            continue

        ckpt = torch.load(weight_file, map_location=device)

        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        model.load_state_dict(sd, strict=False)

        model.eval()

        # 遍历数据集并保存指标

        with torch.no_grad():

            for idx, (x, y) in enumerate(test_loader):

                if idx >= args.num_samples:
                    break

                x = x.to(device)

                logits = model(x)

                # === 这里你需要替换成真正的 invexity 计算 ===

                # 我先写 dummy 值

                p_LI = 1.0

                p_NI = 1.0

                sigma_min = float(torch.rand(1))

                residual = float(torch.rand(1))

                all_records.append({

                    "model": folder,

                    "index": idx,

                    "num_classes_used": num_classes,

                    "p_LI": p_LI,

                    "p_NI": p_NI,

                    "sigma_min": sigma_min,

                    "residual": residual,

                    "source_file": f"{folder}.csv"

                })

    # 保存到 CSV

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    pd.DataFrame(all_records).to_csv(args.out_csv, index=False)

    print(f"[Done] 结果已保存到 {args.out_csv}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", type=str, required=True)

    ap.add_argument("--runs_root", type=str, required=True)

    ap.add_argument("--folders", type=str, nargs="+", required=True)

    ap.add_argument("--num_samples", type=int, default=200)

    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--out_csv", type=str, required=True)

    args = ap.parse_args()

    analyze(args)


if __name__ == "__main__":
    main()







