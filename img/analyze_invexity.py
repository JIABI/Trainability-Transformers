# analyze_invexity.py
import os, json, math, argparse, random, time
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple
import csv
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from models.vit import vit_b16
try:
    from models.linformer import linformer_b16

    HAS_LINFORMER = True
except Exception:
    HAS_LINFORMER = False

# ====== COCO-lite multi-label dataset (val/test only needs images) ======

IMAGENET_MEAN = (0.485, 0.456, 0.406)

IMAGENET_STD = (0.229, 0.224, 0.225)


class CocoMultiLabelLite(torch.utils.data.Dataset):
    def __init__(self, data_root: str, split: str, image_size: int, cat2idx: Dict[int, int] | None = None):
        from pycocotools.coco import COCO
        assert split in ["train", "val"]
        ann = os.path.join(data_root, "annotations", f"instances_{'train2017' if split == 'train' else 'val2017'}.json")
        self.coco = COCO(ann)
        self.img_dir = os.path.join(data_root, f"{split}2017")
        if cat2idx is None:
            cat_ids = sorted(self.coco.getCatIds())
            self.cat2idx = {cid: i for i, cid in enumerate(cat_ids)}
        else:
            self.cat2idx = cat2idx
        self.num_classes = len(self.cat2idx)
        self.samples = []
        for img_id in self.coco.getImgIds():
            info = self.coco.loadImgs(img_id)[0]
            fn = os.path.join(self.img_dir, info["file_name"])
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)
            labels = np.zeros(self.num_classes, dtype=np.float32)
            for a in anns:
                if a["category_id"] in self.cat2idx:
                    labels[self.cat2idx[a["category_id"]]] = 1.0
            self.samples.append((fn, labels))
        self.t = T.Compose([
            T.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        x = Image.open(p).convert("RGB")
        return self.t(x), torch.from_numpy(y)


# ====== utils ======
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _strip_module_prefix(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out


def _align_pos_embed_if_needed(sd: dict, model) -> dict:
    if "pos_embed" not in sd or not hasattr(model, "pos_embed"):
        return sd
    have = sd["pos_embed"].shape[1]
    want = model.pos_embed.shape[1]
    if have == want:
        return sd
    if have == want + 1:
        sd = dict(sd)
        sd["pos_embed"] = sd["pos_embed"][:, 1:, :]
        if "cls_token" in sd:
            sd.pop("cls_token")
        return sd
    if have + 1 == want:
        sd = dict(sd)
        cls = sd.get("cls_token", torch.zeros(1, 1, sd["pos_embed"].shape[2]))
        sd["pos_embed"] = torch.cat([cls, sd["pos_embed"]], dim=1)
        return sd
    raise ValueError(
        f"pos_embed length mismatch: ckpt={have}, model={want}. "
        f"Please set --image_size to the training value."
    )


def build_model(args, num_classes: int):
    sd_raw = torch.load(args.weights, map_location="cpu")
    if isinstance(sd_raw, dict) and "state_dict" in sd_raw:
        sd = sd_raw["state_dict"]
    elif isinstance(sd_raw, dict) and "model" in sd_raw:
        sd = sd_raw["model"]
    else:
        sd = sd_raw
    sd = _strip_module_prefix(sd)

    if args.model == "vit":
        m = vit_b16(img_size=args.image_size, num_classes=num_classes,
                    drop_path_rate=0.0, global_pool="cls")
    elif args.model == "linformer":
        assert HAS_LINFORMER, "linformer.py not found or failed to import"
        k_default = 256 if args.image_size >= 384 else 128
        if "pos_embed" in sd:
            num_patches = (args.image_size // 16) * (args.image_size // 16)
            has_cls_ckpt = (sd["pos_embed"].shape[1] == num_patches + 1)
            pool = "cls" if has_cls_ckpt else "mean"
        else:
            pool = "mean"
        m = linformer_b16(img_size=args.image_size, num_classes=num_classes,
                          drop_path_rate=0.0, k_lin=(args.k_lin or k_default),
                          global_pool=pool)
    else:
        raise ValueError("model must be one of: vit | linformer")

    sd = _align_pos_embed_if_needed(sd, m)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing:
        print(f"[load_state_dict] Missing keys (ok if aux): {missing}")
    if unexpected:
        print(f"[load_state_dict] Unexpected keys (ignored): {unexpected}")
    m.eval()
    return m


def compute_G_rows(model: torch.nn.Module, x: torch.Tensor, class_indices: List[int]) -> torch.Tensor:
    logits = model(x).squeeze(0)  # (C,)
    rows = []
    for i in class_indices:
        model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()
        g, = torch.autograd.grad(logits[i], x, retain_graph=True, create_graph=False, allow_unused=False)
        rows.append(g.detach().flatten().cpu())
    G = torch.stack(rows, dim=0)  # (m_sel, n)
    return G


def rank_full(G: torch.Tensor, rtol: float = 1e-4, atol: float = 1e-8) -> Tuple[bool, float]:
    """
    Relative-threshold full-rank test:
      full rank if s_i > max(atol, rtol * s_max)
    returns (is_full_rank, sigma_min)
    """
    S = torch.linalg.svdvals(G.double())
    if S.numel() == 0:
        return False, 0.0
    smax = float(S.max())
    thresh = max(atol, rtol * smax)
    r = int((S > thresh).sum())
    smin = float(S.min())
    return (r == G.shape[0]), smin


def neg_independent(G: torch.Tensor, steps: int = 200, lr: float = 0.1, delta_scale: float = 1e-4) -> Tuple[
    bool, float]:
    """
    Negative independence (approx) test:
      min_{mu >= 0, ||mu||_2=1} || G^T mu ||_2
      If min residual < delta => negative dependence (NOT independent).
      Else => negative independence holds.
    Returns (is_neg_independent, residual).
    """
    m = G.shape[0]
    device = torch.device("cpu")
    GT = G.t().contiguous().to(device)
    mu = torch.rand(m, device=device)
    mu = F.relu(mu);
    mu = mu / (mu.norm() + 1e-12)
    mu.requires_grad_(True)
    opt = torch.optim.SGD([mu], lr=lr, momentum=0.0)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        r = GT @ mu
        loss = 0.5 * (r @ r)
        loss.backward()
        opt.step()
        with torch.no_grad():
            mu.clamp_(min=0.0)
            nrm = mu.norm()
            if nrm > 0:
                mu.div_(nrm)
    with torch.no_grad():
        res = (GT @ mu).norm().item()
        row_norm = torch.norm(G, dim=1).mean().item() + 1e-12
        delta = delta_scale * row_norm
        is_neg_indep = res > delta
        return is_neg_indep, res


def pick_classes(logits: torch.Tensor, mode: str, k: int, num_classes: int, rng: np.random.Generator) -> List[int]:
    if mode == "all" or (k is None) or (k < 0) or (k >= num_classes):
        return list(range(num_classes))
    if mode == "topk":
        probs = torch.sigmoid(logits)
        return probs.topk(k=k).indices.tolist()
    if mode == "random":
        return rng.choice(num_classes, size=min(k, num_classes), replace=False).tolist()
    raise ValueError("class_pick must be topk | random | all")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--split", type=str, default="val", choices=["train", "val"])
    ap.add_argument("--model", type=str, required=True, choices=["vit", "linformer"])
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--k_lin", type=int, default=None, help="Linformer k; if None, set by image_size")
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--topk", type=int, default=5, help="classes per-sample; -1 / large uses all classes")
    ap.add_argument("--class_pick", type=str, default="topk", choices=["topk", "random", "all"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dump_csv", type=str, default=None, help="Path to CSV to dump per-sample metrics")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = CocoMultiLabelLite(args.data_root, args.split, args.image_size)
    num_classes = ds.num_classes
    model = build_model(args, num_classes=num_classes).to(device).float()
    model.eval()

    total = len(ds)
    sel = min(args.num_samples, total)
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(total, size=sel, replace=False)

    li_ok = 0
    ni_ok = 0
    smin_list: List[float] = []
    residual_list: List[float] = []

    csv_file = None
    csv_writer = None
    if args.dump_csv:
        os.makedirs(os.path.dirname(args.dump_csv) or ".", exist_ok=True)
        csv_file = open(args.dump_csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["index", "num_classes_used", "p_LI", "p_NI", "sigma_min", "residual"])

    t0 = time.time()
    for c, idx in enumerate(idxs, 1):
        x, _ = ds[int(idx)]
        x = x.unsqueeze(0).to(device).float().requires_grad_(True)

        with torch.no_grad():
            logits = model(x).squeeze(0).detach().cpu()

        class_indices = pick_classes(
            logits=logits, mode=args.class_pick, k=args.topk, num_classes=num_classes, rng=rng
        )

        with torch.enable_grad():
            x.requires_grad_(True)
            G = compute_G_rows(model, x, class_indices)  # (m_sel, n) on CPU

        li, smin = rank_full(G, rtol=1e-4, atol=1e-8)
        ni, res = neg_independent(G, steps=150, lr=0.2, delta_scale=1e-4)

        li_ok += int(li)
        ni_ok += int(ni)
        smin_list.append(smin)
        residual_list.append(res)

        if csv_writer is not None:
            csv_writer.writerow([int(idx), len(class_indices), int(li), int(ni), smin, res])

        if c % 10 == 0 or c == sel:
            print(f"[{c}/{sel}] p_LI={li_ok / c:.3f} | p_NI={ni_ok / c:.3f} | "
                  f"smin~{np.median(smin_list):.3e} | res~{np.median(residual_list):.3e}",
                  flush=True)

        del x, G
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if csv_file is not None:
        csv_file.close()

    elapsed = (time.time() - t0) / 60.0
    print("=" * 60)
    print(f"Samples: {sel} | Class-pick: {args.class_pick} | Classes per-sample: "
          f"{'all' if args.class_pick == 'all' or args.topk < 0 or args.topk >= num_classes else args.topk}")
    print(f"Linear Independence  p_LI = {li_ok / sel:.4f}")
    print(f"Negative Independence p_NI = {ni_ok / sel:.4f}")

    if smin_list:
        q = np.quantile(np.array(smin_list), [0.1, 0.5, 0.9])
        print(f"sigma_min quantiles (0.1/0.5/0.9): {q}")
    if residual_list:
        q = np.quantile(np.array(residual_list), [0.1, 0.5, 0.9])
        print(f"residual ||G^T mu|| quantiles (0.1/0.5/0.9): {q}")

    print(f"Time: {elapsed:.1f} min")


if __name__ == "__main__":
    main()
