import os, argparse, random, time, math, csv

from typing import Dict, List, Tuple, Optional

import numpy as np

from PIL import Image

import torch

import torch.nn.functional as F

from torch import nn

import torchvision.transforms as T

from torchvision.transforms.functional import InterpolationMode

from torch.utils.data import Dataset, DataLoader

# ========= Vision models you already have =========

try:

    from models.vit import vit_b16

    HAS_VIT = True

except Exception:

    HAS_VIT = False

try:

    from models.linformer import linformer_b16

    HAS_LINF = True

except Exception:

    HAS_LINF = False

# ========= Signal models =========

try:

    from models.conformer import conformer1d_small

    HAS_CONF = True

except Exception:

    HAS_CONF = False

try:

    from models.s4 import s4d1d_small

    HAS_S4 = True

except Exception:

    HAS_S4 = False

# ========= NLP models =========

try:

    from nlp.models.performer import performer_text_small

    HAS_PERF = True

except Exception:

    HAS_PERF = False

try:

    from nlp.models.reformer import reformer_text_small

    HAS_REFM = True

except Exception:

    HAS_REFM = False

# ========= Tokenizer for IMDb =========

try:

    from transformers import AutoTokenizer

    HAS_TOK = True

except Exception:

    HAS_TOK = False

# ========= Normalization =========

IMAGENET_MEAN = (0.485, 0.456, 0.406)

IMAGENET_STD = (0.229, 0.224, 0.225)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)

CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# ========= Utils =========

def set_seed(seed: int = 42):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


def _strip_module_prefix(sd: dict) -> dict:
    out = {}

    for k, v in sd.items():

        out[k[7:]] = v if not k.startswith("module.") else v

        if not k.startswith("module."):
            out[k] = v

    # if both written, overwrite module-less

    real = {}

    for k, v in sd.items():
        real[k[7:]] = v if k.startswith("module.") else v

    return real if real else out


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
            sd.pop("cls_token", None)

        return sd

    if have + 1 == want:
        sd = dict(sd)

        cls = sd.get("cls_token", torch.zeros(1, 1, sd["pos_embed"].shape[2]))

        sd["pos_embed"] = torch.cat([cls, sd["pos_embed"]], dim=1)

        return sd

    raise ValueError(f"pos_embed length mismatch: ckpt={have}, model={want}. "

                     f"Set --image_size to training value.")


# ========= Datasets =========

class CocoMultiLabelLite(Dataset):

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


class CIFAR10OnlyImages(Dataset):

    def __init__(self, root: str, split: str, image_size: int):
        from torchvision import datasets

        train = (split == "train")

        tf = T.Compose([

            T.Resize(image_size, interpolation=InterpolationMode.BICUBIC),

            (T.RandomResizedCrop(image_size, scale=(0.8, 1.0)) if train else T.CenterCrop(image_size)),

            T.RandomHorizontalFlip(p=0.5 if train else 0.0),

            T.ToTensor(),

            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),

        ])

        self.ds = datasets.CIFAR10(root=root, train=train, transform=tf, download=True)

        self.num_classes = 10

    def __len__(self): return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]

        return x, torch.tensor(y, dtype=torch.long)


class ECG5000Lite(Dataset):
    """UCR ECG5000：每行 label(1..5) + 140 点。这里只做评测/分析，不做 augmentation。"""

    def __init__(self, root: str, split: str = "test", z_norm: bool = True):

        base = os.path.join(root, "ECG5000")

        fn = os.path.join(base, f"ECG5000_{'TRAIN' if split == 'train' else 'TEST'}.txt")

        def _load(path):

            import numpy as np

            try:
                arr = np.loadtxt(path, delimiter=",")

            except Exception:
                arr = np.loadtxt(path)

            return arr

        arr = _load(fn).astype(np.float32)

        y = arr[:, 0].astype(np.int64) - 1

        x = arr[:, 1:]

        if z_norm:
            m = x.mean(axis=1, keepdims=True);
            s = x.std(axis=1, keepdims=True) + 1e-6

            x = (x - m) / s

        self.x = torch.from_numpy(x)[:, None, :]  # (N,1,140)

        self.y = torch.from_numpy(y)

        self.num_classes = 5

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class IMDBLite(Dataset):
    """IMDb：用 HF tokenizer；仅用于分析（不做增强）。"""

    def __init__(self, split: str = "test", max_len: int = 1024, tokenizer_name: str = "bert-base-uncased"):
        assert HAS_TOK, "transformers 未安装：pip install transformers"

        from datasets import load_dataset

        ds = load_dataset("imdb", split=split)

        self.texts = ds["text"];
        self.labels = ds["label"]

        self.num_classes = 2

        self.tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

        self.max_len = max_len

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, padding="max_length",

                       max_length=self.max_len, return_tensors="pt")

        input_ids = enc["input_ids"][0]

        attention_mask = enc["attention_mask"][0]

        label = torch.tensor(self.labels[i], dtype=torch.long)

        return (input_ids, attention_mask), label


# ========= Build model by name =========

def build_model(args, num_classes: int, modality: str):
    sd_raw = torch.load(args.weights, map_location="cpu")

    if isinstance(sd_raw, dict) and "state_dict" in sd_raw:
        sd = sd_raw["state_dict"]

    elif isinstance(sd_raw, dict) and "model" in sd_raw:
        sd = sd_raw["model"]

    else:
        sd = sd_raw

    # strip module.

    sd = _strip_module_prefix(sd)

    name = args.model.lower()

    if modality == "vision":

        if name == "vit":

            assert HAS_VIT, "models.vit 未找到"

            m = vit_b16(img_size=args.image_size, num_classes=num_classes, drop_path_rate=0.0, global_pool="cls")

            sd = _align_pos_embed_if_needed(sd, m)

        elif name == "linformer":

            assert HAS_LINF, "models.linformer 未找到"

            k_default = 256 if args.image_size >= 384 else 128

            # pool 选择：看 ckpt 的 pos_embed 是否含 cls

            pool = "mean"

            if "pos_embed" in sd:
                num_patches = (args.image_size // 16) * (args.image_size // 16)

                has_cls = (sd["pos_embed"].shape[1] == num_patches + 1)

                pool = "cls" if has_cls else "mean"

            m = linformer_b16(img_size=args.image_size, num_classes=num_classes,

                              drop_path_rate=0.0, k_lin=(args.k_lin or k_default),

                              global_pool=pool)

            sd = _align_pos_embed_if_needed(sd, m)

        else:

            raise ValueError("vision 模型仅支持 vit | linformer（可自己扩展）")

    elif modality == "signal":

        if name == "conformer":

            assert HAS_CONF, "models.conformer 未找到"

            m = conformer1d_small(num_classes=num_classes, in_chans=1)

        elif name == "s4":

            assert HAS_S4, "models.s4 未找到"

            m = s4d1d_small(num_classes=num_classes, in_chans=1)

        else:

            raise ValueError("signal 模型仅支持 conformer | s4")

    elif modality == "text":

        if name == "performer":

            assert HAS_PERF, "nlp.models.performer 未找到"

            m = performer_text_small(vocab_size=args.vocab_size or 30522, num_classes=num_classes,

                                     max_len=args.max_len)

        elif name == "reformer":

            assert HAS_REFM, "nlp.models.reformer 未找到"

            m = reformer_text_small(vocab_size=args.vocab_size or 30522, num_classes=num_classes,

                                    max_len=args.max_len)

        else:

            raise ValueError("text 模型仅支持 performer | reformer")

    else:

        raise RuntimeError("未知模态")

    missing, unexpected = m.load_state_dict(sd, strict=False)

    if missing:   print(f"[load_state_dict] Missing keys: {missing}")

    if unexpected: print(f"[load_state_dict] Unexpected keys: {unexpected}")

    m.eval()

    return m


# ========= Core: build G rows =========

@torch.no_grad()
def forward_logits(model: nn.Module, x, modality: str):
    if modality in ("vision", "signal"):

        return model(x).squeeze(0).detach().cpu()

    elif modality == "text":

        (input_ids, attention_mask) = x

        return model(input_ids, attention_mask).squeeze(0).detach().cpu()

    else:

        raise RuntimeError


def compute_G_rows_vision_signal(model: nn.Module, x: torch.Tensor, class_indices: List[int]) -> torch.Tensor:
    logits = model(x).squeeze(0)  # (C,)

    rows = []

    for i in class_indices:

        model.zero_grad(set_to_none=True)

        if x.grad is not None: x.grad.zero_()

        g, = torch.autograd.grad(logits[i], x, retain_graph=True, create_graph=False, allow_unused=False)

        rows.append(g.detach().flatten().cpu())

    return torch.stack(rows, dim=0)


def compute_G_rows_text_on_embeddings(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor,

                                      class_indices: List[int]) -> torch.Tensor:
    """

    对 NLP：通过 hook 捕获 embedding 输出 E (B,N,C)，对其求梯度 \nabla_E logit_i。

    要求模型里有 attribute: model.tok (nn.Embedding) 或类似名称；若无，可在你的模型里加个别名。

    """

    # 找到 embedding 层

    emb_layer = None

    for name in ["tok", "embeddings", "embed", "wte"]:

        if hasattr(model, name) and isinstance(getattr(model, name), nn.Embedding):
            emb_layer = getattr(model, name);
            break

    if emb_layer is None:
        raise RuntimeError("未找到 Embedding 层（期望属性名 tok/embeddings/embed/wte）")

    captured: Dict[str, torch.Tensor] = {}

    def hook(_m, _inp, out):

        captured["E"] = out

    h = emb_layer.register_forward_hook(hook)

    # 正常前向（内部会用到 embedding）

    logits = model(input_ids, attention_mask).squeeze(0)  # (C,)

    E = captured.get("E", None)  # (1,N,C)

    h.remove()

    assert E is not None, "未捕获到 embedding 输出"

    rows = []

    for i in class_indices:

        model.zero_grad(set_to_none=True)

        if E.grad is not None: E.grad.zero_()

        # 对中间变量 E 求梯度

        g, = torch.autograd.grad(logits[i], E, retain_graph=True, create_graph=False, allow_unused=False)

        rows.append(g.detach().flatten().cpu())

    return torch.stack(rows, dim=0)


def rank_full(G: torch.Tensor, rtol: float = 1e-4, atol: float = 1e-8) -> Tuple[bool, float]:
    S = torch.linalg.svdvals(G.double())

    if S.numel() == 0: return False, 0.0

    smax = float(S.max())

    thresh = max(atol, rtol * smax)

    r = int((S > thresh).sum())

    smin = float(S.min())

    return (r == G.shape[0]), smin


def neg_independent(G: torch.Tensor, steps: int = 200, lr: float = 0.1, delta_scale: float = 1e-4) -> Tuple[
    bool, float]:
    m = G.shape[0]

    GT = G.t().contiguous()

    mu = torch.rand(m)

    mu = F.relu(mu);
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


def pick_classes(logits: torch.Tensor, mode: str, k: int, num_classes: int, rng: np.random.Generator) -> List[int]:
    if mode == "all" or (k is None) or (k < 0) or (k >= num_classes):
        return list(range(num_classes))

    if mode == "topk":
        probs = torch.sigmoid(logits) if num_classes > 2 else F.softmax(logits, dim=-1)

        return probs.topk(k=k).indices.tolist()

    if mode == "random":
        return rng.choice(num_classes, size=min(k, num_classes), replace=False).tolist()

    raise ValueError("class_pick must be topk | random | all")


# ========= Main =========

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", type=str, required=True, choices=["coco", "cifar10", "ecg", "imdb"])

    ap.add_argument("--data_root", type=str, required=False, default="./data")

    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])

    ap.add_argument("--model", type=str, required=True,

                    choices=["vit", "linformer", "conformer", "s4", "performer", "reformer"])

    ap.add_argument("--weights", type=str, required=True)

    # vision

    ap.add_argument("--image_size", type=int, default=224)

    ap.add_argument("--k_lin", type=int, default=None)

    # text

    ap.add_argument("--max_len", type=int, default=1024)

    ap.add_argument("--tokenizer", type=str, default="bert-base-uncased")

    ap.add_argument("--vocab_size", type=int, default=None)

    # common analyze

    ap.add_argument("--num_samples", type=int, default=200)

    ap.add_argument("--topk", type=int, default=5)

    ap.add_argument("--class_pick", type=str, default="topk", choices=["topk", "random", "all"])

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--dump_csv", type=str, default=None)

    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device} | dataset={args.dataset} | model={args.model}")

    # dataset & modality

    if args.dataset == "coco":

        ds = CocoMultiLabelLite(args.data_root, "val" if args.split != "train" else "train", args.image_size)

        modality = "vision"

        collate = None

    elif args.dataset == "cifar10":

        split = "train" if args.split == "train" else "test"

        ds = CIFAR10OnlyImages(args.data_root, split, args.image_size)

        modality = "vision"

        collate = None

    elif args.dataset == "ecg":

        split = "train" if args.split == "train" else "test"

        ds = ECG5000Lite(args.data_root, split=split, z_norm=True)

        modality = "signal"

        collate = None

    elif args.dataset == "imdb":

        assert HAS_TOK, "需要 transformers/datasets：pip install transformers datasets"

        split = "train" if args.split == "train" else "test"

        ds = IMDBLite(split=split, max_len=args.max_len, tokenizer_name=args.tokenizer)

        modality = "text"

        collate = None

        if args.vocab_size is None:
            # 取 tokenizer 的 vocab_size

            args.vocab_size = ds.tok.vocab_size

    else:

        raise RuntimeError

    num_classes = getattr(ds, "num_classes", None)

    assert num_classes is not None, "dataset 必须包含 num_classes"

    # model

    model = build_model(args, num_classes=num_classes, modality=modality).to(device).float()

    model.eval()

    total = len(ds)

    sel = min(args.num_samples, total)

    rng = np.random.default_rng(args.seed)

    idxs = rng.choice(total, size=sel, replace=False)

    li_ok = ni_ok = 0

    smin_list: List[float] = []

    residual_list: List[float] = []

    csv_file = None;
    csv_writer = None

    if args.dump_csv:
        os.makedirs(os.path.dirname(args.dump_csv) or ".", exist_ok=True)

        csv_file = open(args.dump_csv, "w", newline="")

        csv_writer = csv.writer(csv_file)

        csv_writer.writerow(["index", "num_classes_used", "p_LI", "p_NI", "sigma_min", "residual"])

    t0 = time.time()

    for c, idx in enumerate(idxs, 1):

        sample = ds[int(idx)]

        if modality in ("vision", "signal"):

            x, _ = sample

            x = x.unsqueeze(0).to(device=device, dtype=torch.float32).requires_grad_(True)

            with torch.no_grad():

                logits = model(x).squeeze(0).detach().cpu()

            class_indices = pick_classes(logits, args.class_pick, args.topk, num_classes, rng)

            with torch.enable_grad():

                x.requires_grad_(True)

                G = compute_G_rows_vision_signal(model, x, class_indices)

        else:  # text

            (input_ids, attention_mask), _ = sample

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

        smin_list.append(smin);
        residual_list.append(res)

        if csv_writer is not None:
            csv_writer.writerow([int(idx), len(class_indices), int(li), int(ni), smin, res])

        if c % 10 == 0 or c == sel:
            print(f"[{c}/{sel}] p_LI={li_ok / c:.3f} | p_NI={ni_ok / c:.3f} | "

                  f"smin~{np.median(smin_list):.3e} | res~{np.median(residual_list):.3e}", flush=True)

        # 清理

        if modality in ("vision", "signal"):

            del x, G

        else:

            del input_ids, attention_mask, G

        if device.type == "cuda":
            torch.cuda.empty_cache()

    if csv_file is not None:
        csv_file.close()

    elapsed = (time.time() - t0) / 60.0

    print("=" * 60)

    print(f"Samples: {sel} | Dataset: {args.dataset} | Model: {args.model} | Class-pick: {args.class_pick} | "

          f"Classes per-sample: {'all' if args.class_pick == 'all' or args.topk < 0 or args.topk >= num_classes else args.topk}")

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



