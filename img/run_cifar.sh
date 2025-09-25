#!/usr/bin/env bash
set -euo pipefail

# ==== Config ====
CIFAR_ROOT=${CIFAR_ROOT:-/data/cifar10}
IMG_SIZE=${IMG_SIZE:-128}
EPOCHS=${EPOCHS:-40}
BATCH=${BATCH:-32}
VAL_BATCH=${VAL_BATCH:-64}
WORKERS=${WORKERS:-4}

# ==== 0) Prepare CIFAR-10 ====
echo "[Setup] Ensuring torchvision + CIFAR-10 at ${CIFAR_ROOT}"
python - <<'PY'
import os
try:
    import torchvision # noqa
except Exception:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "torchvision"])
from torchvision import datasets
root = os.environ.get("CIFAR_ROOT", "./data/cifar10")
datasets.CIFAR10(root=root, train=True,  download=True)
datasets.CIFAR10(root=root, train=False, download=True)
print("CIFAR-10 ready at:", root)
PY

# ==== 1) ViT: patch 16 vs 32 ====
echo "[Train] ViT p16"
python -u train_coco_cifar.py --dataset cifar10 --data_root data/cifar10 --image_size ${IMG_SIZE} \
  --model vit --patch_size 16 \
  --batch_size ${BATCH} --val_batch_size ${VAL_BATCH} \
  --epochs ${EPOCHS} --warmup_epochs 5 \
  --lr 3e-4 --weight_decay 0.05 --drop_path 0.1 \
  --amp --amp_dtype bf16 --max_norm 1.0 \
  --num_workers ${WORKERS} --pin_memory --progress \
  --save_dir runs/cifar_vit_p16_${IMG_SIZE}

echo "[Train] ViT p32"
python -u train_coco_cifar.py --dataset cifar10 --data_root data/cifar10 --image_size ${IMG_SIZE} \
  --model vit --patch_size 32 \
  --batch_size ${BATCH} --val_batch_size ${VAL_BATCH} \
  --epochs ${EPOCHS} --warmup_epochs 5 \
  --lr 3e-4 --weight_decay 0.05 --drop_path 0.1 \
  --amp --amp_dtype bf16 --max_norm 1.0 \
  --num_workers ${WORKERS} --pin_memory --progress \
  --save_dir runs/cifar_vit_p32_${IMG_SIZE}

# ==== 2) Swin: with vs without patch merging ====
echo "[Train] Swin-T (with patch merging)"
python -u train_coco_cifar.py --dataset cifar10 --data_root data/cifar10 --image_size ${IMG_SIZE} \
  --model swin --window_size 7 \
  --batch_size ${BATCH} --val_batch_size ${VAL_BATCH} \
  --epochs ${EPOCHS} --warmup_epochs 5 \
  --lr 3e-4 --weight_decay 0.05 --drop_path 0.1 --global_pool mean \
  --amp --amp_dtype bf16 --max_norm 1.0 \
  --num_workers ${WORKERS} --pin_memory --progress \
  --save_dir runs/cifar_swin_t_${IMG_SIZE}

echo "[Train] Swin-T (NO patch merging) ablation"
python -u train_coco_cifar.py --dataset cifar10 --data_root data/cifar10 --image_size ${IMG_SIZE} \
  --model swin --window_size 7 --disable_merge \
  --batch_size ${BATCH} --val_batch_size ${VAL_BATCH} \
  --epochs ${EPOCHS} --warmup_epochs 5 \
  --lr 3e-4 --weight_decay 0.05 --drop_path 0.1 --global_pool mean \
  --amp --amp_dtype bf16 --max_norm 1.0 \
  --num_workers ${WORKERS} --pin_memory --progress \
  --save_dir runs/cifar_swin_t_nomerge_${IMG_SIZE}

# ==== 3) Linformer: k in {24,49,98} ====
for K in 24 49 98; do
  echo "[Train] Linformer k=${K}"
  python -u train_coco_cifar.py --dataset cifar10 --data_root data/cifar10 --image_size ${IMG_SIZE} \
    --model linformer --k_lin ${K} --global_pool mean \
    --batch_size ${BATCH} --val_batch_size ${VAL_BATCH} \
    --epochs ${EPOCHS} --warmup_epochs 5 \
    --lr 3e-4 --weight_decay 0.05 --drop_path 0.1 \
    --amp --amp_dtype bf16 --max_norm 1.0 \
    --num_workers ${WORKERS} --pin_memory --progress \
    --save_dir runs/cifar_linf_k${K}_${IMG_SIZE}
done

# ==== 4) CvT: pyramid on vs off ====
echo "[Train] CvT-13 (pyramid ON)"
python -u train_coco_cifar.py --dataset cifar10 --data_root data/cifar10 --image_size ${IMG_SIZE} \
  --model cvt \
  --batch_size ${BATCH} --val_batch_size ${VAL_BATCH} \
  --epochs ${EPOCHS} --warmup_epochs 5 \
  --lr 3e-4 --weight_decay 0.05 --drop_path 0.1 --global_pool mean \
  --amp --amp_dtype bf16 --max_norm 1.0 \
  --num_workers ${WORKERS} --pin_memory --progress \
  --save_dir runs/cifar_cvt13_${IMG_SIZE}

echo "[Train] CvT-13 (pyramid OFF) ablation"
python -u train_coco_cifar.py --dataset cifar10 --data_root data/cifar10 --image_size ${IMG_SIZE} \
  --model cvt --disable_pyramid \
  --batch_size ${BATCH} --val_batch_size ${VAL_BATCH} \
  --epochs ${EPOCHS} --warmup_epochs 5 \
  --lr 3e-4 --weight_decay 0.05 --drop_path 0.1 --global_pool mean \
  --amp --amp_dtype bf16 --max_norm 1.0 \
  --num_workers ${WORKERS} --pin_memory --progress \
  --save_dir runs/cifar_cvt13_nopyramid_${IMG_SIZE}

echo "[Done] All CIFAR-10 trainings finished."