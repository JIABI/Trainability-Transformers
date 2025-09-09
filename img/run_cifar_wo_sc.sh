python train_wo_sc.py \
  --models vit swin cvt linformer \
  --ablations mlp_off \
  --dataset cifar10 --image_size 224 \
  --epochs 40 --warmup_epochs 3 \
  --batch_size 128 --val_batch_size 256 \
  --lr 3e-4 --weight_decay 0.05 \
  --amp --amp_dtype bf16 \
  --zero_drop_path_when_ablate \
  --save_dir runs_ablate_cifar10

