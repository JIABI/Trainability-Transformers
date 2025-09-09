# Both ablations in sequence, saving two .pt files per model

python train_wo_sc.py \
  --models performer reformer \
  --ablations mlp_off \
  --epochs 15 --batch_size 32 --amp_dtype bf16