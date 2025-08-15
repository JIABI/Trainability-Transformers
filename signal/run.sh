# S4D: N 扫 + 多种子

for N in 32 64 128; do

  for S in 0 1 2; do

    python -u train.py --model s4 --data_root ./data --batch_size 256 --epochs 40 --lr 3e-4 \
      --s4_d_model 144 --s4_depth 6 --s4_state_dim $N --s4_subsample 1 \
      --amp_dtype bf16 --seed $S --save_dir runs/ecg_s4d_n${N}_s${S}

    python -u analyze_invexity.py --dataset ecg --data_root data --model s4 \
      --weights runs/ecg_s4d_n${N}_s${S}/best_s4.pt \
      --num_samples 200 --class_pick all \
      --dump_csv results/s4d_n${N}_s${S}.csv
  done

done

# Conformer: kernel/head 扫（示例）

for K in 9 15 31; do

  for H in 4 6; do

    python -u train.py --model conformer --data_root ./data --batch_size 256 --epochs 40 --lr 3e-4 \
      --conf_d_model 144 --conf_depth 6 --conf_heads $H --conf_subsample 2 \
      --amp_dtype bf16 --seed 0 --save_dir runs/ecg_conf_k${K}_h${H}

    python -u analyze_invexity.py --dataset ecg --data_root data \
    --model conformer --weights runs/ecg_conf_k${K}_h${H}/best_conformer.pt \
    --num_samples 200 --class_pick all --dump_csv results/conf_k${K}_h${H}.csv

  done

done



