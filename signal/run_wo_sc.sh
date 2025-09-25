# 仅去掉 注意力/状态空间 残差
python train_wo_sc.py --model conformer --ablation attn_off --epochs 40 --save_dir runs_ecg
python train_wo_sc.py --model s4        --ablation attn_off --epochs 40 --save_dir runs_ecg
 
# 仅去掉 MLP 残差
python train_wo_sc.py --model conformer --ablation mlp_off  --epochs 40 --save_dir runs_ecg
python train_wo_sc.py --model s4        --ablation mlp_off  --epochs 40 --save_dir runs_ecg
 
# 两个分支残差都去掉
python train_wo_sc.py --model conformer --ablation all_off  --epochs 40 --save_dir runs_ecg --zero_drop_path_when_ablate
python train_wo_sc.py --model s4        --ablation all_off  --epochs 40 --save_dir runs_ecg --zero_drop_path_when_ablate
