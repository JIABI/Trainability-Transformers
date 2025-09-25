# conformer gelu, selu, silu, relu
python train_act.py --model conformer --act selu --epochs 40 --save_dir runs_act
python train_act.py --model conformer --act silu --epochs 40 --save_dir runs_act
python train_act.py --model conformer --act relu --epochs 40 --save_dir runs_act
python train_act.py --model conformer --act gelu --epochs 40 --save_dir runs_act

# S4 gelu, selu, silu, relu
python train_act.py --model s4 --act selu --epochs 40 --save_dir runs_act
python train_act.py --model s4 --act silu --epochs 40 --save_dir runs_act
python train_act.py --model s4 --act relu --epochs 40 --save_dir runs_act
python train_act.py --model s4 --act gelu --epochs 40 --save_dir runs_act
