
#linformer
# ReLU
python train_act.py --model linformer --dataset cifar10 --image_size 224 --epochs 30 --act relu

# SiLU
python train_act.py --model linformer --dataset cifar10 --image_size 224 --epochs 30 --act silu

#SeLu
python train_act.py --model linformer --dataset cifar10 --image_size 224 --epochs 30 --act selu
