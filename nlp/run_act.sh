python train_act.py \
    --models performer \
    --ablations all_on \
    --epochs 10 \
    --batch_size 32 \
    --act_layer relu

python train_act.py \
    --models reformer \
    --ablations all_on \
    --epochs 10 \
    --batch_size 32 \
    --act_layer relu

python train_act.py \
    --models performer \
    --ablations all_on \
    --epochs 10 \
    --batch_size 32 \
    --act_layer selu
python train_act.py \
    --models reformer \
    --ablations all_on \
    --epochs 10 \
    --batch_size 32 \
    --act_layer selu

python train_act.py \
    --models performer \
    --ablations all_on \
    --epochs 10 \
    --batch_size 32 \
    --act_layer silu
python train_act.py \
    --models reformer \
    --ablations all_on \
    --epochs 10 \
    --batch_size 32 \
    --act_layer silu

python train_act.py \
    --models performer \
    --ablations all_on \
    --epochs 10 \
    --batch_size 32 \
    --act_layer gelu
python train_act.py \
    --models reformer \
    --ablations all_on \
    --epochs 10 \
    --batch_size 32 \
    --act_layer gelu