# Performer：m-sweep（64/128/256）

python -u train.py --model performer --m_features 64  --epochs 15 --batch_size 32 --amp_dtype bf16 --save_dir runs/imdb_perf_m64

python -u train.py --model performer --m_features 128 --epochs 15 --batch_size 32 --amp_dtype bf16 --save_dir runs/imdb_perf_m128

python -u train.py --model performer --m_features 256 --epochs 15 --batch_size 32 --amp_dtype bf16 --save_dir runs/imdb_perf_m256

# Reformer：bucket_size sweep（32/64/128）

python -u train.py --model reformer --bucket_size 32  --epochs 15 --batch_size 32 --amp_dtype bf16 --save_dir runs/imdb_reformer_b32

python -u train.py --model reformer --bucket_size 64  --epochs 15 --batch_size 32 --amp_dtype bf16 --save_dir runs/imdb_reformer_b64

python -u train.py --model reformer --bucket_size 128 --epochs 15 --batch_size 32 --amp_dtype bf16 --save_dir runs/imdb_reformer_b128

