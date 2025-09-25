python analyze.py \
  --data_root data/cifar10 \
  --folders cifar_vit_p16_224 cifar_vit_p32_224 cifar_swin_t_224 cifar_swin_t_nomerge_128 cifar_cvt13_128 cifar_cvt13_nopyramid_128 cifar_linf_k24_128 cifar_linf_k49_128 cifar_linf_k98_128 \
  --samples 512 --batch_size 16 --device cuda:0 \
  --out_csv image_models_metrics.csv \
  --dump_imdb_like_file dumps/all_models_per_sample.csv