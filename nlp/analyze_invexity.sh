python analyze_invexity.py \
  --runs_root /home/ubuntu/PycharmProjects/transformer/nlp/runs \
  --folders imdb_perf_m64 imdb_perf_m128 imdb_perf_m256 \
            imdb_reformer_b32 imdb_reformer_b64 imdb_reformer_b128 \
  --max_len 1024 --samples 200 --topk 5 \
  --device cuda --seed 42 \
  --out_csv imdb_invexity_metrics.csv