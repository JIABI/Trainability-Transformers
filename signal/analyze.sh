python analyze_invexity.py \
  --data_root ./data \
  --runs_root runs \
  --folders ecg_conf_k9_h4 ecg_conf_k9_h6 ecg_conf_k15_h4 ecg_conf_k15_h6 ecg_conf_k31_h4 ecg_conf_k31_h6 \
           ecg_s4d_n32_s0 ecg_s4d_n32_s1 ecg_s4d_n32_s2 ecg_s4d_n64_s0 ecg_s4d_n64_s1 ecg_s4d_n64_s2 \
           ecg_s4d_n128_s0 ecg_s4d_n128_s1 ecg_s4d_n128_s2 \
  --num_samples 200 \
  --device cuda \
  --seed 42 \
  --out_csv results/ecg_invexity_metrics.csv