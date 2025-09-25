# Trainability-Transformers
weight .pt file saved in: https://drive.google.com/drive/folders/1nZ9D95lk9px2I0qtjlXJbcihMGOBISgr?usp=share_link

## 📖 Overview

Transformer-based deep neural networks (DNNs) have become the backbone of modern AI, achieving state-of-the-art performance in **language**, **vision**, and **time-series** tasks.  
Despite their non-convex and extremely high-dimensional landscapes, Transformers are **remarkably trainable** — a long-standing mystery in deep learning theory.

We introduce a **unifying theoretical framework based on invexity** — a generalization of convexity that guarantees all critical points are global minimizers. Our analysis demonstrates that **widely adopted Transformer architectures (ViT, Swin, Linformer, Reformer, Conformer, S4)** satisfy invexity, providing the first theoretical foundation for their trainability.

---

## 🚀 Contributions

- **Theory:**  
  - Prove that modern Transformer variants are **invex**, ensuring global optimality of all stationary points.  
  - Establish **trainability as a structural property**, not just an empirical phenomenon.  
  - Extend beyond convex/quasi-convex frameworks to **support non-convex activations and complex loss functions**.  

- **Experiments:**  
  - Empirically validate invexity across diverse modalities:
    - **Vision**: ViT, Swin, CvT, Linformer  
    - **NLP**: Reformer, Performer  
    - **Signal processing**: Conformer, S4  
  - Metrics include:
    - **NI margin** (Negative Independence)  
    - **σ_min** (minimum singular value of Jacobians)  
    - **Residual landscape analysis**  

---

## 📂 Repository Structure

```text
.
├── image/          # Vision experiments (ViT, Swin, CvT, Linformer)
├── nlp/            # NLP experiments (Reformer, Performer)
├── signal/         # Time-series experiments (Conformer, S4)
├── scripts/        # Training & plotting scripts
├── figs/           # Plots: NI margin, sigma_min, landscapes
├── requirements.txt
└── README.md

---
```

## ⚙️ Installation

```bash
git clone https://github.com/your-username/transformer-invexity.git
cd transformer-invexity
pip install -r requirements.txt

```
## ⚙️ Citation
```text
@article{yourname2025invexity,
  title={Trainability-Transformers},
  author={Jia Bi, Haochen Liu, Samuel Sanchez Pinilla},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
```

