# word2vec — Skip-Gram with Negative Sampling (pure NumPy)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)

A from-scratch implementation of the word2vec Skip-Gram model with Negative Sampling (SGNS) — **no PyTorch, no shortcuts**. Every gradient is derived and applied by hand using only NumPy.

Built to understand exactly what word2vec does, not just use it.

Reference: [Mikolov et al., NeurIPS 2013](https://arxiv.org/abs/1310.4546)

## Results (text8, 1M tokens, 100-d, 5 epochs)

**Nearest neighbours**
```
king  → queen, prince, emperor, throne, royal
paris → london, berlin, rome, madrid, vienna
good  → better, great, bad, best, well
```

**Analogy accuracy (3CosAdd): 3/4 = 75%**
```
man:king   :: woman:?  → queen    ✓
paris:france :: berlin:? → germany ✓
good:better  :: bad:?    → worse   ✓
```

## Quick start

```bash
# Demo on built-in sample text (no downloads needed)
python main.py

# Train on the text8 corpus (~100 MB, downloads automatically)
python main.py --text8

# Custom settings
python main.py --text8 --tokens 5000000 --dim 200 --epochs 10 --lr 0.025
```

| Flag | Default | Description |
|---|---|---|
| `--file PATH` | — | Path to a plain-text corpus |
| `--text8` | off | Download and use the text8 corpus |
| `--tokens N` | 1,000,000 | Number of text8 tokens to use |
| `--dim D` | 100 | Embedding dimension |
| `--window W` | 5 | Context window size |
| `--negatives K` | 5 | Negative samples per pair |
| `--lr LR` | 0.025 | SGD learning rate |
| `--epochs E` | 5 | Training epochs |
| `--min-count M` | 5 | Minimum token frequency |

## Model

**Loss function** — per-pair SGNS objective for center word `c`, positive context `o`, and `K` noise words:

```
L = -log σ(v_c · u_o) - Σ_{k=1}^K log σ(−v_c · u_k)
```

**Gradients** (applied via SGD, no momentum):
```
∂L/∂v_c = e_pos · u_o + Σ_k e_k · u_k    (center word)
∂L/∂u_o = e_pos · v_c                      (positive context)
∂L/∂u_k = e_k · v_c                        (each negative sample)
```
where `e_pos = σ(v_c · u_o) − 1` and `e_k = σ(v_c · u_k)`.

## Key design decisions

| Topic | Choice | Reason |
|---|---|---|
| Initialisation | `W_in ~ Uniform(−0.5/d, 0.5/d)`, `W_out = 0` | Matches original C code; keeps initial dot-products O(1) |
| Optimizer | SGD (no momentum) | Correct for sparse updates |
| Negative distribution | Unigram^0.75 | Smooths over rare words (Mikolov 2013) |
| Negative update | `np.add.at` | Correctly accumulates gradient when index appears multiple times |
| Subsampling | `P(keep) = min(1, sqrt(t/f) + t/f)` | Reduces influence of high-frequency function words |

## Project structure

```
model.py         sigmoid, initialisation, forward pass + gradients, SGD update
sampling.py      unigram^0.75 distribution, batched negative sampler, subsampling
vocab.py         vocabulary building, token-to-id, skip-gram generation
training.py      stochastic training loop
evaluate.py      most_similar (cosine), analogy (3CosAdd)
preprocessing.py text normalisation and tokenisation
main.py          CLI entry point and evaluation reporting
```

## Stack

Python · NumPy
