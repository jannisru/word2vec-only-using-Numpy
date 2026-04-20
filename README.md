# Word2Vec — Skip-Gram with Negative Sampling (pure NumPy)

Implementation of the word2vec Skip-Gram model with Negative Sampling (SGNS)
using only NumPy.  Reference: Mikolov et al., *Distributed Representations of
Words and Phrases and their Compositionality*, NeurIPS 2013.

---

## Model

### Loss function

For a center word *c* and a positive context word *o*, with *K* noise words
drawn from the unigram distribution, the per-pair SGNS objective is:

```
L = -log σ(v_c · u_o) - Σ_{k=1}^{K} log σ(−v_c · u_k)
```

where σ is the sigmoid function, `v_c = W_in[c]` is the center embedding, and
`u_o, u_k = W_out[o], W_out[k]` are output embeddings.

### Gradients

Let `e_pos = σ(v_c · u_o) − 1` and `e_k = σ(v_c · u_k)`. Then:

```
∂L/∂v_c  = e_pos · u_o + Σ_k e_k · u_k   (center word — aggregates all K+1 pairs)
∂L/∂u_o  = e_pos · v_c                    (positive context)
∂L/∂u_k  = e_k   · v_c   for each k       (each negative sample)
```

---

## Key design decisions

| Topic | Choice | Reason |
|---|---|---|
| **Initialisation** | `W_in ~ Uniform(−0.5/d, 0.5/d)`, `W_out = 0` | Matches original C code; 1/d scaling keeps initial dot-products O(1) regardless of dimension |
| **Optimizer** | SGD (no momentum) | Correct for sparse updates; momentum accumulates over all parameters and gives wrong effective learning rates for rarely-updated embeddings |
| **Negative distribution** | Unigram^0.75 | Smooths over rare words; from Mikolov et al. (2013) |
| **Negative sampling** | Batched `np.random.choice` + `np.isin` filter | Avoids per-sample CDF recomputation; O(V) memory, no large pre-allocated table |
| **Negative update** | `np.add.at` | Correctly accumulates gradient contributions when an index appears more than once in `negative_ids`; fancy-index assignment would silently drop all but the last update |
| **View safety** | `.copy()` on scalar-indexed rows | Scalar indexing returns a NumPy view; `.copy()` prevents the in-place SGD update from corrupting the embedding vectors used during gradient computation |
| **Subsampling** | `P(keep) = min(1, sqrt(t/f) + t/f)`, t=1e-5 | Reduces influence of high-frequency function words; improves speed and quality (§2.3 of the paper) |
| **Sigmoid clipping** | `clip(x, −10, 10)` | Float32 exp() saturates beyond ±10; tighter bound avoids overflow warnings and is numerically identical for all practical inputs |

---

## Evaluation

Embeddings are evaluated by:

1. **Nearest neighbours** — cosine similarity via normalised dot products  
2. **Word analogies** — 3CosAdd method: find *d* maximising
   `cos(d, b) − cos(d, a) + cos(d, c)`  
   (e.g. *man : king :: woman : ?* → queen)

### Sample output (text8, 1 M tokens, 100-d, 5 epochs)

```
Nearest neighbours:
  king      : ['queen', 'prince', 'emperor', 'throne', 'royal']
  paris     : ['london', 'berlin', 'rome', 'madrid', 'vienna']
  good      : ['better', 'great', 'bad', 'best', 'well']

Analogy accuracy: 3/4 = 75%
  [correct] man:king::woman:?    -> ['queen', ...]
  [correct] paris:france::berlin:? -> ['germany', ...]
  [correct] good:better::bad:?   -> ['worse', ...]
```

---

## Run

```bash
# Quick demo (built-in sample text, no downloads needed):
python main.py

# Train on a custom text file:
python main.py --file data/text.txt

# Train on text8 (downloads ~100 MB on first run):
python main.py --text8

# Custom settings:
python main.py --text8 --tokens 5000000 --dim 200 --epochs 10 --lr 0.025
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--file PATH` | — | Path to a plain-text corpus file |
| `--text8` | off | Download and use the text8 corpus |
| `--tokens N` | 1 000 000 | Number of text8 tokens to use |
| `--dim D` | 100 | Embedding dimension |
| `--window W` | 5 | Context window size |
| `--negatives K` | 5 | Negative samples per training pair |
| `--lr LR` | 0.025 | SGD learning rate |
| `--epochs E` | 5 | Number of training epochs |
| `--min-count M` | 5 | Minimum token frequency for vocabulary |

---

## Project structure

```
model.py        sigmoid, initialisation, forward pass + gradients, SGD update
sampling.py     unigram^0.75 distribution, batched negative sampler, subsampling
vocab.py        vocabulary building (with min_count), token-to-id, skip-gram generation
training.py     stochastic training loop
evaluate.py     most_similar (cosine), analogy (3CosAdd)
preprocessing.py text normalisation and tokenisation
main.py         entry point with CLI and evaluation reporting
```

---

## Reference

T. Mikolov, I. Sutskever, K. Chen, G. Corrado, J. Dean.  
*Distributed Representations of Words and Phrases and their Compositionality.*  
NeurIPS 2013. arXiv:1310.4546
