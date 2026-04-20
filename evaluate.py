import numpy as np


def _row_normalize(W):
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return W / norms


def most_similar(word, W_in, word_id, id_word, top_k=10):
    if word not in word_id:
        return []
    idx = word_id[word]
    W_norm = _row_normalize(W_in)
    sims = W_norm @ W_norm[idx]
    sims[idx] = -np.inf                  # exclude the query word itself
    top = np.argpartition(sims, -top_k)[-top_k:]
    top = top[np.argsort(sims[top])[::-1]]
    return [(id_word[i], float(sims[i])) for i in top]


def analogy(word_a, word_b, word_c, W_in, word_id, id_word, top_k=5):
    missing = [w for w in (word_a, word_b, word_c) if w not in word_id]
    if missing:
        return [], missing

    exclude = {word_id[word_a], word_id[word_b], word_id[word_c]}
    W_norm = _row_normalize(W_in)

    query = W_norm[word_id[word_b]] - W_norm[word_id[word_a]] + W_norm[word_id[word_c]]
    sims = W_norm @ query
    for idx in exclude:
        sims[idx] = -np.inf

    top = np.argpartition(sims, -top_k)[-top_k:]
    top = top[np.argsort(sims[top])[::-1]]
    return [(id_word[i], float(sims[i])) for i in top], []
