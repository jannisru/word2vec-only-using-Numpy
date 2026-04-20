import numpy as np


def build_negative_sampling_dist(word_counts):
    adjusted = word_counts.astype(np.float64) ** 0.75
    dist = adjusted / adjusted.sum()
    dist /= dist.sum()   
    return dist


def sample_negatives(num_samples, distribution, forbidden_ids=None):
    if num_samples == 0:
        return np.empty(0, dtype=np.int64)

    vocab_size = len(distribution)
    forbidden = (np.fromiter(forbidden_ids, dtype=np.int64)
                 if forbidden_ids else np.empty(0, dtype=np.int64))

    result = np.empty(num_samples, dtype=np.int64)
    n_filled = 0

    batch = max(num_samples * 2, num_samples + len(forbidden) + 8)

    while n_filled < num_samples:
        needed = num_samples - n_filled
        candidates = np.random.choice(vocab_size, size=needed * 2 + len(forbidden) + 4,
                                      p=distribution)
        if len(forbidden):
            candidates = candidates[~np.isin(candidates, forbidden)]
        take = min(len(candidates), needed)
        result[n_filled:n_filled + take] = candidates[:take]
        n_filled += take

    return result


def subsample_token_ids(token_ids, word_counts, t=1e-5):
    total = float(word_counts.sum())
    freq = word_counts[token_ids] / total
    keep_prob = np.minimum(1.0, np.sqrt(t / freq) + t / freq)
    keep_mask = np.random.uniform(0.0, 1.0, size=len(token_ids)) < keep_prob
    return token_ids[keep_mask]
