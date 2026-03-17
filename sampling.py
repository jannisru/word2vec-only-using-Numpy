import numpy as np


def build_negative_sampling_dist(word_counts):
    adjusted = word_counts ** 0.75
    dist = adjusted / adjusted.sum()
    
    return dist


def sample_negative_ids(num_samples, vocab_size, distribution, forbidden_ids=None):
    if len(distribution) != vocab_size:
        raise ValueError("distribution not equal to length vocab_size")
    
    if num_samples < 0:
        raise ValueError("num_samples not >= 0")
    
    if forbidden_ids is None:
        forbidden_ids = set()
    else:
        forbidden_ids = set(forbidden_ids)

    negative_ids = []

    while len(negative_ids) < num_samples:
        candidate = np.random.choice(vocab_size, p=distribution)
        if candidate in forbidden_ids:
            continue
        negative_ids.append(candidate)

    return np.array(negative_ids, dtype=np.int64)