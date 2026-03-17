import numpy as np

def build_vocab(tokens):
    word_id = {}
    index = 0

    for token in tokens:
        if token not in word_id:
            word_id[token] = index
            index += 1

    id_word = {idx: token for token, idx in word_id.items()}
    return word_id, id_word


def tokens_to_ids(tokens, word_id):
    return np.array([word_id[token] for token in tokens], dtype=np.int64)


def compute_word_counts(token_ids, vocab_size):
    return np.bincount(token_ids, minlength=vocab_size).astype(np.float32)


def generate_skipgrams(token_ids, window_size):
    skipgrams = []
    
    for i in range(len(token_ids)):
        left = max(0, i - window_size)
        right = min(len(token_ids), i + window_size + 1)
        
        for j in range(left, right):    
            if j != i:
                skipgrams.append((token_ids[i], token_ids[j]))
    return skipgrams