import argparse
import os
import urllib.request
import zipfile

import numpy as np

from preprocessing import tokenize, load_text
from vocab import build_vocab, tokens_to_ids, compute_word_counts, generate_skipgrams
from sampling import build_negative_sampling_dist, subsample_token_ids
from model import initialize_parameters
from training import train
from evaluate import most_similar, analogy


_SAMPLE = (
    "the king rules the kingdom and the queen also rules the kingdom "
    "the prince is the son of the king and the princess is the daughter of the queen "
    "man works in the city and woman works in the city too "
    "paris is the capital of france and berlin is the capital of germany "
    "london is the capital of england and rome is the capital of italy "
    "madrid is the capital of spain and lisbon is the capital of portugal "
    "the dog barks and the cat meows and the bird sings "
    "the cat chases the mouse and the dog chases the cat "
    "a man plays football and a woman plays tennis "
    "the river flows through the valley and the mountain rises above the valley "
    "the sun rises in the east and sets in the west "
    "the student reads the book and the teacher writes the lesson on the board "
    "the doctor heals the patient and the nurse helps the doctor "
    "the programmer writes the code and the engineer builds the system "
    "the author writes the novel and the reader reads the novel "
    "the king and the queen live in the palace near the city "
    "the prince and the princess study in the school of the kingdom "
) * 30  



_TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"
_TEXT8_PATH = os.path.join(os.path.dirname(__file__), "text8")
_TEXT8_ZIP = os.path.join(os.path.dirname(__file__), "text8.zip")


def _load_text8(max_tokens=None):
    if not os.path.exists(_TEXT8_PATH):
        print(f"Downloading text8 from {_TEXT8_URL} ...")
        urllib.request.urlretrieve(_TEXT8_URL, _TEXT8_ZIP)
        with zipfile.ZipFile(_TEXT8_ZIP) as zf:
            zf.extractall(os.path.dirname(_TEXT8_PATH))
        os.remove(_TEXT8_ZIP)
        print("Download complete.")
    with open(_TEXT8_PATH, "r", encoding="utf-8") as fh:
        tokens = fh.read().split()
    if max_tokens:
        tokens = tokens[:max_tokens]
    return tokens



def main(use_text8=False, file_path=None, max_tokens=1_000_000,
         embedding_dim=100, window_size=5,
         num_negatives=5, lr=0.025, epochs=5,
         min_count=5, subsampling_t=1e-5):

    if use_text8:
        tokens = _load_text8(max_tokens=max_tokens)
        print(f"Corpus: text8, {len(tokens):,} tokens")
    elif file_path:
        tokens = tokenize(load_text(file_path))
        print(f"Corpus: {file_path}, {len(tokens):,} tokens")
    else:
        tokens = tokenize(_SAMPLE)
        print(f"Corpus: built-in sample, {len(tokens):,} tokens")

    mc = min_count if use_text8 else 1
    word_id, id_word = build_vocab(tokens, min_count=mc)
    token_ids = tokens_to_ids(tokens, word_id)
    vocab_size = len(word_id)
    print(f"Vocabulary: {vocab_size:,} words  (min_count={mc})")

    counts = compute_word_counts(token_ids, vocab_size)
    distribution = build_negative_sampling_dist(counts)


    if use_text8:
        before = len(token_ids)
        token_ids = subsample_token_ids(token_ids, counts, t=subsampling_t)
        print(f"Tokens after subsampling: {len(token_ids):,} "
              f"({100.0 * len(token_ids) / before:.1f}% retained)")

    skipgrams = generate_skipgrams(token_ids, window_size=window_size)
    print(f"Skip-gram pairs: {len(skipgrams):,}\n")

    W_in, W_out = initialize_parameters(vocab_size, embedding_dim)

    train(skipgrams, W_in, W_out, distribution,
          num_negatives=num_negatives, lr=lr, epochs=epochs)

    print("\n--- Nearest neighbours (cosine similarity) ---")
    probes = ["king", "man", "woman", "paris", "cat", "good"]
    for word in probes:
        if word in word_id:
            nbrs = most_similar(word, W_in, word_id, id_word, top_k=5)
            words = [w for w, _ in nbrs]
            print(f"  {word:10s}: {words}")

    if use_text8:
        print("\n--- Analogy evaluation  (a : b :: c : ?) ---")
        tasks = [
            ("man",    "king",    "woman",   "queen"),
            ("paris",  "france",  "berlin",  "germany"),
            ("good",   "better",  "bad",     "worse"),
            ("king",   "man",     "queen",   "woman"),
        ]
        correct = 0
        total = 0
        for a, b, c, expected in tasks:
            results, missing = analogy(a, b, c, W_in, word_id, id_word, top_k=5)
            if missing:
                print(f"  {a}:{b}::{c}:?  -- missing from vocab: {missing}")
                continue
            top_words = [w for w, _ in results]
            hit = expected in top_words
            correct += int(hit)
            total += 1
            status = "correct" if hit else "wrong  "
            print(f"  [{status}] {a}:{b}::{c}:? -> {top_words}  (expected: {expected})")
        if total:
            print(f"\nAnalogy accuracy: {correct}/{total} = {100*correct/total:.0f}%")

    return W_in, W_out, word_id, id_word


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text8", action="store_true",
                        help="Download and train on text8 corpus")
    parser.add_argument("--file", metavar="PATH",
                        help="Path to a plain-text corpus file (e.g. data/text.txt)")
    parser.add_argument("--tokens", type=int, default=1_000_000,
                        help="Max tokens to use from text8 (default 1M)")
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--negatives", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=5)
    args = parser.parse_args()

    main(
        use_text8=args.text8,
        file_path=args.file,
        max_tokens=args.tokens,
        embedding_dim=args.dim,
        window_size=args.window,
        num_negatives=args.negatives,
        lr=args.lr,
        epochs=args.epochs,
        min_count=args.min_count,
    )
