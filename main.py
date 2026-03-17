from preprocessing import tokenize
from vocab import build_vocab, tokens_to_ids, compute_word_counts, generate_skipgrams
from sampling import build_negative_sampling_dist
from model import initialize_parameters
from training import train

def main():
    text = "the cat sits on the mat and the cat likes the mat"

    tokens = tokenize(text)
    word_id, id_word = build_vocab(tokens)
    token_ids = tokens_to_ids(tokens, word_id)
    counts = compute_word_counts(token_ids, vocab_size=len(word_id))
    skipgrams = generate_skipgrams(token_ids, window_size=2)

    W_in, W_out = initialize_parameters(
        vocab_size=len(word_id),
        embedding_dimension=10
    )

    distribution = build_negative_sampling_dist(counts)

    losses = train(
        skipgrams,
        W_in,
        W_out,
        distribution,
        num_negatives=3,
        learning_rate=0.05,
        epochs=50
    )

    print("Final loss:", losses[-1])

if __name__ == "__main__":
    main()