import numpy as np

from sampling import sample_negative_ids
from model import (
    lookup_embedding,
    compute_scores,
    compute_loss,
    compute_gradients,
    update_parameters,
)

def train_on_example(target_id, context_id, W_in, W_out,
                     distribution, num_negatives, learning_rate):
    forbidden_ids = {target_id, context_id}

    negative_ids = sample_negative_ids(
        num_samples=num_negatives,
        vocab_size=W_out.shape[0],
        distribution=distribution,
        forbidden_ids=forbidden_ids,
    )

    v, u_pos, u_negs = lookup_embedding(
        target_id, context_id, negative_ids, W_in, W_out
    )

    pos_score, neg_scores = compute_scores(v, u_pos, u_negs)
    loss = compute_loss(pos_score, neg_scores)

    grad_v, grad_u_pos, grad_u_negs = compute_gradients(
        v, u_pos, u_negs, pos_score, neg_scores
    )

    update_parameters(
        target_id, context_id, negative_ids,
        grad_v, grad_u_pos, grad_u_negs,
        W_in, W_out, learning_rate
    )

    return loss

def train_epoch(skipgrams, W_in, W_out, distribution, num_negatives, learning_rate):
    total_loss = 0.0

    for target_id, context_id in skipgrams:
        total_loss += train_on_example(
            target_id, context_id,
            W_in, W_out,
            distribution,
            num_negatives,
            learning_rate
        )

    return total_loss / len(skipgrams)


def train(skipgrams, W_in, W_out, distribution, num_negatives, learning_rate, epochs):
    losses = []

    for epoch in range(epochs):
        np.random.shuffle(skipgrams)

        loss = train_epoch(
            skipgrams,
            W_in,
            W_out,
            distribution,
            num_negatives,
            learning_rate
        )

        losses.append(loss)
        print(f"epoch {epoch + 1} / {epochs}, Loss: {loss: .4f}")

    return losses
