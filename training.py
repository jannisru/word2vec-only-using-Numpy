import numpy as np

from sampling import sample_negatives
from model import forward_and_backward, sgd_update


def train_on_pair(center_id, context_id, W_in, W_out,
                  distribution, num_negatives, lr):
    negative_ids = sample_negatives(
        num_negatives, distribution,
        forbidden_ids={int(center_id), int(context_id)},
    )
    loss, grad_v_c, grad_u_o, grad_u_k = forward_and_backward(
        int(center_id), int(context_id), negative_ids, W_in, W_out
    )
    sgd_update(
        int(center_id), int(context_id), negative_ids,
        grad_v_c, grad_u_o, grad_u_k,
        W_in, W_out, lr,
    )
    return loss


def train(skipgrams, W_in, W_out, distribution, num_negatives, lr, epochs):
    losses = []
    for epoch in range(1, epochs + 1):
        np.random.shuffle(skipgrams)
        total = 0.0
        for center_id, context_id in skipgrams:
            total += train_on_pair(
                center_id, context_id,
                W_in, W_out, distribution, num_negatives, lr,
            )
        avg = total / len(skipgrams)
        losses.append(avg)
        print(f"Epoch {epoch}/{epochs}  loss={avg:.4f}")
    return losses
