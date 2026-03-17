import numpy as np


def sigmoid(x):
    x = np.clip(x, -50, 50)
    
    return 1 / (1 + np.exp(-x))

def initialize_parameters(vocab_size, embedding_dimension):
    W_in = np.random.randn(vocab_size, embedding_dimension) * 0.01
    W_out = np.random.randn(vocab_size, embedding_dimension) * 0.01

    return W_in, W_out

def lookup_embedding(target_id, context_id, negative_ids, W_in, W_out):
    v = W_in[target_id]
    u_positive = W_out[context_id]
    u_negative = W_out[negative_ids]

    return v, u_positive, u_negative

def compute_scores(v, u_positive, u_negative):
    pos_score = np.dot(u_positive, v)
    neg_scores = u_negative @ v

    return pos_score, neg_scores

def compute_loss(pos_score, neg_scores):
    eps = 1e-10

    return -np.log(sigmoid(pos_score) + eps) - np.sum(np.log(sigmoid(-neg_scores) + eps))

def compute_gradients(v, u_pos, u_negs, pos_score, neg_scores):
    sig_pos = sigmoid(pos_score)
    sig_neg = sigmoid(neg_scores)

    g_pos = sig_pos - 1.0
    g_neg = sig_neg

    grad_v = g_pos * u_pos + np.sum(g_neg[:, None] * u_negs, axis=0)
    grad_u_pos = g_pos * v
    grad_u_negs = g_neg[:, None] * v[None, :]

    return grad_v, grad_u_pos, grad_u_negs

def update_parameters(target_id, context_id, negative_ids,
                      grad_v, grad_u_pos, grad_u_negs,
                      W_in, W_out, learning_rate):
    W_in[target_id] -= learning_rate * grad_v
    W_out[context_id] -= learning_rate * grad_u_pos

    for neg_id, grad in zip(negative_ids, grad_u_negs):
        W_out[neg_id] -= learning_rate * grad