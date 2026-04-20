import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10.0, 10.0)))


def initialize_parameters(vocab_size, embedding_dim):
    scale = 0.5 / embedding_dim
    W_in = np.random.uniform(-scale, scale,
                             (vocab_size, embedding_dim)).astype(np.float32)
    W_out = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
    return W_in, W_out


def forward_and_backward(center_id, context_id, negative_ids, W_in, W_out):
    v_c = W_in[center_id].copy()      
    u_o = W_out[context_id].copy()    
    u_k = W_out[negative_ids].copy()  
                                      

    pos_score = v_c @ u_o             
    neg_scores = u_k @ v_c            

    sig_pos = sigmoid(pos_score)      
    sig_neg = sigmoid(neg_scores)     

    loss = -np.log(sig_pos) - np.sum(np.log(1.0 - sig_neg))

    e_pos = sig_pos - 1.0
    e_neg = sig_neg      

    grad_v_c = e_pos * u_o + e_neg @ u_k    
    grad_u_o = e_pos * v_c                  
    grad_u_k = e_neg[:, None] * v_c         

    return loss, grad_v_c, grad_u_o, grad_u_k


def sgd_update(center_id, context_id, negative_ids,
               grad_v_c, grad_u_o, grad_u_k,
               W_in, W_out, lr):
    W_in[center_id] -= lr * grad_v_c
    W_out[context_id] -= lr * grad_u_o
    np.add.at(W_out, negative_ids, -lr * grad_u_k)
