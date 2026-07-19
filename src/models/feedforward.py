import torch
import math

def apply_ff_first_linear_and_relu(x, w1, b1):
    #x has dimensions of [L, d_model] and w1 has [d_model, d_diff]
    #d_diff is in general 4. d_model
    scaled_x = torch.matmul(x,w1)
    if b1 is None:
        scaled_x = scaled_x + b1
    
    output = torch.relu(scaled_x)

    return output

def apply_ff_second_linear(x, w2, b2):
    #x has dimensions of [L, d_diff]
    #w2 has dimensions of [d_diff, d_model]

    if b2 is not None:
        output = torch.matmul(x, w2) + b2
    else:
        output = torch.matmul(x, w2)

    return output


def position_wise_feed_forward_network(x, w1, b1, w2, b2):

    first_linear_and_relu = apply_ff_first_linear_and_relu(x, w1, b1)
    output = apply_ff_second_linear(first_linear_and_relu, w2, b2)
    return output


def compute_layer_norm_mean_and_variance(x):

    var, mean = torch.var_mean(x, dim=-1, keepdim=True, correction=0)
    #correction=0 uses N for  the variance
    return mean, var


def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):

    mean, var = compute_layer_norm_mean_and_variance(x)
    normalized_x = (x - mean) / (math.sqrt(var + eps))
    y = (gamma * normalized_x) + beta
    return y

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):

    output = residual_input + sublayer_output
    return normalize_and_scale_with_gamma_beta(output, gamma, beta, eps=eps)

def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):

    x_with_mask = x * keep_mask #dot product
    return x_with_mask/keep_prob