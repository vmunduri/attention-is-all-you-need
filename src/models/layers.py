import math
import torch


def scale_embeddings_by_sqrt_dmodel(embeddings, d_model):
    #we need to scale our embeddings, so that the sinusoidal position weights wont dominate the embeddings distribution
    return embeddings * math.sqrt(d_model)


def compute_positional_div_term(d_model):
    #Log, exp makes better use of GPU Parallization than Pow function
    # The denominator of sin(wt) or cost(wt) in sinusoidal positional encoding is 10000^(2i/d_model)
    #we can convert it to exp(-i *  2 * log(10000) * (1/d_model))
    #Because this is same for cos and sin, the constant (2 * log(1000) * (1/d_model)) can have shape of (d_model//2,)

    frequency_divisor_const = torch.arange(d_model//2)
    frequency_divisor_const = (frequency_divisor_const * math.exp(-2 * math.log(10000) * (1/d_model)))
    return frequency_divisor_const

