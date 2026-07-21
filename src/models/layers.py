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


def build_position_column(max_len):
    return torch.arange(max_len, dtype=torch.float32).unsqueeze(1)


def fill_even_indices_with_sin(pe, position, div_term):

    #position is [L,1]
    #div_term is (d_model//2,)
    frequency = position * div_term #(L,d_model//2)
    pe[:,0::2] = torch.sin(frequency)
    return pe


def fill_odd_indices_with_cos(pe, position, div_term):

    frequency = position * div_term
    pe[:,1::2] = torch.cos(frequency)
    return pe

def build_sinusoidal_positional_encoding(max_len, d_model):

    div_term = compute_positional_div_term(d_model)
    pe = torch.zeros(max_len, d_model)
    position = build_position_column(max_len)
    pe = fill_even_indices_with_sin(pe, position, div_term)
    pe = fill_odd_indices_with_cos(pe, position, div_term)
    return pe



def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):

    return embedded_batch + positional_encoding[: embedded_batch.shape[1], :]
