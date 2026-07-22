import torch
import math

def init_encoder_layer_parameters(d_model, num_heads, d_ff):
    """Return a dict of leaf tensors with requires_grad=True for one encoder layer."""
    
    w_q = torch.rand(d_model, d_model, requires_grad=True, dtype=torch.float32)
    w_k = torch.rand(d_model, d_model, requires_grad=True, dtype=torch.float32)
    w_v = torch.rand(d_model, d_model, requires_grad=True, dtype=torch.float32)
    w_o = torch.rand(d_model, d_model, requires_grad=True, dtype=torch.float32)
    attn_gamma = torch.ones(d_model, requires_grad=True, dtype=torch.float32)
    attn_beta = torch.zeros(d_model, requires_grad = True, dtype=torch.float32)

    w1 = torch.rand(d_model, d_ff, requires_grad=True, dtype=torch.float32)
    w2 = torch.rand(d_ff, d_model, requires_grad=True, dtype=torch.float32)
    b1 = torch.zeros(d_ff, requires_grad = True, dtype=torch.float32)
    b2 = torch.zeros(d_model, requires_grad = True, dtype=torch.float32)

    ffn_gamma = torch.ones(d_model, requires_grad=True, dtype=torch.float32)
    ffn_beta = torch.zeros(d_model, requires_grad=True, dtype=torch.float32)

    return {'w_q' : w_q, 'w_k': w_k, 'w_v':w_v, 'w_o':w_o, 'attn_gamma':attn_gamma, 'attn_beta':attn_beta, 
            'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2, 'ffn_gamma': ffn_gamma, 'ffn_beta': ffn_beta}


def init_decoder_layer_parameters(d_model, num_heads, d_ff):
    
    

    w_q_self = torch.rand(d_model, d_model, requires_grad=True, dtype=torch.float32)
    w_k_self = torch.rand(d_model, d_model, requires_grad=True, dtype=torch.float32)
    w_v_self = torch.rand(d_model, d_model, requires_grad=True, dtype=torch.float32)
    w_o_self = torch.rand(d_model, d_model, requires_grad=True, dtype=torch.float32)
    w_q_cross = torch.rand(d_model, d_model, requires_grad=True, dtype=torch.float32)
    w_k_cross = torch.rand(d_model, d_model, requires_grad=True, dtype=torch.float32)
    w_v_cross = torch.rand(d_model, d_model, requires_grad=True, dtype=torch.float32)
    w_o_cross = torch.rand(d_model, d_model, requires_grad=True, dtype=torch.float32)

    self_gamma = torch.ones(d_model, requires_grad=True, dtype=torch.float32)
    self_beta = torch.zeros(d_model, requires_grad=True, dtype=torch.float32)
    cross_gamma = torch.ones(d_model, requires_grad=True, dtype=torch.float32)
    cross_beta = torch.zeros(d_model, requires_grad=True, dtype=torch.float32)

    w1 = torch.rand(d_model, d_ff, requires_grad=True, dtype=torch.float32)
    w2 = torch.rand(d_ff, d_model, requires_grad=True, dtype=torch.float32)
    b1 = torch.zeros(d_ff, requires_grad=True, dtype=torch.float32)
    b2 = torch.zeros(d_model, requires_grad=True, dtype=torch.float32)

    ffn_gamma = torch.ones(d_model, requires_grad=True, dtype=torch.float32)
    ffn_beta = torch.zeros(d_model, requires_grad=True, dtype=torch.float32)

    return {
        'w_q_self': w_q_self,
        'w_k_self': w_k_self,
        'w_v_self': w_v_self,
        'w_o_self': w_o_self,
        'w_q_cross': w_q_cross,
        'w_k_cross': w_k_cross,
        'w_v_cross': w_v_cross,
        'w_o_cross': w_o_cross,
        'w1' : w1,
        'w2' : w2,
        'b1' : b1,
        'b2' : b2,
        'self_gamma' : self_gamma,
        'self_beta' : self_beta,
        'cross_gamma' : cross_gamma,
        'cross_beta' : cross_beta,
        'ffn_gamma' : ffn_gamma,
        'ffn_beta' : ffn_beta
    }


def init_embedding_and_projection_parameters(vocab_size, d_model, tie_weights=True):
    """Allocate src/tgt embeddings and output projection (optionally tied)."""
    
    src_embedding = torch.rand(vocab_size, d_model, requires_grad = True)
    tgt_embedding = torch.rand(vocab_size, d_model, requires_grad = True)
    if(tie_weights):
        output_projection = tgt_embedding
    else:
        output_projection = torch.rand(vocab_size, d_model, requires_grad = True)

    return{
        'src_embedding' : src_embedding,
        'tgt_embedding' : tgt_embedding,
        'output_projection' : output_projection
    }

def collect_model_parameters_into_list(encoder_layer_params, decoder_layer_params, embedding_params):
    
    seen_ids = set()
    params = []

    def add_tensor(t):
        if id(t) not in seen_ids:
            seen_ids.add(id(t))
            params.append(t)

    for enc in encoder_layer_params:
        for value in enc.values():
            add_tensor(value)
    
    for dec in decoder_layer_params:
        for value in dec.values():
            add_tensor(value)
        
    for key,value in embedding_params.items():
        add_tensor(value)

    return params