import torch
from src.data.vocab import pad_id_sequence
from src.models.attention import build_causal_mask, build_padding_mask, compute_padding_and_causal_masks
from src.models.layers import scale_embeddings_by_sqrt_dmodel, build_sinusoidal_positional_encoding, add_positional_encoding_to_embeddings
from src.models.multi_head_attention import assemble_multi_head_attention_forward, apply_linear_projection
from src.models.feedforward import apply_residual_add_and_norm, position_wise_feed_forward_network

def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):

    self_attention_output = assemble_multi_head_attention_forward(x, x, x, w_q, w_k, w_v, w_o, num_heads=num_heads, mask=src_mask)
    output = apply_residual_add_and_norm(x, self_attention_output, gamma, beta)
    return output


def encoder_layer_feedforward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    
    feed_forward_output = position_wise_feed_forward_network(x, w1, b1, w2, b2)
    output = apply_residual_add_and_norm(x, feed_forward_output, gamma, beta)
    return output

def assemble_encoder_layer(x, layer_params, num_heads, src_mask):

    w_q, w_k, w_v, w_o, attn_gamma, attn_beta,  w1, b1, w2, b2, ffn_gamma, ffn_beta = layer_params.values()
    self_attention_output = encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, attn_gamma, attn_beta, num_heads=num_heads, src_mask = src_mask)
    output = encoder_layer_feedforward_sublayer(self_attention_output, w1, b1, w2, b2, ffn_gamma, ffn_beta)
    return output

def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):

    layer_output = x
    for params in encoder_layer_params_list:
        y = assemble_encoder_layer(layer_output, params, num_heads, src_mask)
        layer_output = y
    
    return layer_output

def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):

    self_attention_output = assemble_multi_head_attention_forward(y,y,y,w_q, w_v, w_o, num_heads, tgt_mask )
    output = apply_residual_add_and_norm(y, self_attention_output, gamma, beta)
    return output


def decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):

    #Because the query come from y the shape of y can be different from seq_len of the input
    if src_mask is not None:
        new_src_mask = src_mask.view(src_mask.shape[0], 1, src_mask.shape[1])

    cross_attention_output = assemble_multi_head_attention_forward(y, encoder_output, encoder_output, w_q, w_k, w_v, w_o, num_heads, new_src_mask )
    output = apply_residual_add_and_norm(y, cross_attention_output, gamma, beta)
    return output

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):

    ff_output = position_wise_feed_forward_network(y, w1, b1, w2, b2)
    output = apply_residual_add_and_norm(y, ff_output, gamma, beta)
    return output


def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):

    w_q1, w_k1, w_v1, w_o1, attn_gamma1, attn_beta1, w_q2, w_k2, w_v2, w_o2, attn_gamma2, attn_beta2, w1, b1, w2, b2, ffn_gamma, ffn_beta = layer_params.values()

    y_self_attention = decoder_layer_masked_self_attention_sublayer(y, w_q1, w_k1, w_v1, w_o1, attn_gamma1, attn_beta1, num_heads, tgt_mask )

    cross_attention_output = decoder_layer_cross_attention_sublayer(y_self_attention, encoder_output, w_q2, w_k2, w_v2, w_o2, attn_gamma2, attn_beta2, num_heads, src_mask)
    
    output = position_wise_feed_forward_network(cross_attention_output, w1, b1, w2, b2, ffn_gamma, ffn_beta)

    return output

def stack_decoder_layers(y, encoder_output, decoder_layer_params_list, num_heads, src_mask, tgt_mask):

    layer_output = y
    for params in decoder_layer_params_list:

        output = assemble_decoder_layer(layer_output, encoder_output, num_heads, src_mask, tgt_mask )
        layer_output = output

    return layer_output

def apply_final_output_projection(decoder_output, output_projection_weight, output_projection_bias=None):

    output = apply_linear_projection(decoder_output, output_projection_weight, output_projection_bias )
    return output


def tie_output_projection_to_token_embeddings(token_embedding_weight):

    return token_embedding_weight.transpose(-1, -2)


def apply_log_softmax_over_vocab(logits):

    return torch.log_softmax(logits, dim=-1)



def run_transformer_forward(src_ids, tgt_ids, model_params, num_heads, pad_id):

    #model_params = {'token_embedding': emb, 'encoder_layers': [], 'decoder_layers': [], 'output_projection': proj}


    src_max_len = max(len(src) for src in src_ids)
    tgt_max_len = max(len(tgt) for tgt in tgt_ids)

    src_ids_padded = pad_id_sequence(src_ids, src_max_len, pad_id)
    tgt_ids_padded = pad_id_sequence(tgt_ids, tgt_max_len, pad_id)

    src_emb = torch.nn.functional(src_ids_padded, model_params['token_embedding'])
    tgt_emb = torch.nn.functional(tgt_ids_padded, model_params['token_embedding'])

    scaled_src_emb = scale_embeddings_by_sqrt_dmodel(src_emb, len(model_params['token_embedding'][0]))
    scaled_tgt_emb = scale_embeddings_by_sqrt_dmodel(tgt_emb, len(model_params['token_embedding'][0]))

    src_positional_encoding = build_sinusoidal_positional_encoding(src_max_len, len(model_params['token_embedding'][0]))
    tgt_positional_encoding = build_sinusoidal_positional_encoding(tgt_max_len, len(model_params['token_embedding'][0]))

    src_position_encoded = add_positional_encoding_to_embeddings(scaled_src_emb, src_positional_encoding)
    tgt_position_encoded = add_positional_encoding_to_embeddings(scaled_tgt_emb, tgt_positional_encoding)


    src_padding_mask = build_padding_mask(src_ids_padded, pad_id)
    tgt_padding_mask = build_padding_mask(tgt_ids_padded, pad_id)
    tgt_causal_mask = build_causal_mask(tgt_position_encoded.shape[1])
    tgt_padding_and_causal_mask = compute_padding_and_causal_masks(tgt_padding_mask, tgt_causal_mask)

    encoder_output = stack_encoder_layers(src_position_encoded, model_params['encoder_layers'], num_heads, src_mask=src_padding_mask)
    decoder_output = stack_decoder_layers(tgt_position_encoded, encoder_output, model_params['decoder_layers'], num_heads, src_mask = src_padding_mask, tgt_mask = tgt_padding_and_causal_mask )

    output = apply_final_output_projection(decoder_output, output_projection_bias=model_params['output_projection'])

    return apply_log_softmax_over_vocab(output)