from src.models.multi_head_attention import assemble_multi_head_attention_forward
from src.models.feedforward import apply_residual_add_and_norm

def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):

    self_attention_output = assemble_multi_head_attention_forward(x, x, x, w_q, w_k, w_v, w_o, num_heads=num_heads, mask=src_mask)
    output = apply_residual_add_and_norm(x, self_attention_output, gamma, beta)
    return output