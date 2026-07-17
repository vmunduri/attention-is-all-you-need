import torch
from src.models.attention import scaled_dot_product_attention

def split_last_dim_into_heads(tensor, num_heads):
    # Reshape (B, L, d_model) into (B, L, num_heads, d_model // num_heads)

    return tensor.view(tensor.shape[0], tensor.shape[1], num_heads, tensor.shape[-1]//num_heads)

def transpose_heads_before_sequence(split_tensor):
    # Rearrange (B, L, num_heads, d_k) into (B, num_heads, L, d_k)

    return split_tensor.transpose(-2, -3)

def merge_heads_back_to_model_dim(multi_head_tensor):
    # Reshape (B, H, L, d_model) to (B, L, H * d_model)

    multi_head_tensor_transpose = multi_head_tensor.transpose(-2, -3)
    merge_head_tensor = multi_head_tensor.reshape(multi_head_tensor_transpose.shape[0],
                                                  multi_head_tensor_transpose.shape[1],
                                                  multi_head_tensor_transpose.shape[-2] * multi_head_tensor_transpose.shape[-1])
    
    return merge_head_tensor


def apply_linear_projection(x, weights, bias):
    # Return x @ weight^T + bias (bias may be None) with shape (..., out_features)

    product = torch.matmul(x, weights.t())
    if bias is not None:
        return product + bias
    else:
        return product
    

def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    # Project x into separate query, key, and value tensors via three linear layers

    query = apply_linear_projection(x,w_q,b_q)
    key = apply_linear_projection(x, w_k, b_k)
    value = apply_linear_projection(x, w_v, b_v)
    return (query, key, value)

def split_qkv_into_heads(q, k, v, num_heads):
    #Split each of q, k, v into (B, num_heads, L, d_k) and return as a tuple

    q_h = split_last_dim_into_heads(q, num_heads)
    q_h = transpose_heads_before_sequence(q_h)
    k_h = split_last_dim_into_heads(k, num_heads)
    k_h = transpose_heads_before_sequence(k_h)
    v_h = split_last_dim_into_heads(v, num_heads)
    v_h = transpose_heads_before_sequence(v_h)

    return (q_h, k_h, v_h)


def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):

    context, attention = scaled_dot_product_attention(q_h, k_h, v_h, mask=mask)
    return (context, attention)


def merge_heads_and_project_output(context, w_o, b_o):
    #Merge the head axis back into d_model and apply the output linear projection.

    collapsed_multi_head = merge_heads_back_to_model_dim(context)
    return apply_linear_projection(collapsed_multi_head, w_o, b_o)

def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):

    q = apply_linear_projection(query, w_q, None)
    k = apply_linear_projection(key, w_k, None)
    v = apply_linear_projection(value, w_v, None)
    q_h, k_h, v_h = split_qkv_into_heads(q, k, v, num_heads=num_heads)
    context, attention = multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=mask)
    output = merge_heads_and_project_output(context, w_o, None)
    return output
