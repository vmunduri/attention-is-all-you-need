import torch
import math

def build_padding_mask(token_ids: torch.Tensor, pad_id:int):

    mask = token_ids!=pad_id
    reshaped_mask = mask.view(mask.shape[0],1,1,mask.shape[1])
    return reshaped_mask

def build_causal_mask(seq_len):

    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    return mask[None, None, : , :]

def compute_padding_and_causal_masks(padding_mask, causal_mask):

    return torch.logical_and(padding_mask, causal_mask)


def compute_raw_attention_scores(query,key):

    key_transpose = key.transpose(-2, -1)
    return torch.matmul(query, key_transpose)

def scale_attention_scores(scores, d_k):

    return scores/math.sqrt(d_k)


def mask_attention_scores_with_neg_inf(scores, mask):

    masked_scores = torch.where(mask==False, -1e9, scores)
    return masked_scores

def softmax_attention_weights(masked_scores):

    exponential_scores = torch.exp(masked_scores)
    sum_of_exponential_column_scores = torch.sum(exponential_scores, dim=-1, keepdim=True)
    filtered_sum_of_exponential_column_scores = torch.where(sum_of_exponential_column_scores==0, 1, sum_of_exponential_column_scores)
    weights = exponential_scores / filtered_sum_of_exponential_column_scores
    return weights 

def apply_attention_weights_to_values(attention_weights, values):

    return torch.matmul(attention_weights, values)


def scaled_dot_product_attention(query, key, value, mask=None):

    raw_attention_scores = compute_raw_attention_scores(query, key)
    scaled_raw_attention_scores = scale_attention_scores(raw_attention_scores, key.shape[-1])
    if mask is not None:
        masked_attention_scores = mask_attention_scores_with_neg_inf(scaled_raw_attention_scores,
                                                                     mask)
    else:
        masked_attention_scores = scaled_raw_attention_scores
    
    attention_weights = softmax_attention_weights(masked_attention_scores)
    context = apply_attention_weights_to_values(attention_weights, value)
    return (context, attention_weights)
