import torch

def apply_ff_first_linear_and_relu(x, w1, b1):

    scaled_x = torch.matmul(x,w1)
    if b1 is None:
        scaled_x = scaled_x + b1
    
    output = torch.relu(scaled_x)

    return output