import torch

def to_float_tensor(x):
    return torch.tensor(x, dtype=torch.float32)