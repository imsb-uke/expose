import torch

def compute_pos_weight_from_dataset(dataset):
    positives = 0
    total = 0

    for _, y in dataset:
        positives += (y == 1).sum().item()
        total += y.numel()

    negatives = total - positives
    if positives == 0:
        raise ValueError("No positive samples in training set")
    print(f"Calculated pos_weight based on training dataset: {negatives / positives}.")

    return negatives / positives