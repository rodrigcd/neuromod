import torch


def detach_torch(tensor):
    return tensor.detach().cpu().numpy()