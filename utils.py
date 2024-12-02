import torch
import numpy as np
import random


def accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return correct/len(y_pred)*100


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False