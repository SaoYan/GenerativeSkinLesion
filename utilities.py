import torch
import torch.nn as nn
import torch.nn.functional as F

def rescale_dynamic_range(data):
    # [0,1] --> [-1,1]
    return data * 2 - 1
