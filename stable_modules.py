import torch
import torch.nn as nn

class LyapunovFunction(nn.Module):
    def __init__(self, type) -> None:
        super().__init__()
        self.type = type

    def forward(self, x):
        if self.type == 'square':
            return (x**2).mean(), 2 * x
        