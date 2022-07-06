import torch
import torch.nn as nn

from torch.nn import MSELoss

from .register import LOSSES

@LOSSES.register()
class RMSELoss(nn.Module):

    def __init__(self, eps: float = 1e-6, *args, **kwargs) -> None:
        super().__init__()
        self.mse = MSELoss(*args, **kwargs)
        self.eps = eps
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(input, target) + self.eps)