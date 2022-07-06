import torch

from typing import Optional

from .register import LOSSES


@LOSSES.register()
class DistanceLoss(torch.nn.Module):

    def __init__(
        self, 
        p: float = 2.0, 
        reduction: str = 'sum', 
        eps: float = 1e-06, 
        keepdim: bool = False,
        weights: Optional[torch.Tensor] = None
        ) -> None:

        assert reduction in ['mean', 'sum'], \
            f'Reduction must be either \'mean\' or \'sum\', got {reduction}'
        
        super().__init__()

        self.reduction = reduction
        self.pdist = torch.nn.PairwiseDistance(p=p, eps=eps, keepdim=keepdim)
        self.weights = weights
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        return self._distance_loss(output, target)
    
    def _distance_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        B, C, _, _ = target.shape

        if self.weights is not None:
            assert self.weights.shape[0] == C, \
                'Number of weights must match the number of channels, ' \
                    f'got {C} channels and {self.weights.shape[0]} weights'

        loss = torch.zeros((B,C))

        for b in range(B):
            for c in range(C):
                loss[b][c] = self.pdist(output[b][c], target[b][c]).sum()
        
        if self.weights is not None:
            loss = self.weights * loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()