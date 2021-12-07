import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES

@LOSSES.register_module()
class BCELoss(nn.Module):
    
    def __init__(self, weight=None, reduction='mean'):
        super(BCELoss, self).__init__()
        
        self.register_buffer('weight', weight)
        self.reduction = reduction
    
    def forward(self, input, target):
        return F.binary_cross_entropy(input, target, weight=self.weight, reduction=self.reduction)