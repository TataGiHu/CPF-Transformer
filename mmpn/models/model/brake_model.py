import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import SMODELS, build_loss

@SMODELS.register_module()
class BrakeModel(nn.Module):
    def __init__(self, in_features, n_class=1, hidden=64):
        super(BrakeModel, self).__init__()
        
        self.in_features = in_features
        self.n_class = n_class
        self.hidden = hidden
        
        self.linear_1 = nn.Linear(self.in_features, self.hidden)
        self.linear_2 = nn.Linear(self.hidden, self.n_class)
            
    def forward(self, x):
               
        x = F.relu(self.linear_1(x))
        x = torch.sigmoid(self.linear_2(x))
        
        return x
 
