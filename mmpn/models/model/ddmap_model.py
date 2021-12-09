import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import SMODELS, build_loss
from .transformer import Transformer 



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



@SMODELS.register_module()
class DdmapModel(nn.Module):
  def __init__(self):
    super(DdmapModel, self).__init__()
    num_queries = 1
    d_model = 128
    nhead=2
    num_encoder_layers=1
    num_decoder_layers=1

    hidden_dim =  d_model
    input_feature_size = 3

    self.input_proj = MLP(input_feature_size, hidden_dim, hidden_dim, 2)
    self.transformer = Transformer(d_model=d_model, nhead=nhead,
                      num_encoder_layers=num_encoder_layers,
                      num_decoder_layers=num_decoder_layers)
    self.query_embed = nn.Embedding(num_queries, hidden_dim) 
    #self.class_embed = nn.Linear(hidden_dim, 1)
    self.coeff_embed = MLP(hidden_dim, hidden_dim, 3, 3)

  def forward(self, src, mask):
    
    proj = self.input_proj(src) 
    hs = self.transformer(proj, mask, self.query_embed.weight)[0]
  
    #outputs_class = self.class_embed(hs).sigmoid()
    outputs_coeff = self.coeff_embed(hs) 
 
 
    return outputs_coeff
 
