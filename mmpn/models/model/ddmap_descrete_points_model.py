import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import SMODELS, build_loss
from .transformer import Transformer 
from .common_layer import MLP 

@SMODELS.register_module()
class DdmapDescreteModel(nn.Module):
  def __init__(self):
    super(DdmapDescreteModel, self).__init__()
    num_queries = 1
    d_model = 128
    nhead=2
    num_encoder_layers=1
    num_decoder_layers=1

    hidden_dim =  d_model
    input_feature_size = 2

    self.input_proj = MLP(input_feature_size, hidden_dim, hidden_dim, 2)
    self.transformer = Transformer(d_model=d_model, nhead=nhead,
                      num_encoder_layers=num_encoder_layers,
                      num_decoder_layers=num_decoder_layers)
    self.query_embed = nn.Embedding(num_queries, hidden_dim) 
    self.class_embed = nn.Linear(hidden_dim, 3)
    self.lane_point_embed = MLP(hidden_dim, hidden_dim, 60, 3) # the output layer now has 60 elements per frame (3 * 20)

  def forward(self, src, mask):
    
    proj = self.input_proj(src) 
    hs = self.transformer(proj, mask, self.query_embed.weight)[0]
    outputs_class = self.class_embed(hs).sigmoid() # classification embedding enabled
    output_y_points = self.lane_point_embed(hs) 
 
 
    return output_y_points, outputs_class # now returning a tuple
 


@SMODELS.register_module()
class DdmapDescreteModelThreeQueries(nn.Module):
  def __init__(self):
    super(DdmapDescreteModelThreeQueries, self).__init__()
    num_queries = 3
    d_model = 128
    nhead=8
    num_encoder_layers=6
    num_decoder_layers=6

    hidden_dim =  d_model
    input_feature_size = 202

    self.input_proj = MLP(input_feature_size, hidden_dim, hidden_dim, 2)
    self.transformer = Transformer(d_model=d_model, nhead=nhead,
                      num_encoder_layers=num_encoder_layers,
                      num_decoder_layers=num_decoder_layers)
    self.query_embed = nn.Embedding(num_queries, hidden_dim) 
    self.class_embed = nn.Linear(hidden_dim, 1)
    self.lane_point_embed = MLP(hidden_dim, hidden_dim, 25, 3) 

  def forward(self, src, mask):
    
    proj = self.input_proj(src) 
    hs = self.transformer(proj, mask, self.query_embed.weight)[0]
    outputs_class = self.class_embed(hs).sigmoid() # classification embedding enabled
    output_y_points = self.lane_point_embed(hs) 
 
    return output_y_points, outputs_class # now returning a tuple