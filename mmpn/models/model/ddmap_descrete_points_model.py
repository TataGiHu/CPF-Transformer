from ctypes import sizeof
from hashlib import new
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
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
    num_encoder_layers=1
    num_decoder_layers=1

    hidden_dim =  d_model
    input_feature_size = 202

    self.input_proj = MLP(input_feature_size, hidden_dim, hidden_dim, 2)
    self.transformer = Transformer(d_model=d_model, nhead=nhead,
                      num_encoder_layers=num_encoder_layers,
                      num_decoder_layers=num_decoder_layers)
    self.query_embed = nn.Embedding(num_queries, hidden_dim) 
    self.class_embed = nn.Linear(hidden_dim, 1)
    self.lane_point_embed = MLP(hidden_dim, hidden_dim, 25, 3) 

  def forward(self, src, mask, label, do_particle):
    
    proj = self.input_proj(src) 
    hs = self.transformer(proj, mask, self.query_embed.weight)[0]
    outputs_class = self.class_embed(hs).sigmoid() # classification embedding enabled
    output_y_points = self.lane_point_embed(hs) 
    # print('out:{}'.format(output_y_points))
    # print('size: {}'.format(output_y_points.size()))
    
    ##------------particle filter-----------------------------
    # if torch.cuda.is_available():
    #       torch_device = torch.device("cuda")
    # particle_batch = 1
    # opt_output = torch.tensor(output_y_points)
    # base_loss = abs(output_y_points-label)
    # for n in range(particle_batch):
    #       fluctuation_reduction = torch.full(output_y_points.size() , 0.3 , device=torch_device)
    #       particle = torch.randn(output_y_points.size(), device=torch_device)
    #       particle = particle * fluctuation_reduction + output_y_points
    #       particle_loss = abs(particle-label)
    #       for i in range(len(output_y_points)):
    #             for j in range(len(output_y_points[0])):
    #                   for k in range(len(output_y_points[0][0])):
    #                         if particle_loss[i][j][k]<base_loss[i][j][k]:
    #                               opt_output[i][j][k] += particle[i][j][k]
    #                         else:
    #                               opt_output[i][j][k] += output_y_points[i][j][k]
    # opt_output=opt_output/(particle_batch+1)
    # print("particle_num:{}".format(particle_batch*len(output_y_points)*len(output_y_points[0])*len(output_y_points[0][0])))
    #-----------------------------------------------------------------

    #----------Combined particle filter---------
    if torch.cuda.is_available():
          torch_device = torch.device("cuda")
    opt_output_1 = output_y_points.clone()
    opt_output_2 = output_y_points.clone()
    if do_particle:
      particle_batch = 4
      base_loss = abs(output_y_points-label)
      for n in range(particle_batch):
        fluctuation_reduction = torch.full(output_y_points.size() , 0.05*(n+2) , device=torch_device)
        particle = torch.randn(output_y_points.size(), device=torch_device)
        particle = particle * fluctuation_reduction + opt_output_1
        particle_loss = abs(particle-label)
        for i in range(len(output_y_points)):
              for j in range(len(output_y_points[0])):
                    for k in range(len(output_y_points[0][0])):
                          if particle_loss[i][j][k]<base_loss[i][j][k]:
                                opt_output_1[i][j][k] += particle[i][j][k]
                          else:
                                opt_output_1[i][j][k] += output_y_points[i][j][k]
        if n == 0:
              opt_output_2 = opt_output_1.clone()/2
      opt_output_1=opt_output_1/(particle_batch+1)
      print("particle_num:{}".format(particle_batch*len(output_y_points)*len(output_y_points[0])*len(output_y_points[0][0])))

    return output_y_points, outputs_class, opt_output_1, opt_output_2 # now returning a tuple
