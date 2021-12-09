import os
import pickle
import json
import numpy as np
from torch.utils.data import Dataset
from ..builder import DATASETS

@DATASETS.register_module()
class DdmapDataset(Dataset):
  def __init__(self, data_path):
    
    with open(data_path, 'r') as f:
      data_raw = f.readlines()

    self.data_info = data_raw.pop(0)    

    self.datas_input = []    
    self.gts_input = []

    for data in data_raw: 
      data_json = json.loads(data)
      dt = data_json['dt']
      n_frame_lanes = dt["lanes"]
      data_input = []
      for frame_lane in n_frame_lanes:
        data_input.extend(frame_lane) 
      self.datas_input.append(data_input)

      gt = data_json['gt']
      self.gts_input.append(gt)
  
  def __len__(self):
    return len(self.datas_input)

  def __getitem__(self, idx):
    x = np.array(self.datas_input[idx], dtype=np.float32)
    y = np.array(self.gts_input[idx], dtype=np.float32) # just for interface placeholder        
    y = np.expand_dims(y, axis = 0)

    return x, y
    
    
    
    