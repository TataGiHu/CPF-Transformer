import os
import pickle
import json
import numpy as np
from torch.utils.data import Dataset
from ..builder import DATASETS
from mmcv.parallel import DataContainer
import torch

@DATASETS.register_module()
class DdmapDatasetDC(Dataset):
  def __init__(self, data_path):
    
    with open(data_path, 'r') as f:
      data_raw = f.readlines()

    self.data_info = data_raw.pop(0)    

    self.datas_input = []    
    self.gts_input = []
    self.meta = []

    for data in data_raw: 
      data_json = json.loads(data)
      dt = data_json['dt']
      n_frame_lanes = dt["lanes"]
      data_input = []
      for frame_lane in n_frame_lanes:
        data_input.extend(frame_lane) 
      if len(data_input) == 0:
        continue
      self.datas_input.append(data_input)

      gt = data_json['gt'][0]
      self.gts_input.append(gt)

      ts = data_json["ts"]["egopose"]
      self.meta.append(ts)

  def __len__(self):
    return len(self.datas_input)

  def __getitem__(self, idx):
    x = np.array(self.datas_input[idx], dtype=np.float32)
    y = np.array(self.gts_input[idx], dtype=np.float32) # just for interface placeholder        
    y = np.expand_dims(y, axis = 0)

    meta = np.array(self.meta[idx], dtype=np.int64) # just for interface placeholder        
    meta = np.expand_dims(meta, axis = 0)

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    meta = torch.from_numpy(meta)

    data = {
      "x": x,
      "y": y,
      "meta": meta
    }
    data = DataContainer(data)
    return data
    
    
    
    





@DATASETS.register_module()
class DdmapDataset(Dataset):
  def __init__(self, data_path):
    
    with open(data_path, 'r') as f:
      data_raw = f.readlines()

    self.data_info = data_raw.pop(0)    

    self.datas_input = []    
    self.gts_input = []
    self.meta = []

    for data in data_raw: 
      data_json = json.loads(data)
      dt = data_json['dt']
      n_frame_lanes = dt["lanes"]
      data_input = []
      for frame_lane in n_frame_lanes:
        data_input.extend(frame_lane) 
      if len(data_input) == 0:
        continue
      self.datas_input.append(data_input)

      gt = data_json['gt']
      self.gts_input.append(gt)

      ts = data_json["ts"]
      self.meta.append(ts)

  def __len__(self):
    return len(self.datas_input)

  def __getitem__(self, idx):
    x = np.array(self.datas_input[idx], dtype=np.float32)
    y = np.array(self.gts_input[idx], dtype=np.float32) # just for interface placeholder        
    y = np.expand_dims(y, axis = 0)

    meta = np.array(self.meta[idx], dtype=np.int64) # just for interface placeholder        
    meta = np.expand_dims(meta, axis = 0)


    return x, y, meta
    
    
    
    