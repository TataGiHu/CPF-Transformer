import os
import pickle
import json
import numpy as np
from numpy.lib.shape_base import expand_dims
from torch.utils.data import Dataset
from ..builder import DATASETS
from mmcv.parallel import DataContainer
import torch

@DATASETS.register_module()
class DdmapDescreteDatasetDC(Dataset):
  def __init__(self, data_path):
    self.datas_input = []    
    self.gts_input = []
    self.meta = []
    for file in os.listdir(data_path):
      if ".txt" not in file:
        continue
      with open(data_path + "/" + file, 'r') as f:
        data_raw = f.readlines()
        self.data_info = data_raw.pop(0)    


        for data in data_raw: 
          data_json = json.loads(data)
          gt = data_json['gt']
          gt_input = []
          for lane in gt:
            if len(lane) == 0:
                  continue
            for lane_point in lane:
              gt_input.append(lane_point[1])
          if len(gt_input)!= 0:
            self.gts_input.append(gt_input)
          else:
            continue
            
          dt = data_json['dt']
          n_frame_lanes = dt["lanes"]
          data_input = []
          for frame_lane in n_frame_lanes:
            if len(frame_lane) == 0:
              continue
            data_input.extend(frame_lane[0]) 
          if len(data_input) == 0:
            continue
          self.datas_input.append(data_input)



          ts = data_json["ts"]
          self.meta.append(ts)

  def __len__(self):
    return len(self.datas_input)

  def __getitem__(self, idx):
    x = np.array(self.datas_input[idx], dtype=np.float32)
    y = np.array(self.gts_input[idx], dtype=np.float32) # just for interface placeholder        
    y = np.expand_dims(y, axis = 0)

    meta_dict = self.meta[idx]
    meta = np.array([meta_dict['wm'], meta_dict['egopose'], meta_dict['vision']], dtype=np.int64) # just for interface placeholder        
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
class DdmapDescreteDatasetWithRoadEdge(Dataset):
  def __init__(self, data_path):
    self.datas_input = []    
    self.gts_input = []
    self.meta = []
    for file in os.listdir(data_path):
      if ".txt" not in file:
        continue
      with open(data_path + "/" + file, 'r') as f:
        data_raw = f.readlines()
        self.data_info = data_raw.pop(0)    


        for data in data_raw: 
          data_json = json.loads(data)
          gt = data_json['gt']
          gt_input = []
          class_input = []
          for lane in gt:
            if len(lane) == 0:
                  class_input.append(0)
                  for i in range(20):
                        gt_input.append(0)
                  # gt_input.append([0 for i in range(20)])
                  continue
            class_input.append(1)
            for lane_point in lane:
              gt_input.append(lane_point[1])
          if len(gt_input)!= 0:
            self.gts_input.append([gt_input,class_input])
          else:
            continue

          dt = data_json['dt']
          n_frame_lanes = dt["lanes"]
          n_frame_road_edges = dt["road_edges"]
          data_input = []
          
          assert len(n_frame_lanes) == len(n_frame_road_edges)
          for i in range(len(n_frame_lanes)):
                for point in n_frame_lanes[i]:
                      data_input.extend(point)
                for point in n_frame_road_edges[i]:
                      data_input.extend(point)
          if len(data_input) == 0:
            continue
          self.datas_input.append(data_input)



          ts = data_json["ts"]
          self.meta.append(ts)

  def __len__(self):
    return len(self.datas_input)

  def __getitem__(self, idx):
    x = np.array(self.datas_input[idx], dtype=np.float32)
    y = np.array(self.gts_input[idx][0], dtype=np.float32) # just for interface placeholder  
    existence = np.array(self.gts_input[idx][1], dtype=np.float32)     #Determine if a centerline exists
    y = np.expand_dims(y, axis = 0)
    existence = np.expand_dims(existence, axis=0)

    meta_dict = self.meta[idx]
    meta = np.array([meta_dict['wm'], meta_dict['egopose'], meta_dict['vision']], dtype=np.int64) # just for interface placeholder        
    meta = np.expand_dims(meta, axis = 0)

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    existence = torch.from_numpy(existence)
    meta = torch.from_numpy(meta)

    data = {
      "x": x,
      "y": y,
      "existence": existence,
      "meta": meta
    }
    data = DataContainer(data)
    return data
    
    
    
    





@DATASETS.register_module()
class DdmapDescreteDataset(Dataset):
  def __init__(self, data_path):
    
    self.datas_input = []    
    self.gts_input = []
    self.meta = []
    for file in os.listdir(data_path):
      if ".txt" not in file:
        continue
      with open(data_path + "/" + file, 'r') as f:
        data_raw = f.readlines()
        self.data_info = data_raw.pop(0)    


        for data in data_raw: 
          data_json = json.loads(data)
          dt = data_json['dt']
          n_frame_lanes = dt["lanes"]
          data_input = []
          for frame_lane in n_frame_lanes:
            if len(frame_lane) == 0:
              continue
            data_input.extend(frame_lane[0]) 
          if len(data_input) == 0:
            continue
          self.datas_input.append(data_input)

          gt = data_json['gt']
          gt_input = []
          for lane in gt:
            if len(lane) == 0:
                  continue
            for lane_point in lane:
              gt_input.append(lane_point[1])
          self.gts_input.append(gt_input)

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
    
    
    
    