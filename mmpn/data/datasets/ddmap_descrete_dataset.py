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
class DdmapDescreteDatasetWithRoadEdgeAndDashedAttribute(Dataset):
  def __init__(self, data_path):
    self.datas_input = []    
    self.gts_input = []
    self.meta = []
    self.output_mask = []
    self.furthest_point_mark = []
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
                  for i in range(24):
                        gt_input.append(0)
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
          data_len = 100
          furthest_point_x = -float('inf')
          assert len(n_frame_lanes) == len(n_frame_road_edges)
          ##########################################################################
          #input unit: lane
          #input format: [{lane(0)/edge(1)},{solid(0)/dashed(1)},x1,y1,...,x100.y100]
          #input sample: [1. , 0. , 3.2123 , 4.5272 , ...]
          ##########################################################################
          #upload lanes into data_input
          for frame in n_frame_lanes:
            if len(frame)==0:
              continue
            else:
              for lane in frame:
                cur_lane = [0.]
                if lane["lane_type"]==1:
                  cur_lane.append(1.)
                else:
                  cur_lane.append(0.)  
                for i in range(data_len):
                  if i<len(lane["points"]):
                    cur_lane.extend(lane["points"][i])
                    if lane["points"][i][0] > furthest_point_x:
                      furthest_point_x = lane["points"][i][0]
                  else:
                    cur_lane.extend([0.,0.])
                data_input.append(cur_lane)
          #upload road_edges into data_input
          for frame in n_frame_road_edges:
            if len(frame)==0:
              continue
            else:
              for lane in frame:
                cur_lane = [1.,0.]
                for i in range(data_len):
                  if i<len(lane):
                    cur_lane.extend(lane[i])
                  else:
                    cur_lane.extend([0.,0.])
                
          if len(data_input) == 0:
            continue
          self.datas_input.append(data_input)
             
          ts = data_json["ts"]
          self.meta.append(ts)
          
          #generate output mask
          output_mask_cur = []
          output_mask_begin = -20
          output_mask_end = 100
          output_mask_step = 5
          furthest_mark = output_mask_begin
          for i in range(output_mask_begin, output_mask_end, output_mask_step):
            if furthest_point_x <= i:
              furthest_mark = i
              break
          num_reserve = int((furthest_mark-output_mask_begin)/output_mask_step+1)
          num_discard = int((output_mask_end-output_mask_begin)/output_mask_step-num_reserve)
          output_mask_unit = [1.]*num_reserve
          discard_mask = [0.]*num_discard
          output_mask_unit.extend(discard_mask)
          for i in range(len(gt)):
            output_mask_cur.append(output_mask_unit)
          self.output_mask.append(output_mask_cur)
          #restore the furthest lane length
          self.furthest_point_mark.append(num_reserve)
          
  def __len__(self):
    return len(self.datas_input)

  def __getitem__(self, idx):
    x = np.array(self.datas_input[idx], dtype=np.float32)
    y = np.array(self.gts_input[idx][0], dtype=np.float32) # just for interface placeholder  
    existence = np.array(self.gts_input[idx][1], dtype=np.float32)     #Determine if a centerline exists
    y = np.expand_dims(y, axis = 0).reshape(-1,24)
    existence = np.expand_dims(existence, axis=0)
    meta_dict = self.meta[idx]
    meta = np.array([meta_dict['wm'], meta_dict['egopose'], meta_dict['vision']], dtype=np.int64) # just for interface placeholder        
    meta = np.expand_dims(meta, axis = 0)
    output_mask = np.array(self.output_mask[idx], dtype=np.float32)
    furthest_point_mark = np.array(self.furthest_point_mark[idx], dtype=np.float32)
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    existence = torch.from_numpy(existence)
    meta = torch.from_numpy(meta)
    output_mask = torch.from_numpy(output_mask)
    furthest_point_mark = torch.from_numpy(furthest_point_mark)
    data = {
      "x": x,
      "y": y,
      "existence": existence,
      "meta": meta,
      "output_mask": output_mask,
      "furthest_point_mark": furthest_point_mark
      
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
    
    
    
    