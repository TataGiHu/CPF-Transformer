import torch.nn as nn
import torch.nn.functional as F
from ..builder import BATCH_PROCESS , build_loss, CUSTOM_HOOKS
from collections import OrderedDict
from mmcv.runner.hooks.hook import HOOKS, Hook
from mmcv.parallel import DataContainer
import torch
import json,os
from torch.utils.data.dataloader import default_collate
from .common import collate_fn



@BATCH_PROCESS.register_module()
class DdmapDescreteBatchProcessDC(nn.Module):
    
    def __init__(self, **kwargs):
      super().__init__()
      print("kwargs:", kwargs)

      assert "loss_reg" in kwargs
      self.loss = build_loss(kwargs["loss_reg"])
      self.weight = kwargs["weight"] 
   
      self.weight_device = torch.Tensor(self.weight).cuda(non_blocking=True)

    def __call__(self, model, data, train_mode):

        #input_data, mask, label, meta = self.collate_fn(data)
        input_data, mask, label, meta = collate_fn(data)

        pred = model(input_data, mask)

        outputs = dict()

        if train_mode:
            loss = self.loss(pred, label, self.weight_device)

            log_vars = OrderedDict()
            log_vars['loss'] = loss.item()
            outputs = dict(loss=loss, log_vars=log_vars, num_samples=input_data.size(0))
            
        else:
            outputs = dict(pred=pred, meta=meta)
            # calc acc ...        
        return outputs


@CUSTOM_HOOKS.register_module()
class DdmapDescreteTestDCHook(Hook):
    def __init__(self):
      self.result = []
      pass

    def after_val_iter(self, runner): 
      outputs = runner.outputs
      pred = outputs['pred'].cpu().numpy().tolist() 
      meta = outputs['meta'].cpu().numpy().tolist()

      for batch_pred, me in zip(pred, meta):
        res = json.dumps(dict(pred=batch_pred,ts=me[0]))
        self.result.append(res)
      pass

    def after_val_epoch(self, runner):

      work_dir = runner.work_dir
      file_name = os.path.join(work_dir, "preds.txt")
      with open(file_name, 'w') as fout:
        for res in self.result:
          fout.write(res+"\n")
      print("val results save to {}".format(file_name))


@BATCH_PROCESS.register_module()
class DdmapDescreteBatchProcess(nn.Module):
    
    def __init__(self, **kwargs):
      super().__init__()
      print("kwargs:", kwargs)

      assert "loss_reg" in kwargs
      self.loss = build_loss(kwargs["loss_reg"])
      self.weight = kwargs["weight"] 
   
      self.weight_device = torch.Tensor(self.weight).cuda(non_blocking=True)

    def __call__(self, model, data, train_mode):

        input_data, mask, label, meta = collate_fn(data)

        pred = model(input_data, mask)

        outputs = dict()

        if train_mode:
            loss = self.loss(pred, label, self.weight_device)

            log_vars = OrderedDict()
            log_vars['loss'] = loss.item()
            outputs = dict(loss=loss, log_vars=log_vars, num_samples=input_data.size(0))
            
        else:
            outputs = dict(pred=pred, meta=meta)
            # calc acc ...        
        return outputs


@CUSTOM_HOOKS.register_module()
class DdmapDescreteTestHook(Hook):
    def __init__(self):
      self.result = []
      pass

    def after_val_iter(self, runner): 
      outputs = runner.outputs
      pred = outputs['pred'].cpu().numpy().tolist() 
      meta = outputs['meta'].cpu().numpy().tolist()

      prediction = []
      frame_lanes = []
      for prediction_lane_y in pred:
            current_lane = []
            for i, prediction_point_y in enumerate(prediction_lane_y[0]):
                  current_lane.append([-20 + 5 * i, prediction_point_y])
            frame_lanes.append(current_lane)
      prediction.append(frame_lanes)
                  
            
      for batch_pred, me in zip(prediction[0], meta):
        res = json.dumps(dict(pred=[batch_pred],ts=me[0][1]))
        self.result.append(res)
      pass

    def after_val_epoch(self, runner):

      work_dir = runner.work_dir
      file_name = os.path.join(work_dir, "preds.txt")
      with open(file_name, 'w') as fout:
        fout.write(json.dumps(dict(type="points")) + "\n")
        for res in self.result:
          fout.write(res+"\n")
      print("val results save to {}".format(file_name))