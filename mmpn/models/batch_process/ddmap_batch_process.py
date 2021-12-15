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
class DdmapBatchProcessDC(nn.Module):
    
    def __init__(self, **kwargs):
      super().__init__()
      print("kwargs:", kwargs)

      assert "loss_reg" in kwargs
      self.loss = build_loss(kwargs["loss_reg"])
      self.loss_process_type = kwargs["loss_process_type"]

      if self.loss_process_type == "weight":
        self.weight = kwargs["weight"] 
   
        self.weight_device = torch.Tensor(self.weight).cuda(non_blocking=True)


    def __call__(self, model, data, train_mode):

        input_data, mask, label, meta = collate_fn(data)

        pred = model(input_data, mask)

        if train_mode:
            loss = self.process_loss(pred,label)
            log_vars = OrderedDict()
            log_vars['loss'] = loss.item()
            outputs = dict(loss=loss, log_vars=log_vars, num_samples=input_data.size(0))
            
        else:
            outputs = dict(pred=pred, meta=meta)
            # calc acc ...        
        return outputs

    def process_loss(self, pred, label): 
      if self.loss_process_type == "normal":
        loss = self.loss(pred, label)
        return loss 

      if self.loss_process_type == "weight":
        loss = self.loss(pred, label, self.weight_device)
        return loss
      
      if self.loss_process_type == "points":

        def change_to_points_losss(some_res):
          b = torch.arange(-20, 81, 5)
          a = b * b 
          c = torch.ones_like(b)

          coor = torch.stack((c,b,a), dim=0).float().cuda()
          processed_res = some_res.matmul(coor)
          return processed_res

        pred = change_to_points_losss(pred)
        label = change_to_points_losss(label)

        loss = self.loss(pred, label)
        return loss




@CUSTOM_HOOKS.register_module()
class DdmapTestCoeffHook(Hook):
    def __init__(self):
      self.result = []
      info = {"type":"coeff"}
      self.result.append(json.dumps(info))

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