import torch.nn as nn
import torch.nn.functional as F
from ..builder import BATCH_PROCESS , build_loss, CUSTOM_HOOKS
from collections import OrderedDict
from mmcv.runner.hooks.hook import HOOKS, Hook
from mmcv.parallel import DataContainer
import torch
import json,os
from torch.utils.data.dataloader import default_collate


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
        input_data, mask, label, meta = self.collate_fn(data)

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


    def collate_fn(self, batch) :

      gpu_index = 0

      batch_per_gpu = batch.data[gpu_index]

      max_length = 0
      pad_dim = 0
      for i in range(len(batch_per_gpu)):
        max_length = max(max_length, batch_per_gpu[i]['x'].size(pad_dim))
      
      padded_samples = []
      padded_masks = []
      for sample in batch_per_gpu:
        ori_len = sample['x'].size(pad_dim)

        pad = (0,0,0,max_length-ori_len)
        padded_samples.append(F.pad(sample['x'], pad, value = 0))

        mask = torch.zeros(ori_len).bool()
        pad = torch.ones(max_length-ori_len).bool()
        padded_masks.append(torch.cat((mask, pad), 0))

      labels = []
      for i in range(len(batch_per_gpu)):
        labels.append(batch_per_gpu[i]['y'])

      metas = []
      for i in range(len(batch_per_gpu)):
        metas.append(batch_per_gpu[i]['meta'])

      input_datas = default_collate(padded_samples).cuda()
      masks = default_collate(padded_masks).cuda()
      labels = default_collate(labels).cuda()
      metas = default_collate(metas)

      return input_datas, masks, labels, metas




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

        input_data, mask, label, meta = self.collate_fn(data)

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


    def collate_fn(self, batch) :
      input_data, label, meta  = data

      mask = torch.full((input_data.shape[0], input_data.shape[1]), False, dtype=torch.bool)

      input_data = input_data.cuda(non_blocking=True)
      mask = mask.cuda(non_blocking=True)
      label = label.cuda(non_blocking=True)

      return input_data, mask, label, meta


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