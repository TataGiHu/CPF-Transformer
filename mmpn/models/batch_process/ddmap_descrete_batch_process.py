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
from .common import collate_fn_output_mask



@BATCH_PROCESS.register_module()
class DdmapDescreteBatchProcessDC(nn.Module):
    
    def __init__(self, **kwargs):
      super().__init__()
      print("kwargs:", kwargs)

      assert "loss_reg" in kwargs
      assert "bce_loss_reg" in kwargs
      self.loss = build_loss(kwargs["loss_reg"])
      self.classification_loss = build_loss(kwargs["bce_loss_reg"])
      self.weight = kwargs["weight"] 
   
      self.weight_device = torch.Tensor(self.weight).cuda(non_blocking=True)

    def __call__(self, model, data, train_mode):

        #input_data, mask, label, meta = self.collate_fn(data)
        input_data, mask, label, classes, meta = collate_fn(data)

        pred = model(input_data, mask)

        outputs = dict()

        if train_mode:
            # zero out predictions where the lane does not exist 
            pred[0][:,:,0:20] *= classes[:,:,0:1]
            pred[0][:,:,20:40] *= classes[:,:,1:2]
            pred[0][:,:,40:60] *= classes[:,:,2:3]
            
            loss = self.loss(pred[0], label, self.weight_device)
            # left_lane_loss = classes[:,:,0] * self.loss() 
            # TODO: Tune the weighting between classification loss and prediction loss
            classification_loss = self.classification_loss(pred[1], classes)
            
            loss = loss + classification_loss
            log_vars = OrderedDict()
            log_vars['loss'] = loss.item()
            outputs = dict(loss=loss, log_vars=log_vars, num_samples=input_data.size(0))
            
        else:
            outputs = dict(pred=pred, meta=meta)
            # calc acc ...        
        return outputs


@BATCH_PROCESS.register_module()
class DdmapDescreteBatchProcessDCThreeQueries(nn.Module):
    
    def __init__(self, **kwargs):
      super().__init__()
      print("kwargs:", kwargs)

      assert "loss_reg" in kwargs
      assert "bce_loss_reg" in kwargs
      self.loss = build_loss(kwargs["loss_reg"])
      self.classification_loss = build_loss(kwargs["bce_loss_reg"])
      self.weight = kwargs["weight"] 
   
      self.weight_device = torch.Tensor(self.weight).cuda(non_blocking=True)

    def __call__(self, model, data, train_mode):

        input_data, mask, label, classes, meta, output_mask, furthest_point_mark= collate_fn_output_mask(data)
        pred = model(input_data, mask)
        outputs = dict()
        
        if train_mode:
            pred[0][:,0] *= classes[:,:,0]
            pred[0][:,1] *= classes[:,:,1]
            pred[0][:,2] *= classes[:,:,2]
            pred_on_mask = pred[0] * output_mask
            label_on_mask = label * output_mask
            loss = self.loss(pred_on_mask, label_on_mask, self.weight_device)
            classes_reshape = classes.view(-1,3,1)
            classification_loss = self.classification_loss(pred[1], classes_reshape)
            loss = loss + classification_loss
            log_vars = OrderedDict()
            log_vars['loss'] = loss.item()
            outputs = dict(loss=loss, log_vars=log_vars, num_samples=input_data.size(0))
            
        else:
            outputs = dict(pred=pred, meta=meta, furthest_point_mark=furthest_point_mark)
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
      pred = outputs['pred'][0].cpu().numpy().tolist() 
      classification = outputs['pred'][1].cpu().numpy().tolist() 
      meta = outputs['meta'].cpu().numpy().tolist()

      prediction = []
      frame_lanes = []
      current_lane = []
      for prediction_lane_y in pred:
            for i, prediction_point_y in enumerate(prediction_lane_y[0]):
                  current_lane.append([-20 + 5 * (i % 20), prediction_point_y])
                  if ((i + 1) % 20) == 0:
                        frame_lanes.append(current_lane)
                        current_lane = []
                        continue
            prediction.append(frame_lanes)
            frame_lanes = []
            # frame_lanes.append(current_lane)
                  
      
      for batch_pred, batch_class,  me in zip(prediction, classification, meta):
        res = json.dumps(dict(pred=batch_pred, score=batch_class[0], ts=me[0][1]))
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
      
      
      
@CUSTOM_HOOKS.register_module()
class DdmapDescreteTestHookThreeQueries(Hook):
    def __init__(self):
      self.result = []
      pass

    def after_val_iter(self, runner): 
      outputs = runner.outputs
      pred = outputs['pred'][0].cpu().numpy().tolist() 
      classification = outputs['pred'][1].cpu().numpy().tolist() 
      meta = outputs['meta'].cpu().numpy().tolist()
      furthest_point_mark = outputs['furthest_point_mark'].cpu().numpy().tolist()
      prediction = []
      frame_lanes = []
      current_lane = []

      for k,prediction_frame_y in enumerate(pred):
            for prediction_lane_y in prediction_frame_y:
                  for i, prediction_point_y in enumerate(prediction_lane_y):
                      if i < furthest_point_mark[k]:
                        current_lane.append([-20 + 5 * (i%25), prediction_point_y])
                  frame_lanes.append(current_lane)
                  current_lane = []
            prediction.append(frame_lanes)
            frame_lanes = []
              
               
      for batch_pred, batch_class,  me in zip(prediction, classification, meta):
        class_reshape = []
        for item in batch_class:
              class_reshape+=item
        res = json.dumps(dict(pred=batch_pred, score=class_reshape, ts=me[0][1]))
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