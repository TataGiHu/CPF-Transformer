import torch.nn as nn
import torch.nn.functional as F
from ..builder import BATCH_PROCESS , build_loss, CUSTOM_HOOKS
from collections import OrderedDict
from mmcv.runner.hooks.hook import HOOKS, Hook
import torch
import json,os

@BATCH_PROCESS.register_module()
class DdmapBatchProcess(nn.Module):
    
    def __init__(self, **kwargs):
      super().__init__()
      print("kwargs:", kwargs)

      assert "loss_reg" in kwargs
      self.loss = build_loss(kwargs["loss_reg"])
   
    def __call__(self, model, data, train_mode):

        input_data, mask, label, meta = self.collate_fn(data)

        pred = model(input_data, mask)

        outputs = dict()

        if train_mode:
            loss = self.loss(pred, label)

            log_vars = OrderedDict()
            log_vars['loss'] = loss.item()
            outputs = dict(loss=loss, log_vars=log_vars, num_samples=input_data.size(0))
            
        else:
            outputs = dict(pred=pred, meta=meta)
            # calc acc ...        
        return outputs


    def collate_fn(self, data) :
      input_data, label, meta  = data

      mask = torch.full((input_data.shape[0], input_data.shape[1]), False, dtype=torch.bool)

      input_data = input_data.cuda(non_blocking=True)
      mask = mask.cuda(non_blocking=True)
      label = label.cuda(non_blocking=True)

      return input_data, mask, label, meta


@CUSTOM_HOOKS.register_module()
class DdmapTestHook(Hook):
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

    def after_val_epoch(self, runner):

      work_dir = runner.work_dir
      file_name = os.path.join(work_dir, "preds.txt")
      with open(file_name, 'w') as fout:
        for res in self.result:
          fout.write(res+"\n")
      print("val results save to {}".format(file_name))