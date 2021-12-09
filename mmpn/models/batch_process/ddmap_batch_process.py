import torch.nn as nn
import torch.nn.functional as F
from ..builder import BATCH_PROCESS , build_loss
from collections import OrderedDict
import torch

@BATCH_PROCESS.register_module()
class DdmapBatchProcess(nn.Module):
    
    def __init__(self, **kwargs):
      super().__init__()
      print("kwargs:", kwargs)

      assert "loss_reg" in kwargs

      self.loss = build_loss(kwargs["loss_reg"])
   
    def __call__(self, model, data, train_mode):

        input_data, mask, label = self.collate_fn(data)

        pred = model(input_data, mask)

        if train_mode:
            loss = self.loss(pred, label)

            log_vars = OrderedDict()
            log_vars['loss'] = loss.item()
            outputs = dict(loss=loss, log_vars=log_vars, num_samples=input_data.size(0))
            
        else:
            pass
            # calc acc ...        
        return outputs


    def collate_fn(self, data) :
      input_data, label  = data

      mask = torch.full((input_data.shape[0], input_data.shape[1]), False, dtype=torch.bool)

      input_data = input_data.cuda(non_blocking=True)
      mask = mask.cuda(non_blocking=True)
      label = label.cuda(non_blocking=True)

      return input_data, mask, label