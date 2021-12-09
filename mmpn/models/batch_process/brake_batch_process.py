import torch.nn as nn
import torch.nn.functional as F
from ..builder import BATCH_PROCESS , build_loss
from collections import OrderedDict

@BATCH_PROCESS.register_module()
class BrakeBatchProcess(nn.Module):
    
    def __init__(self, **kwargs):
      super().__init__()
      print("kwargs:", kwargs)

      assert "loss_cls" in kwargs

      self.loss = build_loss(kwargs["loss_cls"])
   
    def __call__(self, model, data, train_mode):
        input, label = data
        label = label.cuda(non_blocking=True)
        pred = model(input)
        if train_mode:
            loss = self.loss(pred, label)

            log_vars = OrderedDict()
            log_vars['loss'] = loss.item()
            outputs = dict(loss=loss, log_vars=log_vars, num_samples=input.size(0))
            
        else:
            pass
            # calc acc ...        
        return outputs
