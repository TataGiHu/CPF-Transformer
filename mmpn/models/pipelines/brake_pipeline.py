import torch.nn as nn
from ..builder import PIPELINES

@PIPELINES.register_module()
class BrakePipeline(nn.Module):
    def __init__(self, smodels, brake_model, **kwargs):
        super().__init__(**kwargs)

        assert smodels is not None
        assert brake_model in smodels
        self.brake_model = smodels[brake_model]

    @module_result
    def forward(self, return_loss=False, return_preds=False, **kwargs):
        res = self.det_model.forward(frames=kwargs, return_loss=return_loss, return_preds=return_preds)
        return res
