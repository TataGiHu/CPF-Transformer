from mmcv import Registry, build_from_cfg

PIPELINES = Registry('pipelines')
SMODELS = Registry('sub_models')
LOSSES = Registry('losses')

def build_submodel(cfg):
    return build_from_cfg(cfg, SMODELS)

def build_loss(cfg):
    return build_from_cfg(cfg, LOSSES)