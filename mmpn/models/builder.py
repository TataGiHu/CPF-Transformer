from mmcv import Registry, build_from_cfg

PIPELINES = Registry('pipelines')
SMODELS = Registry('sub_models')
LOSSES = Registry('losses')
BATCH_PROCESS = Registry('batch_process')

def build_submodel(cfg):
    return build_from_cfg(cfg, SMODELS)

def build_loss(cfg):
    return build_from_cfg(cfg, LOSSES)

def build_batch_process(cfg):
    return build_from_cfg(cfg, BATCH_PROCESS)