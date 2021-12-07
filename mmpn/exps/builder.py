from mmcv import Registry, build_from_cfg

EXPERIMENTS = Registry('experiments')

def build_experiment(cfg):
    exp_cfg = dict(
        type=cfg.exp_type,
        cfg=cfg
    )
    return build_from_cfg(exp_cfg, EXPERIMENTS)