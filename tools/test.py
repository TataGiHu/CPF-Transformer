import os
import argparse
import os.path as osp
import time
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
this_dir = osp.abspath(osp.join(osp.dirname('__file__')))
add_path(this_dir)

#projects_dir = this_dir[:(this_dir.index('mmpn'))]
#add_path(osp.join(projects_dir, 'mmpn'))

from mmcv import Config
from mmpn import exps

import mmcv
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='.', help='path to config file')
    parser.add_argument('--ckpt_file', type=str, default='', help='path to model file')
    parser.add_argument('--dist', type=str, default='False', help='DDP')
    parser.add_argument('--local_rank', type=int, default=-1, help='node rank for distributed training')
    args = parser.parse_args()
        
    cfg = Config.fromfile(args.config_file)
    
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
     
    if eval(args.dist):
        from mmcv.runner.dist_utils import init_dist
        init_dist('pytorch', **cfg.dist_params)

    
    exp = exps.build_experiment(cfg)

    ckpt_file = args.ckpt_file
    if not ckpt_file:
      ckpt_file = os.path.join(cfg.work_dir, "latest.pth")
    print("load from ckpt_file: {}".format(ckpt_file))
    exp.test(ckpt_file=ckpt_file, dist=eval(args.dist))
    

if __name__ == '__main__':
    main()

