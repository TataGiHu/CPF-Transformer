from collections import OrderedDict

import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.parallel import scatter, is_module_wrapper

from mmcv.runner import load_checkpoint

from ..data import build_data_provider
# from ..utils import get_logger, is_distributed, is_root
from .builder import EXPERIMENTS
from ..core.runner import MRunner

from mmcv.runner.hooks.hook import HOOKS, Hook
from mmcv.runner import get_dist_info, DistSamplerSeedHook, build_optimizer
from mmcv.runner import Runner, save_checkpoint

import mmcv
from ..models.builder import build_batch_process, build_submodel, build_custom_hook


import logging
def get_logger(name='mpn', log_file=None, log_level=logging.INFO):
    if isinstance(log_file, str):
        mmcv.mkdir_or_exist(os.path.dirname(log_file))
    return mmcv.get_logger(name, log_file, log_level)


@EXPERIMENTS.register_module()
class BaseExp:
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = get_logger()        
        
        self.dp = None
        self.model = None
        self.optimizer = None
        self.pipeline_scheduler = None
        self.test_pipeline_scheduler = None
        self.evaluators = None
        
        self.loss = None
        self.batch_process = None
        self.hooks = []
        
    def build_data_provider(self):
        if self.dp is not None:
            return self.dp
        
        self.dp = build_data_provider(self.cfg.data_provider)

        
    def build_model(self):
        if self.model is not None:
            return self.model
        
        self.model = build_submodel(self.cfg.model)

        ###########################################
   
    def build_batch_process(self):
        if self.batch_process is not None:
            return self.batch_process
       
        self.batch_process = build_batch_process(self.cfg.batch_process)
    def build_hook(self):
        if "hooks" not in self.cfg:
          return 

        for hook in self.cfg.hooks:
          self.hooks.append(build_custom_hook(hook))

    def build_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        self.optimizer = build_optimizer(self.model, self.cfg.optimizer)
        return self.optimizer
    
    """def build_pipeline_scheduler(self):
        if self.pipeline_scheduler is not None:
            return self.pipeline_scheduler

        from ..core.pipeline_scheduler import build_pipeline_scheduler
        self.pipeline_scheduler = build_pipeline_scheduler(self.cfg.pipeline_scheduler)

        return self.pipeline_scheduler
    
    def build_test_pipeline_scheduler(self):
        if self.test_pipeline_scheduler is not None:
            return self.test_pipeline_scheduler

        from ..core.pipeline_scheduler import build_pipeline_scheduler
        self.test_pipeline_scheduler = build_pipeline_scheduler(self.cfg.test_pipeline_scheduler)

        return self.test_pipeline_scheduler"""
    
    def build_evaluators(self):
        pass
    
    def train(self, dist=False):
        
        # Step 1) build pipeline scheduler assert
        #self.build_pipeline_scheduler()
      
        # Step 2) build data
        self.build_data_provider()
        stages = [stage for stage, _ in self.cfg.workflow]
        self.dp.build_loaders(stages=stages, dist=dist)
        
        # Step 3) build model
        self.build_model()
        device_nums = torch.cuda.device_count()
        device = range(device_nums)
        if dist:
            find_unused_parameters = self.cfg.get('find_unused_parameters', True)
            self.model = MMDistributedDataParallel( # 
                self.model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            self.model = MMDataParallel(self.model.cuda(), device_ids=device)


        # Step 4) build optimizer, scheduler
        self.build_optimizer()
        
        # Step 5) build runner
        #######################################

        self.build_batch_process()

        runner = MRunner(
            model=self.model,
            batch_processor=self.batch_process,
            optimizer=self.optimizer, 
            work_dir=self.cfg.work_dir,
            logger=self.logger,
        )
        
        
        runner.register_training_hooks(
            lr_config=self.cfg.lr_config,
            optimizer_config=self.cfg.optimizer_config,
            checkpoint_config=self.cfg.checkpoint_config,
            log_config=self.cfg.log_config
        )

        #runner.register_hook(self.pipeline_scheduler)
        #runner.register_hook(DataLoaderEpochUpdateHook())
        
        if dist:
            runner.register_hook(DistSamplerSeedHook())
        
        if self.cfg.get('resume_from', None):
            runner.resume(self.cfg.resume_from)
        elif self.cfg.get('load_from', None):
            runner.load_checkpoint(self.cfg.load_from)
        
        # Step 6) run
        with torch.autograd.detect_anomaly():
            runner.run(self.dp.get_loaders(stages), self.cfg.workflow, self.cfg.total_epochs)



    def test(self, ckpt_file, dist=False):
        
        # Step 1) build pipeline scheduler assert
        #self.build_pipeline_scheduler()
      
        # Step 2) build data
        assert "val" in self.cfg.data_provider.stages 

        self.build_data_provider()

        stages = ['val']
        self.dp.build_loaders(stages=stages, dist=dist)
        
        # Step 3) build model
        self.build_model()
        
        device_nums = torch.cuda.device_count()
        device = range(device_nums)
        if dist:
            find_unused_parameters = self.cfg.get('find_unused_parameters', True)
            self.model = MMDistributedDataParallel( # 
                self.model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            self.model = MMDataParallel(self.model.cuda(), device_ids=device)

        # Step 5) build runner
        #######################################

        self.build_batch_process()

        runner = MRunner(
            model=self.model,
            batch_processor=self.batch_process,
            work_dir=self.cfg.work_dir,
            logger=self.logger,
        )

        self.build_hook()
        for hook in self.hooks:
          runner.register_hook(hook)

        if dist:
            runner.register_hook(DistSamplerSeedHook())
        runner.load_checkpoint(ckpt_file)

        # Step 6) run
        runner.val(self.dp.get_loaders(stages)[0])



