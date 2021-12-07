import copy
import pickle

import mmcv
import torch
import torch.distributed as dist
from mmcv import Registry, build_from_cfg
from mmcv.runner import Hook

PIPELINE_SCHEDULERS = Registry('pipeline_schedulers')


def build_pipeline_scheduler(cfg):
    return build_from_cfg(cfg, PIPELINE_SCHEDULERS)


@PIPELINE_SCHEDULERS.register_module()
class EpochPipelineScheduler(Hook):
    __instance = None
    __initialized = False

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__initialized = False

        return cls.__instance

    def __init__(self, epoch_data_pipeline, sync_max_length=1024):
        if EpochPipelineScheduler.__initialized:
            return
        EpochPipelineScheduler.__initialized = True

        self.epoch_data_pipeline_cfg = epoch_data_pipeline
        self.epoch = 0
        self.sync_max_length = sync_max_length

        self.sync_tensor = None

        self.normalize_prob()

    def normalize_prob(self):
        for _, cfg_probs in self.epoch_data_pipeline_cfg:
            sum_probs = sum([cfg_prob['prob'] for cfg_prob in cfg_probs])

            assert sum_probs > 0
            for cfg_prob in cfg_probs:
                cfg_prob['prob'] /= sum_probs

    def before_train_epoch(self, runner):
        self.epoch = runner.epoch

    def __get_data_pipeline(self, epoch):
        epoch = self.epoch if epoch < 0 else self.epoch

        for epoch_range, data_pipelines in self.epoch_data_pipeline_cfg:
            if epoch in epoch_range:
                from random import random
                r = random()
                for data_pipeline in data_pipelines:
                    if r > data_pipeline['prob']:
                        r -= data_pipeline['prob']
                    else:
                        res = copy.deepcopy(data_pipeline)
                        res.pop('prob')
                        return res

        raise Exception('Pipeline Scheduler error: %s' % (epoch))

    def get_schedule(self, epoch=-1, sync=True):
        if sync:
            rank, size = mmcv.runner.get_dist_info()

            if size == 1:
                return self.__get_data_pipeline(epoch)

            if self.sync_tensor is None:
                self.sync_tensor = torch.zeros((self.sync_max_length), dtype=torch.uint8, device='cuda')
            else:
                self.sync_tensor[:] = 0

            if rank == 0:
                tmp_tensor = torch.tensor(bytearray(pickle.dumps(self.__get_data_pipeline(epoch))), dtype=torch.uint8)
                self.sync_tensor[:len(tmp_tensor)] = tmp_tensor
            dist.broadcast(self.sync_tensor, 0)
            data_pipeline_cfg = pickle.loads(self.sync_tensor.cpu().numpy().tobytes())

            return data_pipeline_cfg
        else:
            return self.__get_data_pipeline(epoch)
