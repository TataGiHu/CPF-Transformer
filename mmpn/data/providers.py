from abc import ABC, abstractmethod

from .builder import build_data_provider, DATAPROVIDERS

DEFAULT_STAGES = ('train', 'test', 'val')

class BaseProvider(ABC):
    def __init__(self):
        self.loaders = dict()

    @abstractmethod
    def build_loaders(self, stages=DEFAULT_STAGES):
        pass

    def get_loaders(self, stages=DEFAULT_STAGES):
        if stages is None:
            stages = list(self.loaders.keys())

        assert isinstance(stages, list)
        assert all([stage in self.loaders for stage in stages])
        return [self.loaders[stage] for stage in stages]

    
@DATAPROVIDERS.register_module()
class DataLoader(BaseProvider):
    def __init__(self, samples_per_gpu=1, workers_per_gpu=1, stages=None, *args, **kwargs):
        """
        Args:
            samples_per_gpu (int):  number of samples sampled on each gpu
            workers_per_gpu (int):  number of workers forked per gpu
            stages (Dict[stage_name, Dict]): (stage_name, dataset_config) pair 
                of each stage, e.g. {'train', 'test', 'val'}
        """
        super().__init__()
        self.samples_per_gpu = samples_per_gpu
        self.workers_per_gpu = workers_per_gpu
        assert stages is not None and isinstance(stages, dict)
        self.stages2cfg = stages

    def build_loaders(self, stages=DEFAULT_STAGES, dist=False):
        """
        Builder funciton of dataloaders for all stages.

        Args:
            stages (Iterable[str]): list of stage
            distributed (bool): distributed flag, will be passed to `build_dataloader` (Adaptor 
                between MDetection & PyTorch), e.g. whether using PyTorch's DistributedSampler or not
        
        Returns:
            The built loaders will be stored in `self.loaders`.

        """
        for stage in stages:
            if stage not in self.stages2cfg:
                continue
            cfg = self.stages2cfg[stage]
            from .builder import bulid_dataset, build_dataloader
            dataset = bulid_dataset(cfg) #, dict(test_mode=True) if not stage.startswith('train') else None)
   
            #samples_per_gpu = self.samples_per_gpu if stage.startswith('train') else 1
            
            dataloader = build_dataloader(dataset,
                                self.samples_per_gpu,
                                self.workers_per_gpu,
                                num_gpus=1,
                                dist=dist,
                                shuffle=True if stage.startswith('train') else False)
            
            self.loaders[stage] = dataloader


