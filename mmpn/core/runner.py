import os.path as osp

import mmcv
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import Runner, save_checkpoint

#from ..models import CompositeModules


class MRunner(Runner):

    def save_checkpoint(self,
                out_dir,
                filename_tmpl='epoch_{}.pth',
                save_optimizer=True,
                meta=None,
                create_symlink=True):
        
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        if isinstance(self.model, MMDataParallel):
            non_dist_model = self.model.module
        elif isinstance(self.model, MMDistributedDataParallel):
            non_dist_model = self.model.module
        else:
            non_dist_model = self.model
            
        #if 0:#isinstance(non_dist_model, CompositeModules):
        #    non_dist_model.save_checkpoint(osp.join(out_dir, 'ckpts'), filename, optimizer=optimizer, meta=meta)
        
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            mmcv.symlink(filename, osp.join(out_dir, 'latest.pth'))

    def train(self, data_loader, **kwargs):
        with torch.autograd.detect_anomaly():
            super().train(data_loader, **kwargs)
