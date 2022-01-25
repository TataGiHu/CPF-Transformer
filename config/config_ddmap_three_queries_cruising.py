# experiments
from mmpn.models.losses import bce_loss
import numpy as np

exp_type = 'BaseExp'
parallels = 4
work_dir = './work/3_crusing_100000_6layer/' # log and checkpoint dir

# dataset and dataloader
# dataset_type = 'BrakeDataset'
data_provider = dict(
  type='DataLoader',
  samples_per_gpu=128,
  workers_per_gpu=2,
  
  stages = dict(
            train=dict(
              type='DdmapDescreteDatasetWithRoadEdgeAndDashedAttributeNew', 
              root_path='/data/sida/0124_6000_train_data/',
              data_file='/data/sida/train_file/0125/crusing_file_list_100000.txt',
            ),
            val=dict(
              type='DdmapDescreteDatasetWithRoadEdgeAndDashedAttributeNew', 
              data_path='/data/sida/0124_100/',
            ),
 
           )
)

# traing and val workflow
workflow = [('train', 1)]  # now only support train mode

# model 
model = dict(type='DdmapDescreteModelThreeQueries')

# batch process
# batch process
batch_process = dict(
    type="DdmapDescreteBatchProcessDCThreeQueries",
    loss_reg=dict(
      type='L2Loss',
    ),
    bce_loss_reg = dict(
      type="BCELoss"
    ),
    # Current centerline has the largest weight
    weight= [[1 for i in range(25)] , [1 for i in range(25)] , [1 for i in range(25)]]
)

hooks = [dict(type="DdmapDescreteTestHookThreeQueries")]

lr_scale = parallels

lr = 1e-3 * lr_scale    # learning rate
optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=1e-4)
#optimizer = dict(type="AdamW", lr=lr, weight_decay=1e-2)


#lr_config = dict(policy='step', step=[1, 3], gamma=0.1) 
lr_config = dict(policy='Step',
                 warmup='linear',
                 gamma=0.5,
                 warmup_iters=int(100 * np.power(lr_scale, 1 / 3)),
                 warmup_ratio=1.0 / 3 / lr_scale,
                 step=[int(e) for e in [35, 45, 50, 55]])

optimizer_config = dict(grad_clip=dict(max_norm=3, norm_type=2)) 
checkpoint_config = dict(interval=1)  # save checkpoint at every epoch
log_config = dict(
    interval=1,  # log at every 10 iterations
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])


# training related
total_epochs = 60
# using StepLrUpdaterHook: https://github.com/open-mmlab/mmcv/blob/13888df2aa22a8a8c604a1d1e6ac1e4be12f2798/mmcv/runner/hooks/lr_updater.py#L167



# logs and checkpoint
log_level = 'INFO'

#resume_from = None
resume_from = '/home/sida/code/mhw_train/work/3_crusing_100000_6layer/epoch_15.pth'

load_from = None
# load_from = "/workspace/mhw_train/work/test1/epoch_1000.pth"


# DDP related
dist_params = dict(backend='nccl') 
find_unused_parameters = True

