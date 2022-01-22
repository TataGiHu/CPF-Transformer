# experiments
from mmpn.models.losses import bce_loss


exp_type = 'BaseExp'

work_dir = './work/test1/' # log and checkpoint dir

# dataset and dataloader
# dataset_type = 'BrakeDataset'
data_provider = dict(
  type='DataLoader',
  samples_per_gpu=256,
  workers_per_gpu=26,
  
  stages = dict(
            train=dict(
              type='DdmapDescreteDatasetWithRoadEdgeAndDashedAttributeNew', 
              data_path='/home/mingquan/splited_data',
            ),
            val=dict(
              type='DdmapDescreteDatasetWithRoadEdgeAndDashedAttributeNew', 
              data_path='/home/mingquan/splited_data',
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

lr = 1e-2    # learning rate
# optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=1e-4)
optimizer = dict(type="AdamW", lr=lr,weight_decay=1e-2)


lr_config = dict(policy='step', step=[1, 3], gamma=0.1) 
optimizer_config = dict(grad_clip=dict(max_norm=3, norm_type=2)) 
checkpoint_config = dict(interval=1)  # save checkpoint at every epoch
log_config = dict(
    interval=5,  # log at every 10 iterations
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])


# training related
total_epochs = 2000 
# using StepLrUpdaterHook: https://github.com/open-mmlab/mmcv/blob/13888df2aa22a8a8c604a1d1e6ac1e4be12f2798/mmcv/runner/hooks/lr_updater.py#L167



# logs and checkpoint
log_level = 'INFO'

#resume_from = None
resume_from = "/home/mingquan/mhw_train/work/test1/latest.pth"

load_from = None
# load_from = "/workspace/mhw_train/work/test1/epoch_1000.pth"


# DDP related
dist_params = dict(backend='nccl') 
find_unused_parameters = True

