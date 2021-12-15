# experiments
exp_type = 'BaseExp'

work_dir = './work/test1/' # log and checkpoint dir

# dataset and dataloader
# dataset_type = 'BrakeDataset'
data_provider = dict(
  type='DataLoader',
  samples_per_gpu=20,
  workers_per_gpu=0,
  
  stages = dict(
            train=dict(
              type='DdmapDatasetDC', 
              data_path='/share12T5/refline/train_data/PLAFB9216_event_HNP_wm_sharp_turning_filter_20211117-203333_0.txt',
            ),
            val=dict(
              type='DdmapDatasetDC', 
              data_path='/share12T5/refline/train_data/PLAFB9216_event_HNP_wm_sharp_turning_filter_20211117-203333_0.txt',
            ),
 
           )
)

# traing and val workflow
workflow = [('train', 1)]  # now only support train mode

# model 
model = dict(type='DdmapModel')

# batch process
# batch process
batch_process = dict(
    type="DdmapBatchProcessDC",
    loss_reg=dict(
      type='SmoothL1Loss',
    ),
    loss_process_type = "normal",  # "normal", "weight", "points"
    weight=[1,100,1000]
)

hooks = [dict(type="DdmapTestCoeffHook")]

lr = 1e-3     # learning rate
optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=1e-4)


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
total_epochs = 10
# using StepLrUpdaterHook: https://github.com/open-mmlab/mmcv/blob/13888df2aa22a8a8c604a1d1e6ac1e4be12f2798/mmcv/runner/hooks/lr_updater.py#L167



# logs and checkpoint
log_level = 'INFO'


resume_from = None
load_from = None 


# DDP related
dist_params = dict(backend='nccl') 
find_unused_parameters = True

