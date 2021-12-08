
# model 
brake_model = dict(type='BrakeModel', in_features=10, n_class=1, hidden=1024)
model = brake_model

# batch process
batch_process = dict(
    type="BrakeBatchProcess",
    loss_cls=dict(type='BCELoss')
)

# dataset and dataloader
# dataset_type = 'BrakeDataset'
brake_dataset_provider = dict(
     type='DataLoader',
     samples_per_gpu=256,
     workers_per_gpu=0,
    
     stages=
         dict(
          train=dict(
                type='BrakeDataset', 
                text_path='data_example/train_path.txt',
                ),
         )
    
)
data_provider = brake_dataset_provider


# experiments
exp_type = 'BaseExp'


# logs and checkpoint
work_dir = './work' # log and checkpoint dir
log_level = 'INFO'
log_config = dict(
    interval=10,  # log at every 10 iterations
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])

checkpoint_config = dict(interval=1)  # save checkpoint at every epoch
load_from = None 
resume_from = None

# training related
total_epochs = 10
lr = 1e-3     # learning rate
# using StepLrUpdaterHook: https://github.com/open-mmlab/mmcv/blob/13888df2aa22a8a8c604a1d1e6ac1e4be12f2798/mmcv/runner/hooks/lr_updater.py#L167
lr_config = dict(policy='step', step=[1, 3], gamma=0.1) 

optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=1e-4)
optimizer_config = dict(grad_clip=dict(max_norm=3, norm_type=2)) 

# traing and val workflow
workflow = [('train', 1)]  # now only support train mode

# DDP related
dist_params = dict(backend='nccl') 
find_unused_parameters = True

