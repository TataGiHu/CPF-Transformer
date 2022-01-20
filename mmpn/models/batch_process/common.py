
import torch.nn as nn
import torch.nn.functional as F
from ..builder import BATCH_PROCESS , build_loss, CUSTOM_HOOKS
from collections import OrderedDict
from mmcv.runner.hooks.hook import HOOKS, Hook
from mmcv.parallel import DataContainer
import torch
import json,os
from torch.utils.data.dataloader import default_collate




def collate_fn(batch) :

   gpu_index = 0

   batch_per_gpu = batch.data[gpu_index]

   max_length = 0
   pad_dim = 0
   for i in range(len(batch_per_gpu)):
     max_length = max(max_length, batch_per_gpu[i]['x'].size(pad_dim))
   
   padded_samples = []
   padded_masks = []
   for sample in batch_per_gpu:
     ori_len = sample['x'].size(pad_dim)

     pad = (0,0,0,max_length-ori_len)
     padded_samples.append(F.pad(sample['x'], pad, value = 0))

     mask = torch.zeros(ori_len).bool()
     pad = torch.ones(max_length-ori_len).bool()
     padded_masks.append(torch.cat((mask, pad), 0))

   labels = []
   for i in range(len(batch_per_gpu)):
     labels.append(batch_per_gpu[i]['y'])

   classes = []
   for i in range(len(batch_per_gpu)):
      classes.append(batch_per_gpu[i]["existence"])
   metas = []
   for i in range(len(batch_per_gpu)):
     metas.append(batch_per_gpu[i]['meta'])
   
   input_datas = default_collate(padded_samples).cuda()
   masks = default_collate(padded_masks).cuda()
   labels = default_collate(labels).cuda()
   classes = default_collate(classes).cuda()
   metas = default_collate(metas)
   

   return input_datas, masks, labels, classes, metas
 
 
 
 
 
def collate_fn_output_mask(batch) :
    
   gpu_index = 0

   batch_per_gpu = batch.data[gpu_index]

   max_length = 0
   pad_dim = 0
   for i in range(len(batch_per_gpu)):
     max_length = max(max_length, batch_per_gpu[i]['x'].size(pad_dim))
   
   padded_samples = []
   padded_masks = []
   for sample in batch_per_gpu:
     ori_len = sample['x'].size(pad_dim)

     pad = (0,0,0,max_length-ori_len)
     padded_samples.append(F.pad(sample['x'], pad, value = 0))

     mask = torch.zeros(ori_len).bool()
     pad = torch.ones(max_length-ori_len).bool()
     padded_masks.append(torch.cat((mask, pad), 0))

   labels = []
   for i in range(len(batch_per_gpu)):
     labels.append(batch_per_gpu[i]['y'])

   classes = []
   for i in range(len(batch_per_gpu)):
      classes.append(batch_per_gpu[i]["existence"])
   metas = []
   for i in range(len(batch_per_gpu)):
     metas.append(batch_per_gpu[i]['meta'])
   
   output_mask = []
   for i in range(len(batch_per_gpu)):
     output_mask.append(batch_per_gpu[i]['output_mask'])
     
   furthest_point_mark = []
   for i in range(len(batch_per_gpu)):
     furthest_point_mark.append(batch_per_gpu[i]['furthest_point_mark'])
    
   input_datas = default_collate(padded_samples).cuda()
   masks = default_collate(padded_masks).cuda()
   labels = default_collate(labels).cuda()
   classes = default_collate(classes).cuda()
   metas = default_collate(metas)
   output_mask = default_collate(output_mask).cuda()
   furthest_point_mark = default_collate(furthest_point_mark).cuda()
   
   return input_datas, masks, labels, classes, metas, output_mask, furthest_point_mark