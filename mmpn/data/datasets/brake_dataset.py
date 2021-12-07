import os
import pickle
import json
import numpy as np
from torch.utils.data import Dataset
from ..builder import DATASETS

@DATASETS.register_module()
class BrakeDataset(Dataset):
    def __init__(self, text_path, view='F50', wid=5):
        self.txt_path = text_path
        with open(self.txt_path, 'r') as f:
                self.pkl_path_list = f.readlines()
        
        self.data = self.__load_data(view=view, wid=wid)

    def __load_data(self, view, wid):
        data = None
        for path in self.pkl_path_list:
            with open(path.split('\n')[0], 'rb') as f: 
                seq = pickle.load(f)             
                tmp = self.gen_samples(self.gen_seq(seq, view=view), wid=wid)
                if data is None:
                    data = tmp
                else:
                    data = np.concatenate([data, tmp])
            
        return data.astype("float32")
    
    def gen_seq(self, data, view):
        seqs = {}
        for frame in data:
            for obj in frame.get('obj'):
                if obj.get('tag') != view:
                    continue

                track_id = obj.get('track_id')
                if track_id not in seqs.keys():
                    seqs[track_id] = []

                tmp = obj.get('brake')
                seqs[track_id].append([tmp.get('score'), tmp.get('value')])
        return seqs
    
    def gen_samples(self, seqs, wid):
        samples = None
        for key in seqs.keys():
            seq = seqs.get(key)

            if len(seq) < wid:
                continue
            for i in range(len(seq) - wid + 1):
                tmp = np.array(seq[i:i+wid]).reshape(1,wid,2)
                if samples is None:
                    samples = tmp
                else:
                    samples = np.concatenate([samples, tmp], axis=0)
        return samples

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        x = x.flatten()
        y = np.array([0.0], dtype='float32') # just for interface placeholder        
        return x, y
    
    
    
    