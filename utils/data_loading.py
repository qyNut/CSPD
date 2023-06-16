
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import scipy.io as io
import torch
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, mask_suffix: str = '_mask'):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input files found in {images_dir}')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        
        mask_file = list(self.masks_dir.glob(name[:16] + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))
        
        img = np.abs(io.loadmat(img_file[0])['IQData'])
        dirtIdx = np.fromstring(name[-1],dtype=int, sep=' ')[0]  
        
        if dirtIdx==1:
            dirtIdxMask = 2
        else:
            dirtIdxMask = 1    
        filtIdx = np.fromstring(name[-4],dtype=int, sep=' ')[0]  
        
        mask = np.abs(io.loadmat(mask_file[0],mat_dtype=True)['VesselMap'])[:,:, filtIdx-1, dirtIdxMask-1]
        mask = np.expand_dims(mask, axis=0)
        
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).float().contiguous()
        }

