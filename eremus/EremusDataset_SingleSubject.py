import mne
import math
import json
import torch
import warnings
import numpy as np
import pandas as pd
from math import isnan
from typing import Union
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

class EremusDataset_SingleSubject_FE(Dataset):
    """Music-evoked emotions dataset."""
    
    def __init__(self, xls_file, root_dir, transform=None, label_transform=None, **args):
        """
        Args:
            xls_file (string): Path to the xls file with annotations.
            root_dir (string): Directory with all the preprocessed data.
            indices (int, optional): List of indices to use (must be a subset of xls_file indices),
                used for selecting splits.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            label_transofrm (callable, optional): Optional transform to be applied on labels.
            args: args to be passed to label_transform
        """
    
        self.data = pd.read_excel(xls_file)        
        self.root_dir = root_dir
        self.transform = transform
        self.label_transform = label_transform
        self.args = args
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # get original and array indices
        original_index = self.data.iloc[idx].original_index
        array_index = self.data.iloc[idx].array_index

        #get extracted feature
        features = np.load(self.root_dir + str(original_index) + '.npz')['arr_'+str(array_index)]
        
        gew_emotion1 = eval(self.data.iloc[idx]['gew_1'])
    
        if(self.label_transform is not None):
            emotion = self.label_transform(gew_emotion1, **self.args)
        else:
            emotion = gew_emotion1[0]

        sample = features, emotion

        if self.transform:
            sample = self.transform(sample)

        return sample

class FixedCrop(object):
    """Crop features in a sample from given start point.

    Args:
        output_size (int): Desired output size.
    """

    def __init__(self, output_size, start=0):
        assert isinstance(output_size, int)
        assert isinstance(start, int)
        if isinstance(output_size, int):
            self.output_size = output_size 
        if isinstance(start, int):
            self.start = start

    def __call__(self, sample):
        features, emotion = sample[0], sample[1]
        
        C = features.shape[0]
        F = features.shape[1]

        start = self.start
        stop = start + self.output_size
        assert (stop)<F, "start + output_size exceeds the sample length"

        features = features[:, start:stop, :]
        
        return features, emotion
    
class RandomCrop(object):
    """Crop randomly the eeg in a sample.

    Args:
        output_size (int): Desired output size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, int)
        if isinstance(output_size, int):
            self.output_size = output_size

    def __call__(self, sample):
        features, emotion = sample[0], sample[1]
        
        C = features.shape[0]
        F = features.shape[1]

        start = np.random.randint(0, F - self.output_size)
        assert start>=0, "start + output_size exceeds the sample length"
        stop = start + self.output_size

        features = features[:, start:stop, :]
        
        return features, emotion
    
class SelectBand(object):
    """Select specified band.

    Args:
        band (Union[str, int]): Desired band.
    """
    bands = {
        'delta': 0,
        'theta': 1,
        'alpha': 2,
        'beta': 3,
        'gamma': 4
    }

    def __init__(self, band):
        assert isinstance(band, str) or isinstance(band, int)
        if isinstance(band, int):
            assert band<5 and band>-1, 'Provide a band_id in range [0, 4]'
            self.band = band
        elif isinstance(band, str):
            assert band in self.bands.keys(), 'Provide a valid band name. Valid names are ' + self.bands.keys()
            self.band = self.bands[band]

    def __call__(self, sample):
        features, emotion = sample[0], sample[1]
        
        C = features.shape[0]
        F = features.shape[1]

        features = features[:, :, self.band].reshape(C, F)
        
        return features, emotion
    
class ToMatrix(object):
    """Convert features in sample to ndarray (matrix version)."""
    
    def __init__(self):
        self.location = {
            'Cz': (4, 4),
            'Fz': (2, 4),
            'Fp1': (0, 3),
            'F7': (2, 0),
            'F3': (2, 2),
            'FC1': (3, 3),
            'C3': (4, 2),
            'FC5': (3, 1),
            'FT9': (3, 0),
            'T7': (4, 0),
            'CP5': (5, 1),
            'CP1': (5, 3),
            'P3': (6, 2),
            'P7': (6, 0),
            'PO9': (7, 0),
            'O1': (8, 3),
            'Pz': (6, 4),
            'Oz': (8, 4),
            'O2': (8, 5),
            'PO10': (7, 8),
            'P8': (6, 8),
            'P4': (6, 6),
            'CP2': (5, 5),
            'CP6': (5, 7),
            'T8': (4, 8),
            'FT10': (3, 8),
            'FC6': (3, 7),
            'C4': (4, 6),
            'FC2': (3, 5),
            'F4': (2, 6),
            'F8': (2, 8),
            'Fp2': (0, 5)
        }
        
        self.ndlocation = {i: loc for i, (_, loc) in enumerate(self.location.items())}

    def __call__(self, sample):
        features, emotion = sample[0], sample[1]
        
        # Check features dimensions
        n_dim = len(features.shape)
        if n_dim==2:
            # Features is of shape (C, F)
            # Create an empty matrix (filled with 0)
            f_matrix = np.zeros((9, 9, features.shape[1]))
            # Encode array elements in matrix
            for chan, coords in self.ndlocation.items():
                f_matrix[coords][:] = features[chan]
        elif n_dim==3:
            f_matrix = np.zeros((9, 9, features.shape[1],  features.shape[2]))
            # Encode array elements in matrix
            for chan, coords in self.ndlocation.items():
                f_matrix[coords][:][:] = features[chan]

        return f_matrix, emotion 

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, tensor_type=np.float32):
        self.tensor_type=tensor_type

    def __call__(self, sample):
        features, emotion = sample[0], sample[1]
        
        features = features.astype(self.tensor_type)
        features = torch.from_numpy(features)
            
        return features, emotion
