# EremusDataset - ExtractedFeatures

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

class EremusDataset_EF(Dataset):
    """Music-evoked emotions dataset."""
    
    def __init__(self, xls_file, root_dir, indices=None, transform=None, label_transform=None, **args):
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
        self.indices = indices
        self.transform = transform
        self.label_transform = label_transform
        self.args = args
        
    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get partition of dataset
        # convert indices
        if self.indices is not None:
            idx = self.indices[idx]

        #get extracted feature
        features = np.load(self.root_dir + str(idx) + '.npz')['arr_0']
        
        gew_emotion1 = eval(self.data.iloc[idx]['gew_1'])
    
        if(self.label_transform is not None):
            emotion = self.label_transform(gew_emotion1, **self.args)
        else:
            emotion = gew_emotion1[0]

        sample = features, emotion

        if self.transform:
            sample = self.transform(sample)

        return sample

class EremusDataset_EFWS(Dataset):
    """Music-evoked emotions dataset."""

    def __init__(self, xls_file, root_dir, indices=None, transform=None, window_size = 1280, step_size = 128*3, label_transform=None, **args):
        """
        Args:
            xls_file (string): Path to the xls file with annotations.
            root_dir (string): Directory with all the preprocessed data.
            indices (int, optional): List of indices to use. It Must be a subset of 
                range(len(self.index_dict)). Used for selecting splits.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            window_size (int): size of sliding window on data
            step_size(int): size of step for the sliding window
            label_transofrm (callable, optional): Optional transform to be applied on labels.
            args: args to be passed to label_transform
        """
    
        self.data = pd.read_excel(xls_file)        
        self.root_dir = root_dir
        self.indices = indices
        self.transform = transform
        self.label_transform = label_transform
        self.args = args
        
        self.window_size = window_size
        self.step_size = step_size
        
        # create index map
        self.index_dict = {}
        index_count = 0
        for i, row in self.data.iterrows():
            # compute sample length
            sample_len = row.end_index - row.start_index
            # calculate n_frames = number of frames per sample
            n_frames = math.floor((sample_len-self.window_size)/self.step_size) + 1
            # map index count to (i, n_frame)
            for n_frame in range(n_frames):
                self.index_dict[index_count] = (i, n_frame)
                index_count = index_count + 1
        
    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return len(self.index_dict)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get partition of dataset
        # convert indices
        if self.indices is not None:
            idx = self.indices[idx]
        
        # get sample id and offset
        sample_id, offset = self.index_dict[idx]
        file = str(sample_id) + '.npz'
        features = np.load(self.root_dir + file)['arr_' + str(offset)] # use offset in the correct way

        gew_emotion1 = eval(self.data.iloc[sample_id]['gew_1'])
    
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


if __name__ == "__main__":

    
    import random
    from gew import gew_to_hldv5

    #=============
    # Testing EF
    #=============


    # list all indices
    emus_dataset = pd.read_excel('eremus_test.xlsx')
    indices = list(range(len(emus_dataset)))

    # calculate different and neutral emotions' indices
    def select_emotion(labels, emotion_to_delete):    
        return [i for i, x in enumerate(labels) if x == emotion_to_delete]

    labels = [eval(row.gew_1)[0] for _, row in emus_dataset.iterrows()]
    different_emotions = select_emotion(labels, 21)
    neutral_emotions = select_emotion(labels, 20)

    # exclude different emotions
    indices = [i for i in indices if i not in different_emotions]

    # split into train, validation and test
    random.shuffle(indices)

    train_fraction = 0.7
    validation_fraction = 0.15
    test_fraction = 0.15

    num_test = int(len(indices)*test_fraction)
    num_validation = int(len(indices)*validation_fraction)
    num_train = len(indices) - num_test - num_validation

    train_idx = indices[0:num_train]
    validation_idx = indices[num_train:num_train+num_validation]
    test_idx = indices[num_train+num_validation:]

    # print infos
    print("Train samples: " + str(num_train) + "\tValidation samples: " + str(num_validation) + "\tTest samples: " + str(num_test))

    # load train dataset
    emus_train = EremusDataset_EF(xls_file='eremus_test.xlsx', 
                                  root_dir='preprocessed\\de\\', 
                                  indices=train_idx, 
                                  label_transform=gew_to_hldv5)

    #-------------
    # Transforms
    #-------------

    # Define transforms
    crop = RandomCrop(200)
    gamma_band = SelectBand('gamma')
    to_matrix = ToMatrix()
    to_tensor = ToTensor()

    composed = transforms.Compose([crop, gamma_band, to_matrix, to_tensor])

    # load test dataset
    emus_test = EremusDataset_EF(xls_file='eremus_test.xlsx', 
                                 root_dir='preprocessed\\de\\', 
                                 indices=test_idx, 
                                 transform = composed,
                                 label_transform=gew_to_hldv5)

    emus_test[0][0].shape


    #===============
    # Testing EFWS
    #===============

    import random
    from gew import gew_to_hldv5
    WINDOW_SIZE = 10
    STEP_SIZE = 3
    SFREQ = 128
    DIR = str(WINDOW_SIZE) + '_' + str(STEP_SIZE) + '\\'

    # list all indices
    emus_dataset = EremusDataset_EFWS(xls_file='eremus_test.xlsx', 
                                      root_dir='preprocessed\\de_ws\\'+DIR, 
                                      window_size=WINDOW_SIZE*SFREQ, 
                                      step_size=STEP_SIZE*SFREQ)
    indices = list(range(len(emus_dataset)))

    # split into train, validation and test
    random.shuffle(indices)

    train_fraction = 0.7
    validation_fraction = 0.15
    test_fraction = 0.15

    num_test = int(len(indices)*test_fraction)
    num_validation = int(len(indices)*validation_fraction)
    num_train = len(indices) - num_test - num_validation

    train_idx = indices[0:num_train]
    validation_idx = indices[num_train:num_train+num_validation]
    test_idx = indices[num_train+num_validation:]

    # print infos
    print("Train samples: " + str(num_train) + "\tValidation samples: " + str(num_validation) + "\tTest samples: " + str(num_test))

    # load train dataset
    emus_train = EremusDataset_EFWS(xls_file='eremus_test.xlsx', 
                                      root_dir='preprocessed\\de_ws\\'+DIR, 
                                      window_size=WINDOW_SIZE*SFREQ, 
                                      step_size=STEP_SIZE*SFREQ, 
                                      indices=train_idx)

    #-------------
    # Transforms
    #-------------

    # Define transforms
    crop = FixedCrop(32)
    gamma_band = SelectBand('theta')
    to_matrix = ToMatrix()
    to_tensor = ToTensor()

    composed = transforms.Compose([crop, gamma_band, to_matrix, to_tensor])

    # load test dataset
    emus_val = EremusDataset_EFWS(xls_file='eremus_test.xlsx', 
                                  root_dir='preprocessed\\de_ws\\'+DIR, 
                                  window_size=WINDOW_SIZE*SFREQ, 
                                  step_size=STEP_SIZE*SFREQ, 
                                  indices=train_idx,
                                  transform=composed)
    # Expected 9x9x32
    emus_val[0][0].shape


    # Define transforms
    crop = RandomCrop(32)
    gamma_band = SelectBand('beta')
    to_tensor = ToTensor()

    composed = transforms.Compose([crop, gamma_band, to_tensor])

    # load test dataset
    emus_val = EremusDataset_EFWS(xls_file='eremus_test.xlsx', 
                                  root_dir='preprocessed\\de_ws\\'+DIR, 
                                  window_size=WINDOW_SIZE*SFREQ, 
                                  step_size=STEP_SIZE*SFREQ, 
                                  indices=train_idx,
                                  transform=composed)
    # Expected 32x32
    emus_val[0][0].shape

