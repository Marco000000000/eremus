#!/usr/bin/env python
# coding: utf-8

# In[14]:


from __future__ import print_function, division
import os
import mne
import json
import torch
import warnings
import pandas as pd
#from skimage import io, transform
import numpy as np
from math import isnan
import matplotlib.pyplot as plt
from eremus.preprocessing.spatial_filter import spatial_filter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


# In[20]:


class EremusDataset(Dataset):
    """Music-evoked emotions dataset."""

    def __init__(self, xls_file, eeg_root_dir, pruned_eeg_root_dir=None, transform=None, select_data=False):
        """
        Args:
            xls_file (string): Path to the xls file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.eeg_data = pd.read_excel(xls_file)
        self.eeg_root_dir = eeg_root_dir
        self.pruned_eeg_root_dir = pruned_eeg_root_dir
        self.transform = transform
        self.select_data = select_data

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #print(idx)
        #print(self.eeg_data.iloc[idx])
        
        subject_id = self.eeg_data.iloc[idx]['subject_id']
        
        if self.pruned_eeg_root_dir is not None:
            eeg_file = self.eeg_data.iloc[idx]['filename_pruned']
            raw_eeg = mne.io.read_raw_eeglab(self.pruned_eeg_root_dir + eeg_file, verbose = False)
        else:
            eeg_file = self.eeg_data.iloc[idx]['filename']
            raw_eeg = mne.io.read_raw_edf(self.eeg_root_dir + eeg_file, verbose = False)
        start = self.eeg_data.iloc[idx]['start_index']
        stop = self.eeg_data.iloc[idx]['end_index']
        
        if self.select_data:
            #select only data --> no metadata info
            raw_eeg = raw_eeg[:, start:stop]
        else:
            raw_eeg = raw_eeg.crop(tmin=raw_eeg.times[start], tmax=raw_eeg.times[stop],  include_tmax=False)
        
        song = self.eeg_data.iloc[idx]['spotify_track_id']
        
        gew_emotion1 = eval(self.eeg_data.iloc[idx]['gew_1'])
        gew_emotion2 = self.eeg_data.iloc[idx]['gew_2'] 
        
        use_other = bool(self.eeg_data.iloc[idx]['use_other'])
        use_eyes_closed = bool(self.eeg_data.iloc[idx]['use_eyes_closed'])
        
        if isinstance(gew_emotion2, str):
            gew_emotion2 = eval(gew_emotion2)
        elif isinstance(gew_emotion2, float):
            if isnan(gew_emotion2):
                gew_emotion2 = None
        
        sample = {'subject_id': subject_id, 'eeg': raw_eeg, 'emotion1': gew_emotion1, 'emotion2': gew_emotion2, 'song': song, 'use_other': use_other, 'use_eyes_closed': use_eyes_closed}

        if self.transform:
            sample = self.transform(sample)

        return sample


# In[20]:


class EremusDataset_V2(Dataset):
    """Music-evoked emotions dataset."""
    DATA_RAW = 0
    DATA_PRUNED = 1
    DATA_PREPROCESSED = 2

    def __init__(self, xls_file, eeg_root_dir, data_type: int = DATA_RAW, indices=None, transform=None, select_data=False, label_transform=None, **args):
        """
        Args:
            xls_file (string): Path to the xls file with samples annotations.
            eeg_root_dir (string): Directory with all the eeg data.
            data_type (int, optional): If 0, data in eeg_root_dir are raw edf files; if 1 data in eeg_root_dir are pruned set files, if 2 ata in eeg_root_dir are preprocessed set files
            indices (list(int), optional): Indices of xls_file to use in the current dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            select_data (boolean, optional): If data_type != EremusDataset_V2.DATA_PREPROCESSED and select_data = True, only channel data are selected, without metadata of Raw objects. Note that sample is a tuple of array: also time array is extracted.
            label_transform (callable, optional): Conversion function to be applied on a label.
            **args (optional): args to be passed to label_transform
        """
    
        self.eeg_data = pd.read_excel(xls_file)
        self.eeg_root_dir = eeg_root_dir
        self.data_type = data_type
        self.indices = indices
        self.transform = transform
        self.select_data = select_data
        self.label_transform = label_transform
        self.args = args

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return len(self.eeg_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #print(idx)
        #print(self.eeg_data.iloc[idx])
        
        # get partition of dataset
        # convert indices
        if self.indices is not None:
            idx = self.indices[idx]
        
        # get start and stop indices
        start = self.eeg_data.iloc[idx]['start_index']
        stop = self.eeg_data.iloc[idx]['end_index']
        
        #disable warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            #get raw eeg
            if self.data_type == self.DATA_RAW:
                eeg_file = self.eeg_data.iloc[idx]['filename']
                raw_eeg = mne.io.read_raw_edf(self.eeg_root_dir + eeg_file, verbose = False)
            elif self.data_type == self.DATA_PRUNED:
                eeg_file = self.eeg_data.iloc[idx]['filename_pruned']
                raw_eeg = mne.io.read_raw_eeglab(self.eeg_root_dir + eeg_file, verbose = False)
            elif self.data_type == self.DATA_PREPROCESSED:
                eeg_file = self.eeg_data.iloc[idx]['filename_preprocessed']
                raw_eeg = np.load(self.eeg_root_dir + eeg_file, mmap_mode='r')['arr_0'][:, start:stop]
            else:
                raise Exception('Please provide correct data type')
        
        if self.data_type != self.DATA_PREPROCESSED:
            if self.select_data:
                #select only data --> no metadata info (no 'info object')
                raw_eeg = raw_eeg[:, start:stop]
            else:
                raw_eeg = raw_eeg.crop(tmin=raw_eeg.times[start], tmax=raw_eeg.times[stop],  include_tmax=False)
        
        gew_emotion1 = eval(self.eeg_data.iloc[idx]['gew_1'])
        #gew_emotion2 = self.eeg_data.iloc[idx]['gew_2'] 
        
        #if isinstance(gew_emotion2, str):
        #    gew_emotion2 = eval(gew_emotion2)
        #elif isinstance(gew_emotion2, float):
        #    if isnan(gew_emotion2):
        #        gew_emotion2 = None
        
        if(self.label_transform is not None):
            emotion = self.label_transform(gew_emotion1, **self.args)
        else:
            emotion = gew_emotion1[0]

        sample = {'eeg': raw_eeg, 'emotion': emotion}

        if self.transform:
            sample = self.transform(sample)

        return sample


# ## Transforms

# In[25]:


class RandomCropBefore(object):
    """Crop randomly the eeg in a sample.

    Args:
        output_size (int): Desired output size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, int)
        if isinstance(output_size, int):
            self.output_size = output_size

    def __call__(self, sample):
        subject_id, eeg, emotion1, emotion2, song, use_other, use_eyes_closed = sample['subject_id'], sample['eeg'], sample['emotion1'], sample['emotion2'], sample['song'], sample['use_other'], sample['use_eyes_closed']

        d = eeg[:][0].shape[1]
        new_d = self.output_size

        start = np.random.randint(0, d - new_d)
        stop = start + self.output_size

        eeg = eeg.crop(tmin=eeg.times[start], tmax=eeg.times[stop],  include_tmax=False)
        
        return {'subject_id': subject_id, 'eeg': eeg, 'emotion1': emotion1, 'emotion2': emotion2, 'song': song, 'use_other': use_other, 'use_eyes_closed': use_eyes_closed}
    
class PickData(object):
    """Select only data channels.
    """

    def __call__(self, sample):
        subject_id, eeg, emotion1, emotion2, song, use_other, use_eyes_closed = sample['subject_id'], sample['eeg'], sample['emotion1'], sample['emotion2'], sample['song'], sample['use_other'], sample['use_eyes_closed']
        
        #pick data channels
        data_chans = eeg.ch_names[4:36]
        eeg.pick_channels(data_chans)
        #set montage
        #ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
        #eeg.set_montage(ten_twenty_montage)
        
        return {'subject_id': subject_id, 'eeg': eeg, 'emotion1': emotion1, 'emotion2': emotion2, 'song': song, 'use_other': use_other, 'use_eyes_closed': use_eyes_closed}
            
class Filter(object):
    """Filter data through a Band-Pass filter.
       Usually filters are High-Pass with a low band of 1 Hz.

    Args:
        low (int): Desired lower band.
        high (int): Desired higher band.
    """
    
    def __init__(self, low=None, high=None):
        self.low=low
        self.high=high

    def __call__(self, sample):
        subject_id, eeg, emotion1, emotion2, song, use_other, use_eyes_closed = sample['subject_id'], sample['eeg'], sample['emotion1'], sample['emotion2'], sample['song'], sample['use_other'], sample['use_eyes_closed']
        
        # band pass filter
        # For high pass filter at 1 Hz set low = 1
        eeg.load_data()
        eeg.filter(self.low, self.high)
        
        return {'subject_id': subject_id, 'eeg': eeg, 'emotion1': emotion1, 'emotion2': emotion2, 'song': song, 'use_other': use_other, 'use_eyes_closed': use_eyes_closed}
    
class RemoveBaseline(object):
    """Subtract from each channel the mean of the baseline for that channel
        
        You can applicate only if pickData was applied.
    """

    def __call__(self, sample):
        subject_id, eeg, emotion1, emotion2, song, use_other, use_eyes_closed = sample['subject_id'], sample['eeg'], sample['emotion1'], sample['emotion2'], sample['song'], sample['use_other'], sample['use_eyes_closed']
        
        #Has PickData been applied?
        assert(len(eeg.ch_names)==32)
        # get baseline mean 
        # TODO: insert correct sub_id ?
        mean = self.__getBaselineMean__(subject_id)
        # subtract baseline
        new_data = np.array([(x - mean[i]) for i, x in enumerate(eeg.get_data())])
        # create new raw eeg object
        info = eeg.info
        new_eeg = mne.io.RawArray(new_data, info, verbose=False)
        eeg = new_eeg
        
        return {'subject_id': subject_id, 'eeg': eeg, 'emotion1': emotion1, 'emotion2': emotion2, 'song': song}
    
    def __getBaselineMean__(self, subject_id, use_other=False, use_eyes_closed=False):
        baselines = pd.read_excel('baselines.xlsx')
        baselines = baselines[baselines['sub_id']==subject_id]
        if use_other:
            baselines = baselines[baselines['rec_type']=='other']
        else:
            baselines = baselines[baselines['rec_type']=='personal']
        if use_eyes_closed:
            baseline_mean = baselines.iloc[0]['c_mean']
        else:
            baseline_mean = baselines.iloc[0]['o_mean']
        return np.array(json.loads(baseline_mean))

class SetMontageAndSwap(object):
    """Set 10-20 montage.
        If necessary, correct position of sensors.
    """
    
    def __init__(self):
        self.chs = {'Cz': 0,
         'Fz': 1,
         'Fp1': 2,
         'F7': 3,
         'F3': 4,
         'FC1': 5,
         'C3': 6,
         'FC5': 7,
         'FT9': 8,
         'T7': 9,
         'CP5': 10,
         'CP1': 11,
         'P3': 12,
         'P7': 13,
         'PO9': 14,
         'O1': 15,
         'Pz': 16,
         'Oz': 17,
         'O2': 18,
         'PO10': 19,
         'P8': 20,
         'P4': 21,
         'CP2': 22,
         'CP6': 23,
         'T8': 24,
         'FT10': 25,
         'FC6': 26,
         'C4': 27,
         'FC2': 28,
         'F4': 29,
         'F8': 30,
         'Fp2': 31
        }
    
    def __call__(self, sample):
        subject_id, eeg, emotion1, emotion2, song = sample['subject_id'], sample['eeg'], sample['emotion1'], sample['emotion2'], sample['song']

        #correct channel data position
        if(subject_id<6):
            eeg = self.__swap_channels__(eeg)
        #set montage
        ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
        eeg.set_montage(ten_twenty_montage)
        
        return {'subject_id': subject_id, 'eeg': eeg, 'emotion1': emotion1, 'emotion2': emotion2, 'song': song}
    
    def __swap_channels__(self, eeg):
        # During data aquisition, bad configuration was used for a few subjects.
        # Let's correct the name of the channels
        data = eeg.get_data()
        data[[self.chs['Fz'], self.chs['FC2'], self.chs['C4']], :] = data[[self.chs['C4'], self.chs['Fz'], self.chs['FC2']], :]
        info = eeg.info
        new_eeg = mne.io.RawArray(data, info, verbose=False)
        return new_eeg
    
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
        subject_id, eeg, emotion1, emotion2, song = sample['subject_id'], sample['eeg'], sample['emotion1'], sample['emotion2'], sample['song']

        d = eeg[:][0].shape[1]
        new_d = self.output_size

        start = np.random.randint(0, d - new_d)
        stop = start + self.output_size

        eeg = eeg.crop(tmin=eeg.times[start], tmax=eeg.times[stop],  include_tmax=False)
        
        return {'subject_id': subject_id, 'eeg': eeg, 'emotion1': emotion1, 'emotion2': emotion2, 'song': song}
class SetMontage(object):
    """Set 10-20 montage."""
    
    def __call__(self, sample):
        subject_id, eeg, emotion1, emotion2, song = sample['subject_id'], sample['eeg'], sample['emotion1'], sample['emotion2'], sample['song']

        #set montage
        ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
        eeg.set_montage(ten_twenty_montage)
        
        return {'subject_id': subject_id, 'eeg': eeg, 'emotion1': emotion1, 'emotion2': emotion2, 'song': song}

class ToArray(object):
    """Convert eeg in sample to ndarray."""

    def __call__(self, sample):
        subject_id, eeg, emotion1, emotion2, song = sample['subject_id'], sample['eeg'], sample['emotion1'], sample['emotion2'], sample['song']
        # discard times, select only array data
        eeg = eeg[:][0]
        return {'subject_id': subject_id, 'eeg': eeg, 'emotion1': emotion1, 'emotion2': emotion2, 'song': song}
    
class ToMatrix(object):
    """Convert eeg in sample to ndarray (matrix version)."""
    
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

    def __call__(self, sample):
        subject_id, eeg, emotion1, emotion2, song = sample['subject_id'], sample['eeg'], sample['emotion1'], sample['emotion2'], sample['song']
        #create new matrix
        eeg_matrix = np.zeros((9, 9, eeg[:][0].shape[1]))

        for chan, coords in self.location.items():
            eeg_matrix[coords][:] = eeg[chan][0].reshape(-1)

        return {'subject_id': subject_id, 'eeg': eeg_matrix, 'emotion1': emotion1, 'emotion2': emotion2, 'song': song}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        subject_id, eeg, emotion1, emotion2, song = sample['subject_id'], sample['eeg'], sample['emotion1'], sample['emotion2'], sample['song']

        return {'subject_id': subject_id, 'eeg': torch.from_numpy(eeg), 'emotion1': emotion1, 'emotion2': emotion2, 'song': song}


# In[15]:


class FixedCrop_V2(object):
    """Crop the eeg in a sample from given start point.

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
        eeg, emotion= sample['eeg'], sample['emotion']
        
        # Check eeg instance type
        is_eeg_numpy = isinstance(eeg, np.ndarray)

        d = eeg.shape[1] if is_eeg_numpy else eeg[:][0].shape[1] 
        new_d = self.output_size
        assert (self.start + self.output_size)<d, "start + output_size exceeds the sample length"

        start = self.start
        stop = start + self.output_size

        eeg = eeg[:, start:stop] if is_eeg_numpy else eeg.crop(tmin=eeg.times[start], tmax=eeg.times[stop],  include_tmax=False)
        
        return {'eeg': eeg, 'emotion': emotion}
    
class RandomCrop_V2(object):
    """Crop randomly the eeg in a sample.

    Args:
        output_size (int): Desired output size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, int)
        if isinstance(output_size, int):
            self.output_size = output_size

    def __call__(self, sample):
        eeg, emotion= sample['eeg'], sample['emotion']

        # Check eeg instance type
        is_eeg_numpy = isinstance(eeg, np.ndarray)

        d = eeg.shape[1] if is_eeg_numpy else eeg[:][0].shape[1]
        new_d = self.output_size

        start = np.random.randint(0, d - new_d)
        stop = start + self.output_size

        eeg = eeg[:, start:stop] if is_eeg_numpy else eeg.crop(tmin=eeg.times[start], tmax=eeg.times[stop],  include_tmax=False)
        
        return {'eeg': eeg, 'emotion': emotion}

class SetMontage_V2(object):
    """Set 10-20 montage."""
    
    def __call__(self, sample):
        eeg, emotion= sample['eeg'], sample['emotion']
        #set montage
        ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
        eeg.set_montage(ten_twenty_montage)
        
        return {'eeg': eeg, 'emotion': emotion}
    
class ToArray_V2(object):
    """Convert eeg in sample to ndarray."""

    def __call__(self, sample):
        eeg, emotion= sample['eeg'], sample['emotion']
        # discard times, select only array data
        eeg = eeg[:][0]
        return {'eeg': eeg, 'emotion': emotion}
    
class ToMatrix_V2(object):
    """Convert eeg in sample to ndarray (matrix version)."""
    
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
        eeg, emotion= sample['eeg'], sample['emotion']
        
        # Check eeg instance type
        if isinstance(eeg, np.ndarray):
            
            # Eeg is a np.ndarray
            # Create an empty matrix (filled with 0)
            eeg_matrix = np.zeros((9, 9, eeg[:][0].shape[0]))
            # Encode array elements in matrix
            for chan, coords in self.ndlocation.items():
                eeg_matrix[coords][:] = eeg[chan]
        else:
            
            # Eeg is a Raw object
            # Create an empty matrix (filled with 0)
            eeg_matrix = np.zeros((9, 9, eeg[:][0].shape[1]))
            # Encode elements in matrix
            for chan, coords in self.location.items():
                eeg_matrix[coords][:] = eeg[chan][0].reshape(-1)

        return {'eeg': eeg_matrix, 'emotion': emotion}

class ToTensor_V2(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, interface='dict', eeg_tensor_type = 'float32', label_interface='tensor'):
        self.interfaces = ['dict', 'unpacked_values']
        self.eeg_tensor_types = ['float64', 'float32']
        self.label_interfaces = ['tensor', 'long']
        
        assert isinstance(interface, str)
        if isinstance(interface, str) and interface in self.interfaces:
            self.interface = interface
            
        assert isinstance(eeg_tensor_type, str)
        if isinstance(eeg_tensor_type, str) and eeg_tensor_type in self.eeg_tensor_types:
            self.eeg_tensor_type = eeg_tensor_type
        
        assert isinstance(label_interface, str)
        if isinstance(label_interface, str) and label_interface in self.label_interfaces:
            self.label_interface = label_interface

    def __call__(self, sample):
        eeg, emotion= sample['eeg'], sample['emotion']
        
        if self.eeg_tensor_type=='float32':
            eeg = eeg.astype(np.float32)
        eeg = torch.from_numpy(eeg)
            
        if self.label_interface=='tensor':
            emotion = torch.LongTensor([emotion])
        
        if self.interface=='dict':
            return {'eeg': eeg, 'emotion': emotion}
        elif self.interface=='unpacked_values':
            return eeg, emotion
        
class SpatialFilter(object):
    
    def __init__(self, N=7, weighted=True):
        self.N = N
        self.weighted = weighted

    def __call__(self, sample):
        eeg, emotion= sample['eeg'], sample['emotion']
        
        if isinstance(eeg, torch.Tensor):
            warnings.warn("NOT YET IMPLEMENTED: Spatial Filter is only supported for raw objects on numpy arrays:\n\tPlease insert this Transform before ToTensor")
        
        eeg = spatial_filter(eeg, N=self.N, weighted=self.weighted)
        
        return {'eeg': eeg, 'emotion': emotion}
        
class Normalize(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, a, b, sample_stats=False):
        self.a = a
        self.b = b
        self.sample_stats = sample_stats

    def __call__(self, sample):
        #eeg, emotion= sample['eeg'], sample['emotion']
        eeg, emotion= sample[0], sample[1]
        
        if self.sample_stats:
            minim = eeg.min()
            maxim = eeg.max()
            w = self.b - self.a
    
            eeg = w*(eeg - minim)/(maxim - minim) + self.a 
        
        else:
            minim, _ = eeg.min(1)
            maxim, _ = eeg.max(1)
            w = self.b - self.a
    
            eeg = (w*(eeg.transpose(1, 0) - minim)/(maxim - minim) + self.a).transpose(1, 0) 
        
        return eeg, emotion
    
class Standardize(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, sample_stats=False, return_dict=False):
        self.sample_stats = sample_stats
        self.return_dict = return_dict
    
    def __call__(self, sample):
        #eeg, emotion= sample['eeg'], sample['emotion']
        try:
            eeg, emotion= sample[0], sample[1]
        except KeyError:
            eeg, emotion= sample['eeg'], sample['emotion']
        
        if self.sample_stats:
            std = eeg.std()
            mean = eeg.mean()

            eeg = (eeg - mean)/std
        
        else:
            std = eeg.std(1)
            mean = eeg.mean(1)
            
            eeg = ((eeg.transpose(1, 0) - mean)/std).transpose(1, 0)
        
        if self.return_dict:
            return {'eeg': eeg, 'emotion': emotion}
        else:
            return eeg, emotion