import mne
import math
import json
import torch
import warnings
import numpy as np
import pandas as pd
from math import isnan
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

class EremusDataset_WS(Dataset):
    """Music-evoked emotions dataset."""
    DATA_RAW = 0
    DATA_PRUNED = 1
    DATA_PREPROCESSED = 2

    def __init__(self, xls_file, eeg_root_dir, data_type: int = DATA_RAW, window_size = 1280, step_size = 128*3, indices=None, transform=None, select_data=False, label_transform=None, **args):
        """
        Args:
            xls_file (string): Path to the xls file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
    
        self.eeg_data = pd.read_excel(xls_file)
        self.eeg_root_dir = eeg_root_dir
        self.data_type = data_type
        self.indices = indices
        self.transform = transform
        self.select_data = select_data
        self.label_transform = label_transform
        self.args = args
        
        self.window_size = window_size
        self.step_size = step_size
        self.count = 0
        
    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return len(self.eeg_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get partition of dataset
        # convert indices
        if self.indices is not None:
            idx = self.indices[idx]
        
        # get start and stop samples
        start = self.eeg_data.iloc[idx]['start_index']
        stop = self.eeg_data.iloc[idx]['end_index']
        sample_len = stop - start
        
        # calculate n_frames = number of frames per sample
        n_frames = math.floor((sample_len-self.window_size)/self.step_size) + 1   
        
        # window selection
        window_start = start + math.floor(self.count/len(self))%n_frames * self.step_size
        window_stop = window_start + self.window_size
        #print(self.count, window_start - start, window_stop - start)
        
        # update count
        self.count = self.count + 1
        
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
                raw_eeg = np.load(self.eeg_root_dir + eeg_file, mmap_mode='r')['arr_0'][:, window_start:window_stop]
            else:
                raise Exception('Please provide correct data type')
        
        if self.data_type != self.DATA_PREPROCESSED:
            if self.select_data:
                #select only data --> no metadata info (no 'info object')
                raw_eeg = raw_eeg[:, window_start:window_stop]
            else:
                raw_eeg = raw_eeg.crop(tmin=raw_eeg.times[window_start], tmax=raw_eeg.times[window_stop],  include_tmax=False)
        
        gew_emotion1 = eval(self.eeg_data.iloc[idx]['gew_1'])
    
        if(self.label_transform is not None):
            emotion = self.label_transform(gew_emotion1, **self.args)
        else:
            emotion = gew_emotion1[0]

        sample = {'eeg': raw_eeg, 'emotion': emotion}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class EremusDataset_WS_MultipleCounters(Dataset):
    """Music-evoked emotions dataset."""
    DATA_RAW = 0
    DATA_PRUNED = 1
    DATA_PREPROCESSED = 2

    def __init__(self, xls_file, eeg_root_dir, data_type: int = DATA_RAW, window_size = 1280, step_size = 128*3, indices=None, transform=None, select_data=False, label_transform=None, **args):
        """
        Args:
            xls_file (string): Path to the xls file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
    
        self.eeg_data = pd.read_excel(xls_file)
        self.eeg_root_dir = eeg_root_dir
        self.data_type = data_type
        self.indices = indices
        self.transform = transform
        self.select_data = select_data
        self.label_transform = label_transform
        self.args = args
        
        self.window_size = window_size
        self.step_size = step_size
        self.counters = None
        if self.indices is not None:
            self.counters = list(np.zeros(len(self.indices), dtype=np.int))
        
    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return len(self.eeg_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get partition of dataset
        # convert indices
        if self.indices is not None:
            real_idx = self.indices[idx]
            count = self.counters[idx]
        else: 
            real_idx = idx
            count = 0
        
        # get start and stop samples
        start = self.eeg_data.iloc[real_idx]['start_index']
        stop = self.eeg_data.iloc[real_idx]['end_index']
        sample_len = stop - start
        
        # calculate n_frames = number of frames per sample
        n_frames = math.floor((sample_len-self.window_size)/self.step_size) + 1   
        
        # window selection
        window_start = start + math.floor(count)%n_frames * self.step_size
        window_stop = window_start + self.window_size
        #print(count, window_start - start, window_stop - start)
        
        # update count
        if self.counters is not None:
            self.counters[idx] = count + 1
            
        #disable warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            #get raw eeg
            if self.data_type == self.DATA_RAW:
                eeg_file = self.eeg_data.iloc[real_idx]['filename']
                raw_eeg = mne.io.read_raw_edf(self.eeg_root_dir + eeg_file, verbose = False)
            elif self.data_type == self.DATA_PRUNED:
                eeg_file = self.eeg_data.iloc[real_idx]['filename_pruned']
                raw_eeg = mne.io.read_raw_eeglab(self.eeg_root_dir + eeg_file, verbose = False)
            elif self.data_type == self.DATA_PREPROCESSED:
                eeg_file = self.eeg_data.iloc[real_idx]['filename_preprocessed']
                raw_eeg = np.load(self.eeg_root_dir + eeg_file, mmap_mode='r')['arr_0'][:, window_start:window_stop]
            else:
                raise Exception('Please provide correct data type')
        
        if self.data_type != self.DATA_PREPROCESSED:
            if self.select_data:
                #select only data --> no metadata info (no 'info object')
                raw_eeg = raw_eeg[:, window_start:window_stop]
            else:
                raw_eeg = raw_eeg.crop(tmin=raw_eeg.times[window_start], tmax=raw_eeg.times[window_stop],  include_tmax=False)
        
        gew_emotion1 = eval(self.eeg_data.iloc[real_idx]['gew_1'])
    
        if(self.label_transform is not None):
            emotion = self.label_transform(gew_emotion1, **self.args)
        else:
            emotion = gew_emotion1[0]

        sample = {'eeg': raw_eeg, 'emotion': emotion}

        if self.transform:
            sample = self.transform(sample)

        return sample

class FixedCrop(object):
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
        eeg, emotion= sample['eeg'], sample['emotion']

        # Check eeg instance type
        is_eeg_numpy = isinstance(eeg, np.ndarray)

        d = eeg.shape[1] if is_eeg_numpy else eeg[:][0].shape[1]
        new_d = self.output_size

        start = np.random.randint(0, d - new_d)
        stop = start + self.output_size

        eeg = eeg[:, start:stop] if is_eeg_numpy else eeg.crop(tmin=eeg.times[start], tmax=eeg.times[stop],  include_tmax=False)
        
        return {'eeg': eeg, 'emotion': emotion}
    
class ToArray(object):
    """Convert eeg in sample to ndarray."""

    def __call__(self, sample):
        eeg, emotion= sample['eeg'], sample['emotion']
        # discard times, select only array data
        eeg = eeg[:][0]
        return {'eeg': eeg, 'emotion': emotion}
    
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

class ToTensor(object):
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

if __name__ == "__main__":
    
    import splitter
    from gew import gew_to_hldv4

    # split dataset into train, validation and test
    train_idx, validation_idx, test_idx = splitter.simple_split('eremus_test.xlsx', 
                                                                train_frac=0.7, 
                                                                validation_frac=0.15, 
                                                                test_frac=0.15)

    # define transforms
    crop = RandomCrop(512)
    to_matrix = ToMatrix()
    to_tensor = ToTensor(interface='unpacked_values', label_interface='long')

    # compose transforms
    composed = transforms.Compose([crop, to_matrix, to_tensor])

    emus_train = EremusDataset_WS_MultipleCounters(xls_file='eremus_test.xlsx', 
                                  eeg_root_dir='preprocessed\\overall_z_score_base\\', 
                                  data_type=EremusDataset_WS_MultipleCounters.DATA_PREPROCESSED,  
                                  window_size=1280, 
                                  step_size=128*3,
                                  indices = train_idx, 
                                  transform=composed, 
                                  label_transform=gew_to_hldv4)

    emus_validation = EremusDataset_WS_MultipleCounters(xls_file='eremus_test.xlsx', 
                                  eeg_root_dir='preprocessed\\overall_z_score_base\\',
                                  data_type=EremusDataset_WS_MultipleCounters.DATA_PREPROCESSED,  
                                  window_size=1280, 
                                  step_size=128*3,
                                  indices = validation_idx, 
                                  transform=composed, 
                                  label_transform=gew_to_hldv4)

    emus_test = EremusDataset_WS_MultipleCounters(xls_file='eremus_test.xlsx', 
                                  eeg_root_dir='preprocessed\\overall_z_score_base\\', 
                                  data_type=EremusDataset_WS_MultipleCounters.DATA_PREPROCESSED,  
                                  window_size=1280, 
                                  step_size=128*3,
                                  indices = test_idx, 
                                  transform=composed, 
                                  label_transform=gew_to_hldv4)

    print("Train dataset: " + str(len(emus_train)) + "\tValidation dataset: " + str(len(emus_validation)) + "\tTest dataset: " + str(len(emus_test)))

