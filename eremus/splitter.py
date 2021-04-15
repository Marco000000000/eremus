from .gew import gew_to_hldv4
import pandas as pd
import random
import math
import .gew
import re
import os

pd.options.mode.chained_assignment = None  # default='warn'

def split(indices, train_frac=0.7, validation_frac=0.15, test_frac=0.15):
    
    num_test = int(len(indices)*test_frac)
    num_validation = int(len(indices)*validation_frac)
    num_train = len(indices) - num_test - num_validation
    
    train_idx = indices[0:num_train]
    validation_idx = indices[num_train:num_train+num_validation]
    test_idx = indices[num_train+num_validation:]
    
    return train_idx, validation_idx, test_idx

# get original indices
def get_original_indices(df):
    return list(set(list(df.original_index)))

# get indices for specified emotion id
def select_emotion(labels, emotion_to_delete):    
    return [i for i, x in enumerate(labels) if x == emotion_to_delete]

def augmented_split(path_to_augmented_eremus, path_to_eremus, train_frac=0.7, validation_frac=0.15, test_frac=0.15):
    # open augmented eremus
    augmented_emus = pd.read_excel(path_to_augmented_eremus)
    # fetch indices
    original_indices = get_original_indices(augmented_emus)

    # open eremus
    emus_dataset = pd.read_excel(path_to_eremus)
    # fetch all labels
    labels = [eval(row.gew_1)[0] for _, row in emus_dataset.iterrows()]
    # exclude different and neutral emotions
    different_emotions = select_emotion(labels, 21)
    neutral_emotions = select_emotion(labels, 20)
    original_indices = [i for i in original_indices if i not in different_emotions]
    original_indices = [i for i in original_indices if i not in neutral_emotions]

    # shuffle and split original indices
    random.shuffle(original_indices)
    train_idx, validation_idx, test_idx = split(original_indices, train_frac=train_frac, validation_frac=validation_frac, test_frac=test_frac)

    # map to real indices
    train_idx = [item for sublist in [augmented_emus[augmented_emus.original_index == index].index.tolist() for index in train_idx] for item in sublist]
    validation_idx = [item for sublist in [augmented_emus[augmented_emus.original_index == index].index.tolist() for index in validation_idx] for item in sublist]
    test_idx = [item for sublist in [augmented_emus[augmented_emus.original_index == index].index.tolist() for index in test_idx] for item in sublist]
    
    # shuffle new indices
    random.shuffle(train_idx)
    random.shuffle(validation_idx)
    random.shuffle(test_idx)

    return train_idx, validation_idx, test_idx

def simple_split(path_to_eremus, train_frac=0.7, validation_frac=0.15, test_frac=0.15):
    
    # list all indices
    emus_dataset = pd.read_excel(path_to_eremus)
    indices = list(range(len(emus_dataset)))

    # calculate different and neutral emotions' indices
    labels = [eval(row.gew_1)[0] for _, row in emus_dataset.iterrows()]
    different_emotions = select_emotion(labels, 21)
    neutral_emotions = select_emotion(labels, 20)

    # exclude different emotions
    indices = [i for i in indices if i not in different_emotions]
    # exclude neutral emotions
    indices = [i for i in indices if i not in neutral_emotions]

    # split into train, validation and test
    random.shuffle(indices)
    return split(indices, train_frac=train_frac, validation_frac=validation_frac, test_frac=test_frac)

def get_split_boundaries(a, b, validation_frac, test_frac):
    # compute segment lenght
    l = b-a
    # compute sub-lenght
    l_test = int(l*test_frac)
    l_validation = int(l*validation_frac)
    l_train = l - l_test - l_validation
    # compute boundaries
    train_start, train_stop = a, a + l_train
    validation_start, validation_stop = train_stop, train_stop + l_validation
    test_start, test_stop = validation_stop, validation_stop + l_test
    # create dictionary
    boundaries = {
        'train': (train_start, train_stop),
        'validation': (validation_start, validation_stop),
        'test': (test_start, test_stop),
    }
    return boundaries

def temporal_split(xls_file, w_dir, train_frac=0.7, validation_frac=0.15, test_frac=0.15, window_size=10, step_size=3, s_freq=128):
    """
    Given the dataset, the split fractions and the sliding window parameters,
    compute train, test and validation splits, thus producing three xls files.

        Parameters
        ----------
        xls_file : str
        w_dir : str
        train_frac : float
        validation_frac : float
        test_frac : float
        window_size : int
        step_size : int
        s_freq : int

        Returns
        -------
        output : tuple, (n_train, n_val, n_test)
    """
    # read original dataset
    samples = pd.read_excel(xls_file)
    # delete additional column
    del samples['Unnamed: 0']

    # get a sample
    sample = samples.iloc[0]

    # create empty dataframes
    datasets = {
        'train': pd.DataFrame(),
        'validation': pd.DataFrame(),
        'test': pd.DataFrame(),
    }
    
    # initialize dataframes to fit samples dtypes
    for _, df in datasets.items():
        df['original_index'] = pd.Series([], dtype=int)
        for k, v in sample.items():
            if k in ['subject_id', 'start_index', 'end_index']:
                df[k] = pd.Series([], dtype=int)
            else:
                df[k] = pd.Series([], dtype=str)
            
    # iterate samples
    num_samples = len(samples)
    for i, sample in samples.iterrows():
        print("{0:.0%}".format(i/num_samples), end='\r')
        if(eval(sample.gew_1)[0]==20 or eval(sample.gew_1)[0]==21):
            continue
        # split sample into train, validation and test
        boundaries = get_split_boundaries(sample.start_index, sample.end_index, validation_frac, test_frac)
        for split in datasets:
            start, end = boundaries[split]
            # divide sample into sub_sumples of window_size
            for start_index in range(start, end-int(window_size*s_freq), int(step_size*s_freq)):
                # compute end_index
                end_index = start_index + s_freq*window_size
                # update start and stop values
                sample.loc['start_index'] = start_index
                sample.loc['end_index'] = end_index
                # add original index
                sample.loc['original_index'] = i
                # add sample to dataframe
                datasets[split] = datasets[split].append(sample)
    
    # reset indices and write into files
    for split, df in datasets.items():
        df.reset_index(drop=True, inplace=True)
        df.to_excel(w_dir + split + '.xlsx')
               
    return tuple([len(df) for _, df in datasets.items()])

def get_rnd_split_boundaries(a, b, validation_frac, test_frac, verbose=False):
    # compute segment lenght
    l = b-a
    # compute sub-lenght
    l_test = int(l*test_frac)
    l_validation = int(l*validation_frac)
    l_train = l - l_test - l_validation
    # compute random pivot for test set
    pivot = random.randint(a, b-l_test)
    # compute test boundaries
    test_start, test_stop = pivot, pivot + l_test
    # exclude some points for validation to avoid overlap with test
    not_valid = [i for i in list(range(a, b)) if i>test_start-l_validation and i<test_stop]
    # compute random pivot for validation set
    while pivot in not_valid:
        pivot = random.randint(a, b-l_validation)
    # compute validation boundaries
    validation_start, validation_stop = pivot, pivot + l_validation
    # test and set sets are leavig out three segments, that will be used for train
    train_start1, train_stop1 = a, min(test_start, validation_start)
    train_start2, train_stop2 = min(test_stop, validation_stop), max(test_start, validation_start)
    train_start3, train_stop3 = max(test_stop, validation_stop), b
    # print check
    if verbose:
        print('Train: ', train_start1, train_stop1, "|", train_start2, train_stop2, "|", train_start3, train_stop3)
        print('Test: ', test_start, test_stop)
        print('Validation: ', validation_start, validation_stop)
    
    # create dictionary
    boundaries = {
        'train1': (train_start1, train_stop1),
        'train2': (train_start2, train_stop2),
        'train3': (train_start3, train_stop3),
        'validation': (validation_start, validation_stop),
        'test': (test_start, test_stop),
    }
    return boundaries

def print_sets_distribution(boundaries):
    # char lenght
    ch_len = 80
    # extract boundaries
    a, _ = boundaries['train1']
    _, b = boundaries['train3']
    test_start, test_stop = boundaries['test']
    validation_start, validation_stop = boundaries['validation']
    # compute lenght
    l = b-a
    # normalize boundaries to 0-100 range
    tsn, ten = ch_len*(test_start - a)//l, ch_len*(test_stop - a)//l
    vsn, ven = ch_len*(validation_start - a)//l, ch_len*(validation_stop - a)//l
    # determine test and validation order
    if tsn < vsn:
        a1, b1, s1 = tsn, ten, 'T'
        a2, b2, s2 = vsn, ven, 'V'
    else:
        a1, b1, s1 = vsn, ven, 'V'
        a2, b2, s2 = tsn, ten, 'T'
    # print distribution
    print('='*a1, end='')
    print(s1*(b1-a1), end='')
    print('='*(a2-b1), end='')
    print((b2-a2)*s2, end='')
    print('='*(ch_len-b2))
    
def rnd_temporal_split(xls_file, w_dir, train_frac=0.7, validation_frac=0.15, test_frac=0.15, window_size=10, step_size=3, s_freq=128):
    """
    Given the dataset, the split fractions and the sliding window parameters,
    compute train, test and validation splits, thus producing three xls files.

        Parameters
        ----------
        xls_file : str
        w_dir : str
        train_frac : float
        validation_frac : float
        test_frac : float
        window_size : int
        step_size : int
        s_freq : int

        Returns
        -------
        output : tuple, (n_train, n_val, n_test)
    """
    # read original dataset
    samples = pd.read_excel(xls_file)
    # delete additional column
    del samples['Unnamed: 0']

    # get a sample
    sample = samples.iloc[0]

    # create empty dataframes
    datasets = {
        'train': pd.DataFrame(),
        'validation': pd.DataFrame(),
        'test': pd.DataFrame(),
    }
    
    # initialize dataframes to fit samples dtypes
    for _, df in datasets.items():
        df['original_index'] = pd.Series([], dtype=int)
        for k, v in sample.items():
            if k in ['subject_id', 'start_index', 'end_index']:
                df[k] = pd.Series([], dtype=int)
            else:
                df[k] = pd.Series([], dtype=str)
            
    # iterate samples
    num_samples = len(samples)
    for i, sample in samples.iterrows():
        print("{0:.0%}".format(i/num_samples), end='\r')
        if(eval(sample.gew_1)[0]==20 or eval(sample.gew_1)[0]==21):
            continue
        # split sample into train, validation and test
        boundaries = get_rnd_split_boundaries(sample.start_index, sample.end_index, validation_frac, test_frac)
        for split in boundaries:
            start, end = boundaries[split]
            # divide sample into sub_sumples of window_size
            for start_index in range(start, end-int(window_size*s_freq), int(step_size*s_freq)):
                # compute end_index
                end_index = start_index + s_freq*window_size
                # update start and stop values
                sample.loc['start_index'] = start_index
                sample.loc['end_index'] = end_index
                # add original index
                sample.loc['original_index'] = i
                # add sample to dataframe
                if re.search("train", split):
                    datasets["train"] = datasets["train"].append(sample)
                else:
                    datasets[split] = datasets[split].append(sample)
    
    # reset indices and write into files
    for split, df in datasets.items():
        df.reset_index(drop=True, inplace=True)
        df.to_excel(w_dir + split + '.xlsx')
               
    return tuple([len(df) for _, df in datasets.items()])

def de_ws_sample_temporal_split(sample, train_frac=0.7, validation_frac=0.15, test_frac=0.15, window_size=5.0, step_size=1.0, s_freq=128, verbose=False):
    """
    Get train, test and validation array indices for a single sample: 
    chosen arrays for each split are granted to come from not inter-split overlapped windows.
    
    Parameters
    ----------
    sample: pandas.core.series.Series
        a dataset row
    train_frac : float
        train_fraction
    validation_frac : float
        validation_fraction
    test_frac : float
        test_fraction
    window_size : float
        window size used in dataset augmentation (window sliding) in s
    step_size : float
        window size used in dataset augmentation (window sliding) in s
    s_freq : int
        sampling frequency

    Returns
    -------
    dict {split: list (int)}
        a dict with lists of array indices, to use for train, validation, test sets.
    """
    
    # get sample lenght
    sample_len = sample.end_index - sample.start_index
    # compute number of frames for the sample
    n_frames = math.floor((sample_len-window_size)/step_size) + 1
    # compute boundaries beetween splits
    num_train, num_val = int(sample_len*train_frac), int(sample_len*validation_frac)
    #train_stop, val_stop = sample.start_index + num_train, sample.start_index + num_train + num_val
    # compute max start index for a window
    max_window_start = int(sample_len-(window_size*s_freq))
    # compute real step size
    sliding_step = int(step_size*s_freq) 
    # init empty lists
    train_idx = []
    val_idx = []
    test_idx = []
    # window sliding
    for window_id, window_start in enumerate(range(0, max_window_start, sliding_step)):
        #print(window_id, window_start)
        # compute window_end
        window_end = window_start + int(window_size*s_freq)
        # check and assign to split
        if window_end < num_train:
            train_idx = train_idx + [window_id]
        elif window_start >= num_train and window_end < (num_train + num_val):
            val_idx = val_idx + [window_id]
        elif window_start >= (num_train + num_val):
            test_idx = test_idx + [window_id]
        elif verbose:
            print('window id %d is in overlap: excluded' % window_id)
    # create dictionary
    array_indices = {
        'train': train_idx,
        'validation': val_idx,
        'test': test_idx,
    }
    return array_indices

def de_ws_temporal_split(xls_file, w_dir, train_frac=0.7, validation_frac=0.15, test_frac=0.15, window_size=10, step_size=3, s_freq=128):

    """
    Given the dataset, the split fractions and the sliding window parameters,
    compute train, test and validation splits, thus producing three xls files.

        Parameters
        ----------
        xls_file : str
        w_dir : str
        train_frac : float
        validation_frac : float
        test_frac : float
        window_size : int
        step_size : int
        s_freq : int

        Returns
        -------
        output : tuple, (n_train, n_val, n_test)
    """
    # read original dataset
    samples = pd.read_excel(xls_file)
    # delete additional columns
    del samples['Unnamed: 0']

    # get a sample
    sample = samples.iloc[0]

    # create empty dataframes
    datasets = {
        'train': pd.DataFrame(),
        'validation': pd.DataFrame(),
        'test': pd.DataFrame(),
    }
    
    # initialize dataframes to fit samples dtypes
    for _, df in datasets.items():
        df['original_index'] = pd.Series([], dtype=int)
        df['array_index'] = pd.Series([], dtype=int)
        for k, v in sample.items():
            if k == 'subject_id':
                df[k] = pd.Series([], dtype=int)
            else:
                df[k] = pd.Series([], dtype=str)
            
    # iterate samples
    num_samples = len(samples)
    for i, sample in samples.iterrows():
        print("{0:.0%}".format(i/num_samples), end='\r')
        if(eval(sample.gew_1)[0]==20 or eval(sample.gew_1)[0]==21):
            continue
        # split sample into train, validation and test
        array_indices = de_ws_sample_temporal_split(sample, 
                                                    train_frac=train_frac, 
                                                    validation_frac=validation_frac, 
                                                    test_frac=test_frac, 
                                                    window_size=window_size, 
                                                    step_size=step_size, 
                                                    s_freq=s_freq)
        for split in datasets:
            split_indices = array_indices[split]
            # iterate indices
            for array_index in split_indices:
                # add array_index
                sample.loc['array_index'] = array_index
                # add original index
                #sample.loc['original_index'] = sample.original_index
                # add sample to dataframe
                datasets[split] = datasets[split].append(sample)
    
    os.makedirs(w_dir, exist_ok=True)
    # reset indices and write into files
    for split, df in datasets.items():
        del df['start_index']
        del df['end_index']
        df.reset_index(drop=True, inplace=True)
        df.to_excel(w_dir + split + '.xlsx')
        
               
    return tuple([len(df) for _, df in datasets.items()])

def is_balanced(xls_file, t=0.2):
    samples = pd.read_excel(xls_file)
    del samples['Unnamed: 0']
    # extract labels
    labels = [eval(row.gew_1) for _, row in samples.iterrows()]
    # get class distribution
    dd = get_data_distribution(labels, 4, gew_to_hldv4)
    if 1 in dd:
        print('Warning: only 1 element per class')
        return False
    pp = [round(x/sum(dd), 2) for x in dd]
    return True if (max(pp) - min(pp))<t else False

def de_ws_sample_rnd_temporal_split(sample, validation_frac=0.15, test_frac=0.15, window_size=5.0, step_size=1.0, s_freq=128, verbose=False):
    """
    Get train, test and validation array indices for a single sample: 
    chosen arrays for each split are granted to come from not inter-split overlapped windows.
    
    Parameters
    ----------
    sample: pandas.core.series.Series
        a dataset row
    validation_frac : float
        validation_fraction
    test_frac : float
        test_fraction
    window_size : float
        window size used in dataset augmentation (window sliding) in s
    step_size : float
        window size used in dataset augmentation (window sliding) in s
    s_freq : int
        sampling frequency

    Returns
    -------
    dict {split: list (int)}
        a dict with lists of array indices, to use for train, validation, test sets.
    """
    
    # get sample lenght
    sample_len = sample.end_index - sample.start_index
    # compute number of frames for the sample
    n_frames = math.floor((sample_len-window_size)/step_size) + 1
    # compute boundaries beetween splits
    boundaries = get_rnd_split_boundaries(sample.start_index, sample.end_index, validation_frac, test_frac, verbose)
    # compute max start index for a window
    max_window_start = sample.start_index + int(sample_len-(window_size*s_freq))
    # compute real step size
    sliding_step = int(step_size*s_freq) 
    # init empty lists
    train_idx = []
    val_idx = []
    test_idx = []
    # window sliding
    for window_id, window_start in enumerate(range(sample.start_index, max_window_start, sliding_step)):
        #print(window_id, window_start)
        # compute window_end
        window_end = window_start + int(window_size*s_freq)
        # check and assign to split
        if (window_end < boundaries['train1'][1]) or (window_start >= boundaries['train2'][0] and window_end < boundaries['train2'][1]) or (window_start >= boundaries['train3'][0]):
            train_idx = train_idx + [window_id]
        elif window_start >= boundaries['validation'][0] and window_end < boundaries['validation'][1]:
            val_idx = val_idx + [window_id]
        elif window_start >= boundaries['test'][0] and window_end < boundaries['test'][1]:
            test_idx = test_idx + [window_id]
        elif verbose:
            print('window id %d is in overlap: excluded' % window_id)
    # create dictionary
    array_indices = {
        'train': train_idx,
        'validation': val_idx,
        'test': test_idx,
    }
    return array_indices

def de_ws_rnd_temporal_split(xls_file, w_dir, train_frac=0.7, validation_frac=0.15, test_frac=0.15, window_size=10, step_size=3, s_freq=128, verbose=False):

    """
    Given the dataset, the split fractions and the sliding window parameters,
    compute train, test and validation splits, thus producing three xls files.

        Parameters
        ----------
        xls_file : str
        w_dir : str
        train_frac : float
        validation_frac : float
        test_frac : float
        window_size : int
        step_size : int
        s_freq : int

        Returns
        -------
        output : tuple, (n_train, n_val, n_test)
    """
    # read original dataset
    samples = pd.read_excel(xls_file)
    # delete additional columns
    del samples['Unnamed: 0']

    # get a sample
    sample = samples.iloc[0]

    # create empty dataframes
    datasets = {
        'train': pd.DataFrame(),
        'validation': pd.DataFrame(),
        'test': pd.DataFrame(),
    }
    
    # initialize dataframes to fit samples dtypes
    for _, df in datasets.items():
        df['original_index'] = pd.Series([], dtype=int)
        df['array_index'] = pd.Series([], dtype=int)
        for k, v in sample.items():
            if k == 'subject_id':
                df[k] = pd.Series([], dtype=int)
            else:
                df[k] = pd.Series([], dtype=str)
            
    # iterate samples
    num_samples = len(samples)
    for i, sample in samples.iterrows():
        print("{0:.0%}".format(i/num_samples), end='\r')
        if(eval(sample.gew_1)[0]==20 or eval(sample.gew_1)[0]==21):
            continue
        # split sample into train, validation and test
        array_indices = de_ws_sample_rnd_temporal_split(sample,
                                                    validation_frac=validation_frac, 
                                                    test_frac=test_frac, 
                                                    window_size=window_size, 
                                                    step_size=step_size, 
                                                    s_freq=s_freq,
                                                    verbose = verbose)
        for split in datasets:
            split_indices = array_indices[split]
            # iterate indices
            for array_index in split_indices:
                # add array_index
                sample.loc['array_index'] = array_index
                # add original index
                #sample.loc['original_index'] = sample.original_index
                # add sample to dataframe
                datasets[split] = datasets[split].append(sample)
    
    os.makedirs(w_dir, exist_ok=True)
    # reset indices and write into files
    for split, df in datasets.items():
        del df['start_index']
        del df['end_index']
        df.reset_index(drop=True, inplace=True)
        df.to_excel(w_dir + split + '.xlsx')
        
               
    return tuple([len(df) for _, df in datasets.items()])

def de_ws_simple_split(xls_file, w_dir, window_size=10, step_size=3, s_freq=128):

    """
    Given the dataset and the sliding window parameters,
    compute train, test and validation splits, thus producing three xls files.

        Parameters
        ----------
        xls_file : str
        w_dir : str
        window_size : int
        step_size : int
        s_freq : int

        Returns
        -------
        output : tuple, (n_train, n_val, n_test)
    """
    # read original dataset
    samples = pd.read_excel(xls_file)
    # delete additional columns
    del samples['Unnamed: 0']

    # get a sample
    sample = samples.iloc[0]

    # create empty dataframes
    datasets = {
        'train': pd.DataFrame(),
        'validation': pd.DataFrame(),
        'test': pd.DataFrame(),
    }
    
    # initialize dataframes to fit samples dtypes
    for _, df in datasets.items():
        df['original_index'] = pd.Series([], dtype=int)
        df['array_index'] = pd.Series([], dtype=int)
        for k, v in sample.items():
            if k == 'subject_id':
                df[k] = pd.Series([], dtype=int)
            else:
                df[k] = pd.Series([], dtype=str)
    
    indices_n_labels = []
    for i, sample in samples.iterrows():
        indices_n_labels = indices_n_labels + [(i, gew_to_hldv4(eval(sample.gew_1)))]

    random.shuffle(indices_n_labels)
    #print([test[0] for test in indices_n_labels])

    num_classes = 4
    val_test_idx = []
    for c in range(num_classes):
        sample_index, sample_label = next(si for si in indices_n_labels if si[1] == c)
        val_test_idx = val_test_idx + [sample_index]
    train_idx = [j[0] for j in indices_n_labels if j[0] not in val_test_idx]
    #print(sorted(train_idx), sorted(val_test_idx))

    # iterate validation and test samples
    for i in val_test_idx:
        sample = samples.iloc[i]
        # split sample into validation and test
        array_indices = de_ws_sample_temporal_split(sample, 
                                                    train_frac=0, 
                                                    validation_frac=0.5, 
                                                    test_frac=0.5, 
                                                    window_size=window_size, 
                                                    step_size=step_size, 
                                                    s_freq=s_freq)
        for split in ['validation', 'test']:
            split_indices = array_indices[split]
            # iterate indices
            for array_index in split_indices:
                # add array_index
                sample.loc['array_index'] = array_index
                # add original index
                #sample.loc['original_index'] = sample.original_index
                # add sample to dataframe
                datasets[split] = datasets[split].append(sample)
    
    # iterate train samples
    for i in train_idx:
        sample = samples.iloc[i]
        # get sample lenght
        sample_len = sample.end_index - sample.start_index
        # compute number of frames for the sample
        n_frames = math.floor((sample_len-(window_size*s_freq))/(step_size*s_freq)) + 1
        # compute indices
        array_indices = list(range(n_frames))
        # iterate indices
        for array_index in array_indices:
            # add array_index
            sample.loc['array_index'] = array_index
            # add original index
            #sample.loc['original_index'] = sample.original_index
            # add sample to dataframe
            datasets['train'] = datasets['train'].append(sample)
    
    os.makedirs(w_dir, exist_ok=True)
    # reset indices and write into files
    for split, df in datasets.items():
        del df['start_index']
        del df['end_index']
        df.reset_index(drop=True, inplace=True)
        df.to_excel(w_dir + split + '.xlsx')
        
               
    return tuple([len(df) for _, df in datasets.items()])
