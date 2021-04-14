import re
import sys
import numpy as np
import pandas as pd
from typing import NamedTuple
import matplotlib.pyplot as plt

if (sys.version_info.minor<8):
    def prod(iterable, start=1):
        print('custom')
        res = start
        for i in iterable:
            res = res * i
        return res
else:
    from math import prod
    
#get_ipython().run_line_magic('matplotlib', '')

"""
PART 1: GEW definition and utils
In this part we define the gew object and the main interface.
Functions to load gew emotions from rating files are provided.
The main interface of gew emotion is a tuple of type (EMOTION, INTENSITY)
"""

# emotions' dictionaries
emotions = {
    0: 'Interesse',
    1: 'Divertimento',
    2: 'Orgoglio',
    3: 'Gioia',
    4: 'Piacere',
    5: 'Contentezza',
    6: 'Amore',
    7: 'Ammirazione',
    8: 'Sollievo',
    9: 'Compassione',
    10: 'Tristezza',
    11: 'Colpa',
    12: 'Rimpianto',
    13: 'Vergogna',
    14: 'Delusione',
    15: 'Paura',
    16: 'Disgusto',
    17: 'Disprezzo',
    18: 'Odio',
    19: 'Rabbia',
    20: 'NO EMOTION FELT',
    21: 'DIFFERENT EMOTION FELT'
}

reverse_emotions = {v: k for k, v in emotions.items()}

vd_coordinates = {
    0: (0.61, 0.25),
    1: (0.67, 0.19),
    2: (0.72, 0.15),
    3: (0.68, 0.07),
    4: (0.71, 0.02),
    5: (0.77, -0.03),
    6: (0.58, -0.16),
    7: (0.66, -0.09),
    8: (0.66, -0.36),
    9: (-0.05, -0.55),
    10: (-0.68, -0.35),
    11: (-0.57, -0.27),
    12: (-0.70, -0.19),
    13: (-0.61, -0.16),
    14: (-0.77, -0.12),
    15: (-0.61, 0.07),
    16: (-0.68, 0.20),
    17: (-0.55, 0.43),
    18: (-0.45, 0.43),
    19: (-0.37, 0.47),
    20: (0, 0),
    21: None
}

# convert a rating row in to a list of gew tuples
def dumps(rating):
    # access ratings once
    fam1 = rating['gew_fam1']
    int1 = rating['gew_int1']
    fam2 = rating['gew_fam2']
    int2 = rating['gew_int2']
    # type checking
    if(type(fam1)!=str):
        print(type(fam1))
    if(type(fam2)!=str):
        print(type(fam2))
    # check different emotion for fam1
    if(re.search(emotions[21], fam1)!=None):
        different_emotion = fam1.split(' - ')
        fam1 = different_emotion[0]
        different_emotion = different_emotion[1]
        print(different_emotion)
    # convert first rating
    gew1 = (reverse_emotions[fam1], int1)
    # check the presence of second emotion
    if(fam2=='e'):
        return [gew1, None]
    else:
        # check different emotion for fam2
        if(re.search(emotions[21], fam2)!=None):
            different_emotion = fam2.split(' - ')
            fam2 = different_emotion[0]
            different_emotion = different_emotion[1]
            print(different_emotion)
        # convert second rating
        gew2 = (reverse_emotions[fam2], int2)
        return [gew1, gew2]

# convert a list of gew tuples in a dataframe 
def loads(rating):
    gew1 = rating[0]
    gew2 = rating[1]
    result = {}
    result['gew_fam1'] = emotions[gew1[0]]
    result['gew_int1'] = gew1[1]
    if(gew2!=None):
        result['gew_fam2'] = emotions[gew2[0]]
        result['gew_int2'] = gew2[1]
    else:
        result['gew_fam2'] = 'e'
        result['gew_int2'] = 0
    return pd.DataFrame(result, index=[0])


# access GEW by attributes by dotted notation
# access GEW by indices by brackets notation
# example: 
## rating = GEW("Interesse", 4)
## rating.emotion --> "Interesse"
## rating[1] --> 4
class GEW(NamedTuple):
    emotion: str
    intensity: int

# convert a GEW object in a tuple
def dump(gew):
    if gew.emotion == 'e':
        return None
    if re.search(emotions[21], gew.emotion):
        return (21, gew.intensity)
    return (reverse_emotions[gew.emotion], gew.intensity)
    
# convert a tuple in a GEW object
def load(t_gew):
    return GEW(emotions[t_gew[0]], t_gew[1])

"""
PART 2: GEW conversions
In this part we expose functions in order to:
- transform gew objects into class ids
- plot data distributions, given a transform function and a ds
- get data distribution, given a transform function
"""

# CONVERSION OF GEW EMOTIONS INTO VALENCE-AROUSAL-DOMINANCE MODEL
def vad_coordinates(gew_emotion):
    """
    Convert a gew emotion of type (EMOTION, INTENSITY) to VAD model.
    
    Params:
        - gew emotion, (int, int): gew emotion of type (EMOTION, INTENSITY)
    Returns:
        Valence-Arousal-Dominance model coordinates (V, A, D).
    """
    V, D = vd_coordinates[gew_emotion[0]]
    A = round(2*gew_emotion[1]/5 - 1, 2)
    
    return V, A, D

# CONVERSION OF GEW EMOTIONS INTO CLASS IDs

def gew_to_hldv4(gew_emotion, min_arousal=0):
    """
    Convert a gew emotion of type (EMOTION, INTENSITY) to High/Low Dominance/Valence.
    Neutral class (NO EMOTION FELT) is not considered as a class.
    Other classes (DIFFERENT EMOTIONS) are not considered as a class.
    
    Dominance is considered Low in range [5, 15[, High in range [15, 5[
    Valence is considered Low in range [10, 20[, High in range [0, 10[
    Divide VD graph into 4 different areas:
    - HDHV (High Dominance, High Valence)
    - LDHV (Low Dominance, High Valence)
    - LDLV (Low Dominance, Low Valence)
    - HDLV (High Dominance, Low Valence)
    """
    if isinstance(gew_emotion, int):
        emotion = gew_emotion
    elif isinstance(gew_emotion, tuple):
        emotion = gew_emotion[0]
    else:
        raise Exception('Emotion not valid: please provide emotion in gew format (Emotion, Intensity) or with a Emotion iteger id')
    # discard DIFFERENT EMOTION and NO EMOTION
    if emotion==21 or emotion==20:
        raise Exception('Emotion field of gew_emotion out of range: emotion id should be less than 20')
    # HDHV
    elif emotion<5:
        return 0
    # LDHV
    elif emotion>=5 and emotion<10:
        return 1
    # LDLV
    elif emotion>=10 and emotion<15:
        return 2
    # HDLV
    elif emotion>=15 and emotion<20:
        return 3
    else:
        raise Exception('Emotion not valid')
    
def gew_to_hldv5(gew_emotion, min_arousal=3):
    """
    Convert a gew emotion of type (EMOTION, INTENSITY) to High/Low Dominance/Valence.
    Neutral class (NO EMOTION FELT) is considered as a separate class.
    Emotions with low arousal are considered neutral emotions.
    The minimal intensity level to consider emotion non-neutral is specified by min_arousal parameter.
    Other classes (DIFFERENT EMOTIONS) are not considered as a class.
    
    Dominance is considered Low in range [5, 15[, High in range [15, 5[
    Valence is considered Low in range [10, 20[, High in range [0, 10[
    Divide VD graph into 5 different areas:
    - NEUTRAL (Low Emotion Intensity)
    - HDHV (High Dominance, High Valence)
    - LDHV (Low Dominance, High Valence)
    - LDLV (Low Dominance, Low Valence)
    - HDLV (High Dominance, Low Valence)
    """
    if isinstance(gew_emotion, tuple):
        emotion = gew_emotion[0]
        arousal = gew_emotion[1]
    else:
        raise Exception('Emotion not valid: please provide emotion in gew format (Emotion, Intensity)')
    # discard DIFFERENT EMOTION 
    if emotion==21:
        raise Exception('Emotion field of gew_emotion out of range: emotion id should be less than 20')
    # Neutral emotions
    elif emotion==20 or arousal<min_arousal:
        return 0
    # HDHV
    elif emotion<5:
        return 1
    # LDHV
    elif emotion>=5 and emotion<10:
        return 2
    # LDLV
    elif emotion>=10 and emotion<15:
        return 3
    # HDLV
    elif emotion>=15 and emotion<20:
        return 4
    else:
        raise Exception('Emotion not valid')
    
def gew_to_8(gew_emotion, use_neutral=False, use_different=False, min_arousal=2):
    """
    Convert a gew emotion of type (EMOTION, INTENSITY) to custom 8 (or 9, or 10, depending on paramters) 
    classes used in the first phase of the experiment.
    
    Params:
    - gew_emotion:  GEW emotion of type (EMOTION, INTENSITY) or Emotion Integer ID
    - use_neutral: bool, if True use NO EMOTION FELT as neutral class.
        Emotions with intensity less than min_arousal are also considered as NEUTRAL emotions.
        if False, NO EMOTION FELT emotions are discarded as incorrect.
    - use_different: bool, if True use DIFFERENT EMOTION FELT as a different class or (TODO) map into other classes.
        if False DIFFERENT EMOTION FELT emotions are discarded as incorrect.
    - min_arousal: int, the minimal intensity level to consider emotion non-neutral. It is not used if gew_emotion is not a tuple
    """
    
    if isinstance(gew_emotion, int):
        emotion = gew_emotion
        # DIFFERENT EMOTION FELT
        if emotion == 21 and use_different:
            return 9
        # NO EMOTION FELT
        if emotion == 20:
            if use_neutral:
                return 8
            else: raise Exception('Emotion field of gew_emotion out of range: emotion id should be less than 20')
    elif isinstance(gew_emotion, tuple):
        emotion = gew_emotion[0]
        arousal = gew_emotion[1]
        # DIFFERENT EMOTION FELT
        if emotion == 21 and use_different:
            return 9
        # NO EMOTION FELT
        if emotion == 20:
            if use_neutral:
                return 8
            else: raise Exception('Emotion field of gew_emotion out of range: emotion id should be less than 20')
        if use_neutral and arousal<min_arousal:
            return 8
    else:
        raise Exception('Emotion not valid: please provide emotion in gew format (Emotion, Intensity) or with a Emotion iteger id')

    if emotion in range(0, 3):
        return 0
    elif emotion in range(3, 5):
        return 1
    elif emotion in range(5, 8):
        return 2
    elif emotion in range(8, 10):
        return 3
    elif emotion == 10:
        return 4
    elif emotion in range(11, 15):
        return 5
    elif emotion == 15:
        return 6
    elif emotion in range(16, 20):
        return 7
    else:
        raise Exception('Emotion not valid')
        
def gew_to_6a(gew_emotion, min_arousal=0):
    """
    Convert a gew emotion of type (EMOTION, INTENSITY) to 6 intensity classes.
    
    Params:
    - gew_emotion: GEW emotion of type (EMOTION, INTENSITY) or Emotion Integer ID
    - min_arousal: Not used
    """
    if isinstance(gew_emotion, tuple):
        emotion = gew_emotion[0]
        arousal = gew_emotion[1]
        # NO EMOTION FELT
        if emotion == 20 or emotion == 21:
            raise Exception('Emotion field of gew_emotion out of range: emotion id should be less than 20')
        else:
            return arousal
    else:
        raise Exception('Emotion not valid: please provide emotion in gew format (Emotion, Intensity)')
        
def gew_to_5a(gew_emotion, min_arousal=0):
    """
    Convert a gew emotion of type (EMOTION, INTENSITY) to 5 intensity classes.
    
    Params:
    - gew_emotion: GEW emotion of type (EMOTION, INTENSITY) or Emotion Integer ID
    - min_arousal: Not used
    """
    if isinstance(gew_emotion, tuple):
        emotion = gew_emotion[0]
        arousal = gew_emotion[1]
        # NO EMOTION FELT
        if emotion == 20 or emotion == 21:
            raise Exception('Emotion field of gew_emotion out of range: emotion id should be less than 20')
        elif arousal==0:
            return 0
        else:
            return arousal - 1
    else:
        raise Exception('Emotion not valid: please provide emotion in gew format (Emotion, Intensity)')
        
def gew_to_emotion(gew_emotion, min_arousal=0):
    """
    Convert a gew emotion of type (EMOTION, INTENSITY) to 20 emotion classes.
    Neutral class (NO EMOTION FELT) is considered as a separate class.
    Emotions with low arousal are considered neutral emotions.
    The minimal intensity level to consider emotion non-neutral is specified by min_arousal parameter.
    Other classes (DIFFERENT EMOTIONS) are considered as a separated class.
    
    Params:
    - gew_emotion: GEW emotion of type (EMOTION, INTENSITY) or Emotion Integer ID
    - min_arousal: int, minimal arousal to consider it as non-neutral emotion
    """
    if isinstance(gew_emotion, tuple):
        emotion = gew_emotion[0]
        arousal = gew_emotion[1]
        if emotion==21:
            return 21
        elif arousal<min_arousal:
            return 20
        else:
            return emotion
    else:
        raise Exception('Emotion not valid: please provide emotion in gew format (Emotion, Intensity)')

# GET DATA DISTRIBUTIONS

def get_data_distribution(gew_labels, num_classes, transform_function, **args):
    """
    Get data distribution of data, given original gew labels and a transform function.
    
    Params:
    - gew_labels: dataset labels. Please provide a list of tuple in GEW format (EMOTION_ID, INTENSITY)
    - num_classes: number of classes otteined with the transform_function
    - transform_function: function used to convert gew emotion into class id
    - ** args: named args to be passed to transform_function
    
    Returns:
    - a list of size num_classes, whose i-th entry is the number of occurencies in the dataset that has been mapped to i-th class
    """
    
    new_labels = []
    for gew_emotion in gew_labels:
        new_labels.append(transform_function(gew_emotion, **args))
    return [new_labels.count(i) for i in range(num_classes)]

# PLOT DATA DISTRIBUTIONS

def plot_data_distribution(gew_labels, transform_function, normalize=True, verbose=False, **args):
    """
    Plot data distributions of data, given original gew labels and a transform function.
    If transform_function output depends on arousal parameter, data are plotted with all possible min_arousal values
    in n different bar plots, one for each min_arousal value.
    
    Params:
    - gew_labels: dataset labels. Please provide a list of tuple in GEW format (EMOTION_ID, INTENSITY)
    - verbose: if True, print some additional infos (used for debug)
    - normalize: if True, normalize output counts into [0, 1] range
    - transform_function: function used to convert gew emotion into class id
    - ** args: named args to be passed to transform_function
    """
    
    # configure subplots, fig_size, ticks and plt params
    if transform_function==gew_to_hldv4:
        grid = (1, 1)
        classes = ['HDHV','LDHV','LDLV','HDLV']
    elif transform_function==gew_to_hldv5:
        grid = (2, 3)
        classes = ['N', 'HDHV','LDHV','LDLV','HDLV']
    elif transform_function==gew_to_8:
        grid = (2, 3)
        classes = ['I','G','Cn','S','T','Cl','P','Dg','N','D']
    elif transform_function==gew_to_5a:
        grid = (1, 1)
        classes = [x for x in range(1, 6)]
    elif transform_function==gew_to_6a:
        grid = (1, 1)
        classes = [x for x in range(6)]
    elif transform_function==gew_to_emotion:
        grid = (2, 3)
        classes = [x for x in range(22)]
    else: 
        raise Exception('transform_function not valid')
    num_classes = len(classes)
    fig = plt.figure(figsize=(20,10))
    
    for min_arousal in range(0, prod(grid)):
        if min_arousal>6:
            break
        new_labels = []
        for gew_emotion in gew_labels:
            #new_labels.append(gew_to_8(gew_emotion, min_arousal=min_arousal, use_different=True, use_neutral=True))
            #new_labels.append(gew_to_hldv5(gew_emotion, min_arousal=min_arousal))
            new_labels.append(transform_function(gew_emotion, min_arousal=min_arousal, **args))
        new_labels_count = [new_labels.count(i) for i in range(num_classes)]
        if verbose:
            print('DATA DISTRIBUTION. Function: ' + str(transform_function) + ' Min Arousal: ' + str(min_arousal))
            for class_id in range(num_classes):
                  print('Class ' + str(class_id) + ': ' + str(new_labels_count[class_id])) 
        
        fig.add_subplot(grid[0], grid[1], min_arousal + 1)
        x = np.linspace(0,num_classes-1,num_classes,endpoint=True)
        width = 1  # the width of the bars
        plt.xticks(x, tuple(classes))
        if normalize:
            plt.bar(x, [new_label_count/len(new_labels) for new_label_count in new_labels_count])
        else: 
            plt.bar(x, new_labels_count)
        plt.title(min_arousal)

def plot_data_distribution_grouped(gew_labels, transform_function, verbose = False, **args):
    """
    Plot data distributions of data, given original gew labels and a transform function.
    If transform_function output depends on arousal parameter, data are plotted with all possible min_arousal values
    in a single grouped bar plot.
    
    Params:
    - gew_labels: dataset labels. Please provide a list of tuple in GEW format (EMOTION_ID, INTENSITY)
    - verbose: if True, print some additional infos (used for debug)
    - transform_function: function used to convert gew emotion into class id
    - ** args: named args to be passed to transform_function
    """
    
    # configure subplots, fig_size, ticks and plt params
    if transform_function==gew_to_hldv4:
        grid = (1, 1)
        classes = ['HDHV','LDHV','LDLV','HDLV']
    elif transform_function==gew_to_hldv5:
        grid = (2, 3)
        classes = ['N', 'HDHV','LDHV','LDLV','HDLV']
    elif transform_function==gew_to_8:
        grid = (2, 3)
        classes = ['I','G','Cn','S','T','Cl','P','Dg','N','D']
    elif transform_function==gew_to_5a:
        grid = (1, 1)
        classes = [x for x in range(1, 6)]
    elif transform_function==gew_to_6a:
        grid = (1, 1)
        classes = [x for x in range(6)]
    elif transform_function==gew_to_emotion:
        grid = (2, 3)
        classes = [x for x in range(22)]
    else: 
        raise Exception('transform_function not valid')
    num_classes = len(classes)
    
    for min_arousal in range(6):
        if min_arousal>6:
            break
        new_labels = []
        for gew_emotion in gew_labels:
            #new_labels.append(gew_to_8(gew_emotion, min_arousal=min_arousal, use_different=True, use_neutral=True))
            #new_labels.append(gew_to_hldv5(gew_emotion, min_arousal=min_arousal))
            new_labels.append(transform_function(gew_emotion, min_arousal=min_arousal, **args))
        new_labels_count = [new_labels.count(i) for i in range(num_classes)]
        if verbose:
            print('DATA DISTRIBUTION. Function: ' + str(transform_function) + ' Min Arousal: ' + str(min_arousal))
            for class_id in range(num_classes):
                  print('Class ' + str(class_id) + ': ' + str(new_labels_count[class_id])) 
        
        x = np.linspace(0,num_classes-1,num_classes,endpoint=True)
        width = 1/6 - 0.01 # the width of the bars
        graph = plt.bar(x - (2.5 - min_arousal)*width, new_labels_count, width, label=min_arousal)
        plt.xticks(x, tuple(classes))

if __name__ == "__main__":
    
    from torchvision import transforms, utils
    from EremusDataset import EremusDataset_V2, RandomCrop_V2, SetMontage_V2, ToArray_V2, ToMatrix_V2, ToTensor_V2
    from EremusDataset import EremusDataset, RandomCrop, SetMontage, ToArray, ToMatrix, ToTensor
    
    # TESTING - PART1
    rating = GEW("Amore", 3)
    t = dump(rating)
    print(t)
    g = load(t)
    print(g)

    s = 'DIFFERENT EMOTION FELT - Ansia'
    ss = s.split(' - ')
    ss
    
    re.search(emotions[21], s)!=None
    re.search("DIFFERENT EMOTION FELT", s)==None
    
    if(re.search(emotions[21], s)!=None):
        different_emotion = s.split(' - ')[1]
        print(different_emotion)

    type('re')==str
    
    # TESTING - PART2
    
    # define transforms
    set_montage = SetMontage()
    to_array = ToArray()
    to_tensor = ToTensor()

    # compose transforms
    composed = transforms.Compose([set_montage, to_array, to_tensor])

    # load dataset
    emus = EremusDataset(xls_file='eremus_test.xlsx',
                    eeg_root_dir='eeglab_raws\\', 
                    data_type=EremusDataset_V2.DATA_PRUNED,
                    transform=composed)
    
    # load labels
    labels = [emus[index]['emotion1'] for index in range(len(emus))]

    def delete_emotions(labels, emotion_to_delete):    
        tagged_emotions = [i for i, x in enumerate(labels) if x[0] == emotion_to_delete]
        tagged_emotions = sorted(tagged_emotions, reverse = True)
        for i in tagged_emotions:
            del labels[i]
        return labels

    # purge labels from different emotions (nd) and neutral emotions (nn)
    nd_labels = delete_emotions(labels.copy(), 21)
    nn_labels = delete_emotions(nd_labels.copy(), 20)
    
    # get distributions - usage examples
    custom_model_stats = []
    for min_arousal in range (0, 6):
        dd = get_data_distribution(labels, 10, gew_to_8, min_arousal = min_arousal, use_different=True, use_neutral=True)
        custom_model_stats.append(dd)
    custom_model_stats = np.array(custom_model_stats)
    print(custom_model_stats.std(1))
    print(str(gew_to_8), "Best min_arousal_val: " + str(np.argmin(custom_model_stats.std(1))))
    
    dd = get_data_distribution(nn_labels, 5, gew_to_5a, min_arousal = min_arousal)
    dd = np.array(dd)
    print(str(gew_to_5a), "Standard Deviation: " + str(dd.std()))
    
    # plot distributions - all possible usages (use these)
    plot_data_distribution_grouped(labels, gew_to_8, use_different=True, use_neutral=True)
    #plot_data_distribution_grouped(nd_labels, gew_to_hldv5)
    #plot_data_distribution_grouped(nd_labels, gew_to_emotion)
    
    plot_data_distribution(labels, gew_to_8, use_different=True, use_neutral=True)

    # delete DIFFERENT EMOTION (21)
    plot_data_distribution(nd_labels, gew_to_hldv5)
    plot_data_distribution(nd_labels, gew_to_emotion)

    # delete NO EMOTION (20)
    plot_data_distribution(nn_labels, gew_to_hldv4)
    plot_data_distribution(nn_labels, gew_to_5a)
    plot_data_distribution(nn_labels, gew_to_6a) 