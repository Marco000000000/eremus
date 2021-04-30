#!/usr/bin/env python
# coding: utf-8

# # Preprocessing dei dati

# Prima di iniziare ad allenare una rete, i dati devono essere puliti.
# Gli step fondamentali che vanno fatti sono i seguenti.
# Per ogni sessione dobbiamo:
# - estrarre i canali dei dati (ignorare la qualità del contatto al momento)
# - filtrare i dati con un passa-alto (verificare anche che l'effetto sia equivalente a quello di rimozione di una deadline)
# - settare il montaggio e correggere eventuali errori di montaggio dei sensori
# - effettuare una ICA ed eliminare le componenti di disturbo
# - salvare i dati così ottenuti come nuovi file edf o set
# 
# Utilizzare in seguito i file così ottenuti come base per un nuovo dataset.
# Utilizzeremo l'eremus test, cambiando il nome dei file con i dati già preprocessati
# Si creerà una classe Eremus Dataset. I dati dovranno passare dai seguenti passaggi:
# - random crop del segnale (1280 campioni) - non salire a più di un migliaio di campioni!
# - conversione in array o matriced
# - conversione in tensore

# ### Interfaccia

# Per lavorare meglio utilizzerò una classe Session come interfaccia per il lavoro da svolgere

# In[1]:


import os
import re
import mne
import pandas as pd
import matplotlib.pyplot as plt

# In[2]:


class Session(object):
    
    chs = {'Cz': 0,
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

    def __init__(self, subject_id, edf_file, s_type):
        """
        Args:
            subject_id (int): subject identifier for that session
            edf_file (string): Path to the eeg data.
            s_type (string): session type, could be personal or other
        """
        self.subject_id = subject_id
        self.eeg = mne.io.read_raw_edf(edf_file)
        self.s_type = s_type
        
    def pick_data(self):        
        #pick data channels
        chs = self.eeg.ch_names[4:36]
        self.eeg.pick_channels(chs)
        #set montage
        #ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
        #eeg.set_montage(ten_twenty_montage)
    
    def set_montage(self):
        if(self.subject_id<6):
            data = self.eeg.get_data()
            data[[self.chs['Fz'], self.chs['FC2'], self.chs['C4']], :] = data[[self.chs['C4'], self.chs['Fz'], self.chs['FC2']], :]
            info = self.eeg.info
            new_eeg = mne.io.RawArray(data, info, verbose=False)
            self.eeg = new_eeg
        #set montage
        ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
        self.eeg.set_montage(ten_twenty_montage)

    def hp_filter(self):
        #load data
        self.eeg.load_data()
        # delete continous components
        fig = self.eeg.filter(0.5, None)
        
    def plot(self):
        self.eeg.plot()

    def get_eeg(self):
        return self.eeg


# Ricava la lista dei nomi dei file relativi alle sessioni di registrazione

# In[ ]:


# define functions for oder lists
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def invertPersonalAndOther(sessions):
    for sub_id in range(int(len(sessions)/2)):
        ot = sub_id*2
        pr = ot + 1
        temp = sessions[ot]
        sessions[ot] = sessions[pr]
        sessions[pr] = temp
    return sessions

def getRecordings(root_dir, file_ext='edf'):
    # define root directory for recordings
    root = root_dir
    # list all recordings
    recordings = os.listdir(root)
    print("In " + root + " there are " + str(len(recordings)) + " files")
    # compile regex for search file_ext files
    ext = re.compile(r'.'+file_ext)
    # select only edf files
    recordings = list(filter(lambda file: (ext.search(file)!=None), recordings)) 
    # natural sort of recordings
    recordings.sort(key=natural_keys)
    print("Getting " + file_ext +" files...\n" + str(len(recordings)) + " found.")
    return recordings

def getSessions(root_dir, file_ext='edf'):
    return getRecordings(root_dir, file_ext=file_ext)

def getPrunedSessions(root_dir):
    return invertPersonalAndOther(getRecordings(root_dir, file_ext='set'))

def sub(sub_id, other = False):
    if other:
        return sub_id*2 + 1
    else:
        return sub_id*2
    
def sub_ot(sub_id):
    return sub(sub_id, True)


#=============================================================
# GET DATASET LABELS, AND SELECT EMOTIONS
#=============================================================
def get_dataset_labels(data, select_second_label=False):
    """
    Get a list of emotion labels, given a DataFrame containing dataset entries.
    """
    if select_second_label:
        return [eval(row.gew_2)[0] if(type(row.gew_2)==str) else None for _, row in ds.iterrows()]
    return [eval(row.gew_1)[0] for _, row in data.iterrows()]

def select_emotion(labels, emotion_to_select):
    """
    Given a label and a list of labels, returns the indices of samples that have that label
    """
    return [i for i, x in enumerate(labels) if x == emotion_to_select]
