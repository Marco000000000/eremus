from scipy.stats import median_abs_deviation

import mne
import numpy as np
import pandas as pd
import mne_features

from sklearn import svm
from eremus_utils import Session
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (train_test_split, StratifiedKFold)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mne_features.feature_extraction import extract_features, FeatureExtractor

# Symmetric Electrodes
symmetric_electrodes_names = {
 'Fp1': 'Fp2',
 'F7': 'F8',
 'F3': 'F4',
 'FC1': 'FC2',
 'C3': 'C4',
 'FC5': 'FC6',
 'FT9': 'FT10',
 'T7': 'T8',
 'CP5': 'CP6',
 'CP1': 'CP2',
 'P3': 'P4',
 'P7': 'P8',
 'PO9': 'PO10',
 'O1': 'O2'}

symmetric_electrodes = {
    Session.chs[k]: Session.chs[v]
    for k, v in symmetric_electrodes_names.items()
}

def compute_channel_covariance(data):
    """Covariance of the data (per channel, beetween channels).
    Parameters
    ----------
    data : shape (n_channels, n_times)
    Returns
    -------
    output : ndarray, shape (n_channels, n_channels)
    Notes
    -----
    Alias of the feature function: **channel_covariance**
    """
    return np.cov(data, ddof=1)

def compute_covariance(data, k=128):
    """Covariance of the data (per channel, beetween samples at distance k).
    Parameters
    ----------
    data : shape (n_channels, n_times)
    k : int
    Returns
    -------
    output : ndarray, shape (n_channels,)
    Notes
    -----
    Alias of the feature function: **covariance**
    """
    # get shape
    n_channels, n_times = data.shape
    assert n_times > k, 'Please provide a k value smaller than ' + str(n_times)
    
    # init covariance
    cov = np.zeros(n_channels)
    
    # iter channels
    for ch in range(n_channels):
        # select current time-points
        t = data[ch,  0:n_times-k]
        # select future time-points (current+k)
        t_k = data[ch, k:n_times]
        # calculate covariance
        cov[ch] = np.cov(t, t_k, ddof=1)[0, 1]

    return cov

def compute_mean_absolute_deviation(data):
    """Mean absolute deviation of the data (per channel).
    Parameters
    ----------
    data : shape (n_channels, n_times)
    Returns
    -------
    output : ndarray, shape (n_channels,)
    Notes
    -----
    Alias of the feature function: **mean_absolute_deviation**
    """
    return np.sum(data - np.mean(data, axis=-1).reshape(-1, 1), axis=-1)/data.shape[-1]

def compute_median(data):
    """Median of the data (per channel).
    Parameters
    ----------
    data : shape (n_channels, n_times)
    Returns
    -------
    output : ndarray, shape (n_channels,)
    Notes
    -----
    Alias of the feature function: **median**
    """
    return np.median(data, axis=-1)

def compute_median_absolute_deviation(data):
    """Median absolute deviation of the data (per channel).
    Parameters
    ----------
    data : shape (n_channels, n_times)
    Returns
    -------
    output : ndarray, shape (n_channels,)
    Notes
    -----
    Alias of the feature function: **median_absolute_deviation**
    """
    return median_abs_deviation(data, axis=-1)

def compute_maximum(data):
    """Maximum of the data (per channel).
    Parameters
    ----------
    data : shape (n_channels, n_times)
    Returns
    -------
    output : ndarray, shape (n_channels,)
    Notes
    -----
    Alias of the feature function: **maximum**
    """
    return np.max(data, axis=-1)

def compute_minimum(data):
    """Minimum of the data (per channel).
    Parameters
    ----------
    data : shape (n_channels, n_times)
    Returns
    -------
    output : ndarray, shape (n_channels,)
    Notes
    -----
    Alias of the feature function: **minimum**
    """
    return np.min(data, axis=-1)

def compute_upper_quartile(data):
    """Upper Quartile of the data (per channel).
    Parameters
    ----------
    data : shape (n_channels, n_times)
    Returns
    -------
    output : ndarray, shape (n_channels,)
    Notes
    -----
    Alias of the feature function: **upper_quartile**
    """
    return np.quantile(data, 0.75, axis=-1)

def compute_lower_quartile(data):
    """Lower Quartile of the data (per channel).
    Parameters
    ----------
    data : shape (n_channels, n_times)
    Returns
    -------
    output : ndarray, shape (n_channels,)
    Notes
    -----
    Alias of the feature function: **lower_quartile**
    """
    return np.quantile(data, 0.25, axis=-1)


# ### Frequency Domain Features
# 
# Based on Hong He et al. (2020):
# The following features should be calculated over 5 different frequency bands, delta (δ: 1–4 Hz), theta (θ: 4–8 Hz), alpha(α: 8–13 Hz), beta (β:13–30 Hz), and gamma (γ : 31–50 Hz):
# - Power Spectral Density 
# - Band Power
# - Band Power Ratio
# 
# All these features could be computated using mne_features functios:
# - Power Spectral Density --> just wrap utils.power_spectrum function, bands are not equal in length, so we can compute PSD over all the spectrum
# - Band Power --> compute_pow_freq_bands with normalize=False,
# - Band Power Ratio --> compute_pow_freq_bands with normalize=True (default use)

def compute_psd(data, sfreq=128, flatten=False):
    """PSD of the data (per channel).
    Parameters
    ----------
    data : shape (n_channels, n_times)
    sfreq: int
    Returns
    -------
    output : ndarray, shape (n_channels, n_freqs) if flatten is False (default)
             ndarray, shape (n_channels*n_freqs,) if flatten is True
    Notes
    -----
    Alias of the feature function: **psd**
    """
    psd, _ = mne_features.utils.power_spectrum(sfreq, data, fmin=0.5, fmax=sfreq/2)
    if flatten:
        psd = psd.reshape(-1)
    return psd

def compute_pow_bands(data, sfreq=128, freq_bands=np.array([0.5, 4.0, 8.0, 13.0, 30.0, 64.0]), flatten=False):
    """Power Bands of the data (per channel).
    Parameters
    ----------
    data : shape (n_channels, n_times)
    sfreq: int
    freq_bands: see compute_pow_freq_bands docs
    Returns
    -------
    output : ndarray, shape (n_channels, n_bands) if flatten is False (default)
             ndarray, shape (n_channels*n_bands,) if flatten is True
    Notes
    -----
    Alias of the feature function: **pow_bands**
    """
    C, _ = data.shape
    if flatten:
        return mne_features.univariate.compute_pow_freq_bands(sfreq, data, freq_bands=freq_bands, normalize=False)
    else:
        return mne_features.univariate.compute_pow_freq_bands(sfreq, data, freq_bands=freq_bands, normalize=False).reshape(C, -1)

def compute_differential_pow_bands(data, sfreq=128, freq_bands=np.array([0.5, 4.0, 8.0, 13.0, 30.0, 64.0]), flatten=False, apply_padding=True):
    """Differential Bands Power of the data (per symmetric channels).
    Parameters
    ----------
    data : shape (n_channels, n_times)
    sfreq : int
    freq_bands : see compute_pow_freq_bands docs
    flatten : bool
    apply_padding : bool
    
    Returns
    -------
    output : ndarray, shape (n_channels/2, n_bands) if !flatten and apply_padding (default)
             ndarray, shape (n_channels/2*n_bands,) if flatten and apply_padding
             ndarray, shape (n_sym, n_bands) if !flatten and !apply_padding 
             ndarray, shape (n_sym*n_bands,) if flatten and !apply_padding
    Notes
    -----
    Alias of the feature function: **differential_pow_bands**
    """
    C, _ = data.shape
    B = freq_bands.shape[0] - 1
    bp = mne_features.univariate.compute_pow_freq_bands(sfreq, data, freq_bands=freq_bands, normalize=False).reshape(C, B)
    
    dbp = np.zeros((C//2, B))
    for channel, symmetric in symmetric_electrodes.items():
        dbp[channel - 1] = bp[channel, :] - bp[symmetric, :]
        
    if not apply_padding:
        dbp = dbp[1:-1, :]
    
    return dbp.reshape(-1) if flatten else dbp 

# already implemented temporal features
temporal_features = {
     'hjorth_complexity',
     'hjorth_mobility',
     'kurtosis',
     'mean',
     'ptp_amp',
     'skewness',
     'std',
     'variance',
     'zero_crossings'}

# Other frequency domain features
frequency_domain_features = {
 'app_entropy',
 'decorr_time',
 'energy_freq_bands',
 'higuchi_fd',
 'hjorth_complexity_spect',
 'hjorth_mobility_spect',
 'hurst_exp',
 'katz_fd',
 'line_length',
 'samp_entropy',
 'spect_edge_freq',
 'spect_entropy',
 'spect_slope',
 'svd_entropy',
 'svd_fisher_info',
 'teager_kaiser_energy',
 'wavelet_coef_energy'}