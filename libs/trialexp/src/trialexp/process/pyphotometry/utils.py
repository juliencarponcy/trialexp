# Utility functions for pycontrol and pyphotometry files processing

import itertools
import json
import logging

import os
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt, medfilt
from scipy.stats import linregress, zscore
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import re

from trialexp.process.pycontrol.event_filters import extract_event_time
from trialexp.utils.rsync import *
from scipy.optimize import curve_fit
from scipy import signal
import warnings
from scipy.stats import median_abs_deviation
from datetime import datetime
from loguru import logger

'''
Most of the photometry data processing functions are based on the intial design
of the pyPhotometry package. They are stored in a dictionary containing both
metadata and the data. The dictionary is returned by the import_ppd function.

Assumptions:
    - Analog 1 is the isosbestic control
    - Analog 2 is the signal of interest
'''


#----------------------------------------------------------------------------------
# Plotting
#----------------------------------------------------------------------------------


#----------------------------------------------------------------------------------
# Get photometry data by trials
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# Motion correction / Normalization 
#----------------------------------------------------------------------------------

# Note that there is a dependency in the workflow between these filtering 
# and normalization functions. The normalization functions assume that the
# data has already been filtered.



def denoise_filter(photometry_dict:dict, lowpass_freq, highpass_freq=None) -> dict:
    # apply a low-pass filter to remove high frequency noise
    b,a = get_filt_coefs(low_pass=lowpass_freq, high_pass=highpass_freq, sampling_rate=photometry_dict['sampling_rate'])
    pattern = re.compile('analog_[0-9]$')
    signal_names = [ k for k in photometry_dict.keys() if pattern.match(k)]

    for sig in signal_names:
        sig_filt = filtfilt(b, a, photometry_dict[sig], padtype='even')        
        photometry_dict[f'{sig}_filt'] = sig_filt
    
    return photometry_dict
    
def motion_correction_win(photometry_dict: dict) -> dict:
    # use analog_2_filt to remove the motion artifact in analog_1_filt

    if any(['analog_1_filt' not in photometry_dict, 'analog_2_filt' not in photometry_dict]):
        raise Exception('Analog 1 and Analog 2 must be filtered before motion correction')


    try:

        win_size_s = 30

        n_win_size = int(win_size_s * photometry_dict['sampling_rate'])
        n_overlap = int(n_win_size/2)  # 50% overlap

        def overlapping_chunks(data1, data2, n_win_size: int, n_overlap: int):
            # if not isinstance(chunk_size, int) or not chunk_size.is_integer():
            #     raise ValueError("chunk_size must be an integer")
                
            x = n_win_size/n_overlap

            if isinstance(x, float) and not x.is_integer():
                raise ValueError("1/overlap_ratio must be an integer")
            elif isinstance(x, int):
                pass

            step_size = n_win_size - int(n_win_size) // int(x)
            # for i in range(0, len(data1) - n_win_size + 1, step_size): #TODO this will skip the last iteration
            #     yield i, data1[i:i + n_win_size], data2[i:i + n_win_size]
            for i in range(0, len(data1), step_size):
                if i + n_win_size <= len(data1):  # If a full-sized chunk can be taken
                    yield i, data1[i:i + n_win_size], data2[i:i + n_win_size]
                else:  # If only a truncated chunk can be taken
                    yield i, data1[i:], data2[i:]


        start_index_chunks = []
        analog_1_est_motion_chunks = []
        p_vals = []
        r_vals = []
        for start_ind, chunk1, chunk2 in overlapping_chunks(
                photometry_dict['analog_1_filt'], photometry_dict['analog_2_filt'], 
                n_win_size, n_overlap):
            slope, intercept, r_value, p_value, std_err = linregress(chunk2, chunk1)

            start_index_chunks.append(start_ind)

            analog_1_est_motion_chunks.append(slope * chunk2 + intercept)
            p_vals.append(p_value)
            r_vals.append(r_value)

        analog_1_est_motion_joined = np.zeros(np.size(photometry_dict['analog_1_filt']))

        step_size = n_win_size - int(n_win_size) // int(n_win_size/n_overlap)

        for i, _ in enumerate(start_index_chunks):
            ch = analog_1_est_motion_chunks[i]
            
            if i == 0:
                analog_1_est_motion_joined[0:step_size] = ch[0:step_size]

            elif i > 0 and i < len(start_index_chunks) -1:
                ch_prev = analog_1_est_motion_chunks[i-1]
                ind_this =  start_index_chunks[i]
                
                # average two chunks
                analog_1_est_motion_joined[ind_this:ind_this + step_size] \
                    = (ch[0:step_size] + ch_prev[step_size-1:-1])/2

            elif i == len(start_index_chunks) -1 :
                ind_this =  start_index_chunks[i]

                # copy one chunk
                analog_1_est_motion_joined[ind_this:ind_this + step_size] = ch

            elif i == len(start_index_chunks):
                print('should not happen')

        photometry_dict['analog_1_est_motion'] = analog_1_est_motion_joined
        photometry_dict['analog_1_corrected'] = photometry_dict['analog_1_filt'] - analog_1_est_motion_joined
        photometry_dict['motion_corrected'] = 1

    except ValueError:
        print('Motion correction failed. Skipping motion correction')
        # probably due to saturation , do not do motion correction
        photometry_dict['analog_1_corrected'] = photometry_dict['analog_1_filt']
        photometry_dict['motion_corrected'] = 0

    return photometry_dict


def overlapping_chunks(data1, data2, n_win_size: int, n_overlap: int):
    """
    Generate overlapping chunks of data from two arrays.

    Parameters:
    -----------
    data1 : array-like
        First array of data.
    data2 : array-like
        Second array of data.
    n_win_size : int
        Size of the window used to extract chunks of data.
    n_overlap : int
        Number of samples to overlap between consecutive chunks.

    Yields:
    -------
    tuple
        A tuple containing the index of the first sample of the chunk, the chunk of data from `data1`, and the chunk of data from `data2`.

    Raises:
    -------
    ValueError
        If `n_win_size/n_overlap` is not an integer.

    Notes:
    ------
    The function generates chunks of size `n_win_size` from the input arrays, with an overlap of `n_overlap` samples between consecutive chunks. If the last chunk is truncated, it is still returned.

    Examples:
    ---------
    >>> data1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> data2 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    >>> for i, chunk1, chunk2 in overlapping_chunks(data1, data2, n_win_size=4, n_overlap=2):
    ...     print(i, chunk1, chunk2)
    0 [1, 2, 3, 4] [10, 20, 30, 40]
    2 [3, 4, 5, 6] [30, 40, 50, 60]
    4 [5, 6, 7, 8] [50, 60, 70, 80]
    6 [7, 8, 9, 10] [70, 80, 90, 100]
    """

    x = n_win_size/n_overlap

    if isinstance(x, float) and not x.is_integer():
        raise ValueError("1/overlap_ratio must be an integer")

    step_size = n_win_size - int(n_win_size) // int(x)
    # for i in range(0, len(data1) - n_win_size + 1, step_size): #TODO this will skip the last iteration
    #     yield i, data1[i:i + n_win_size], data2[i:i + n_win_size]
    for i in range(0, len(data1), step_size):
        if i + n_win_size <= len(data1):  # If a full-sized chunk can be taken
            yield i, data1[i:i + n_win_size], data2[i:i + n_win_size]
        else:  # If only a truncated chunk can be taken
            yield i, data1[i:], data2[i:]
    
    
    
def window_subtraction(analog1, analog2, sampling_rate, win_size_s=30) -> dict:
    '''
    subtract analog2 from analog1 by spitting into overlapping time windows, do a linear fit, and then subtract
    Have the advantage of only affecting a limited region with shadow artifacts, for region where there is
    no artifacts, the slope should be close to zero. Issue a warning when large negative slope is detected
    '''
    
    n_win_size = int(win_size_s * sampling_rate)
    n_overlap = int(n_win_size/2)  # 50% overlap


    # Splitting the data into chucks and do fitting on each chunk
    start_index_chunks = []
    analog_1_est_motion_chunks = []
    p_vals = []
    r_vals = []
    for start_ind, chunk1, chunk2 in overlapping_chunks(
            analog1, analog2, 
            n_win_size, n_overlap):
        try:
            slope, intercept, r_value, p_value, std_err = linregress(chunk2, chunk1)
        except ValueError as e:
            print(f'Warning: regression error. I will skip wndow subtraction for {start_ind}')
            r_value = 0
            p_value = 0
            r_value = 0
            slope = 0
            
        start_index_chunks.append(start_ind)

        analog_1_est_motion_chunks.append(slope * chunk2 + intercept)
        p_vals.append(p_value)
        r_vals.append(r_value)
        
    if any(np.array(r_vals)< -0.2):
        warnings.warn('Some slope has a large negative value!')

    # Joining the fitted data together
    analog_1_est_motion_joined = np.zeros(np.size(analog1))

    step_size = n_win_size - int(n_win_size) // int(n_win_size/n_overlap)

    for i, _ in enumerate(start_index_chunks):
        ch = analog_1_est_motion_chunks[i] # the motion of analog_1 is estimated from analog_2
        
        if i == 0:
            analog_1_est_motion_joined[0:step_size] = ch[0:step_size]

        elif i > 0 and i < len(start_index_chunks) -1:
            ch_prev = analog_1_est_motion_chunks[i-1]
            ind_this =  start_index_chunks[i]
            
            # average two chunks
            analog_1_est_motion_joined[ind_this:ind_this + step_size] \
                = (ch[0:step_size] + ch_prev[step_size-1:-1])/2

        elif i == len(start_index_chunks) -1 :
            ind_this =  start_index_chunks[i]

            # copy one chunk
            analog_1_est_motion_joined[ind_this:ind_this + step_size] = ch

        elif i == len(start_index_chunks):
            raise ValueError('Accidental reaching the last chunk. Should not happen')

    analog_1_est_motion = analog_1_est_motion_joined
    analog_1_corrected = analog1 - analog_1_est_motion_joined
    
    return analog_1_est_motion, analog_1_corrected
        

def motion_correction(photometry_dict: dict) -> dict:
    
    if any(['analog_1_filt' not in photometry_dict, 'analog_2_filt' not in photometry_dict]):
        raise Exception('Analog 1 and Analog 2 must be filtered before motion correction')
    
    try:
        slope, intercept, r_value, p_value, std_err = linregress(x=photometry_dict['analog_2_filt'], y=photometry_dict['analog_1_filt'])
        photometry_dict['analog_1_est_motion'] = intercept + slope * photometry_dict['analog_2_filt']
        photometry_dict['analog_1_corrected'] = photometry_dict['analog_1_filt'] - photometry_dict['analog_1_est_motion']
        photometry_dict['motion_corrected'] = 1
    except ValueError:
        print('Motion correction failed. Skipping motion correction')
        # probably due to saturation , do not do motion correction
        photometry_dict['analog_1_corrected'] = photometry_dict['analog_1_filt']
        photometry_dict['motion_corrected'] = 0

    return photometry_dict



def compute_df_over_f(photometry_dict: dict, low_pass_cutoff: float = 0.001) -> dict:
    
    if 'analog_1_corrected' not in photometry_dict:
        raise Exception('Analog 1 must be motion corrected before computing dF/F')
    
    b,a = butter(2, low_pass_cutoff, btype='low', fs=photometry_dict['sampling_rate'])
    photometry_dict['analog_1_baseline_fluo'] = filtfilt(b,a, photometry_dict['analog_1_filt'], padtype='even')
    photometry_dict['analog_2_baseline_fluo'] = filtfilt(b,a, photometry_dict['analog_2_filt'], padtype='even')

    # Now calculate the dF/F by dividing the motion corrected signal by the time varying baseline fluorescence.
    photometry_dict['analog_1_df_over_f'] = photometry_dict['analog_1_corrected'] / photometry_dict['analog_1_baseline_fluo'] 
    
    if 'analog_2_corrected' in photometry_dict.keys():
        photometry_dict['analog_2_df_over_f'] = photometry_dict['analog_2_corrected'] / photometry_dict['analog_2_baseline_fluo']
    else:
        photometry_dict['analog_2_df_over_f'] = photometry_dict['analog_2_filt'] / photometry_dict['analog_2_baseline_fluo']
    
    if 'analog_3_detrended' in photometry_dict:
        photometry_dict['analog_3_baseline_fluo'] = filtfilt(b,a, photometry_dict['analog_3_filt'], padtype='even')
        photometry_dict['analog_3_df_over_f'] = photometry_dict['analog_3_detrended'] / photometry_dict['analog_3_baseline_fluo']

    
    return photometry_dict

#----------------------------------------------------------------------------------
# Filtering 
#----------------------------------------------------------------------------------

def compute_zscore(photometry_dict):
    photometry_dict['zscored_df_over_f'] = zscore(photometry_dict['analog_1_df_over_f'])
    
    if 'analog_2_corrected' in photometry_dict:
        photometry_dict['zscored_df_over_f_analog_2'] = zscore(photometry_dict['analog_2_df_over_f'])
        
    if 'analog_3_detrended' in photometry_dict:
        photometry_dict['zscored_df_over_f_analog_3'] = zscore(photometry_dict['analog_3_df_over_f'])

    return photometry_dict

def median_filtering(data, medfilt_size: int = 3) -> np.ndarray:
    
    if medfilt_size % 2 == 0:
        raise Exception('medfilt_size must be an odd number') 
    
    data = medfilt(data,medfilt_size)
    
    return data

def get_filt_coefs(low_pass: int = None, high_pass: int = None, sampling_rate = 1000):
    if low_pass and high_pass:
        b, a = butter(2, np.array([high_pass, low_pass])/(0.5*sampling_rate), 'bandpass')
    elif low_pass:
        b, a = butter(2, low_pass/(0.5*sampling_rate), 'low')
    elif high_pass:
        b, a = butter(2, high_pass/(0.5*sampling_rate), 'high')
    
    return b,a

#----------------------------------------------------------------------------------
# Exponential fitting currently not in use in our pipelines
#----------------------------------------------------------------------------------

# The exponential curve we are going to fit.
def exp_func(x, a, b, c):
   return a*np.exp(-b*x) + c


def fit_exp_func(data, fs: int = 100, medfilt_size: int = 3) -> np.ndarray:
    '''
    compute the exponential fitted to data. This unused in current filtering because
    unsuitable when behavioural box openings / closing provoked transitory changes
    in baseline fluorescence.
    '''
    if medfilt_size % 2 == 0:
        raise Exception('medfilt_size must be an odd number') 
    
    time = np.linspace(1/fs, len(data)/fs, len(data))

    fit_params, parm_cov = curve_fit(
        exp_func, time, medfilt(data,medfilt_size),
        p0=[1,1e-3,1],bounds=([0,0,0],[4,0.1,4]), maxfev=1000)

    fitted_data = exp_func(time, * fit_params)


    return fitted_data


def get_dataformat(df_dataformat, session_id):

    def parse_date(date_string):
        return datetime.strptime(date_string, '%Y-%m-%d')

    # use the session id to determine which dataformat to use
    df_dataformat['date_cutoff'] = df_dataformat.date.apply(parse_date)
    curr_date = '-'.join(session_id.split('-')[1:4])
    curr_date = parse_date(curr_date)
    idx = np.searchsorted(df_dataformat.date_cutoff, curr_date)
    if idx>0:
        data_format = df_dataformat.iloc[idx-1].photom_format
    else:
        data_format = 'v1'
    
    return data_format

def import_ppd_auto(file_path, cutoff_date = datetime(2023,10,23)):
    # Automatically import ppd data using the correct file format
    with open(file_path, 'rb') as f:
        header_size = int.from_bytes(f.read(2), 'little')
        data_header = f.read(header_size)
    # Extract header information
    header_dict = json.loads(data_header)
    
    if header_dict['version'] == '1.0.2':
        return import_ppd_v2(file_path,  low_pass=None, high_pass=None)
    else:
        # parse the datetime to determine the correct file format to use
        start_date = datetime.strptime(header_dict['date_time'], '%Y-%m-%dT%H:%M:%S.%f') 
        if start_date > cutoff_date:
            data_format = 'v2'
        else:
            data_format = 'v1'
                    
        return import_ppd(file_path, data_format)

def import_ppd_v2(file_path, low_pass=20, high_pass=0.01):
    """Function to import pyPhotometry binary data files into Python. The high_pass
    and low_pass arguments determine the frequency in Hz of highpass and lowpass
    filtering applied to the filtered analog signals. To disable highpass or lowpass
    filtering set the respective argument to None.  Returns a dictionary with the
    following items:
        'filename'      - Data filename
        'subject_ID'    - Subject ID
        'date_time'     - Recording start date and time (ISO 8601 format string)
        'end_time'      - Recording end date and time (ISO 8601 format string)
        'mode'          - Acquisition mode
        'sampling_rate' - Sampling rate (Hz)
        'LED_current'   - Current for LEDs 1 and 2 (mA)
        'version'       - Version number of pyPhotometry
        'analog_1'      - Raw analog signal 1 (volts)
        'analog_2'      - Raw analog signal 2 (volts)
        'analog_3'      - Raw analog signal 3 (if present, volts)
        'analog_1_filt' - Filtered analog signal 1 (volts)
        'analog_2_filt' - Filtered analog signal 2 (volts)
        'analog_3_filt' - Filtered analog signal 2 (if present, volts)
        'digital_1'     - Digital signal 1
        'digital_2'     - Digital signal 2 (if present)
        'pulse_inds_1'  - Locations of rising edges on digital input 1 (samples).
        'pulse_inds_2'  - Locations of rising edges on digital input 2 (samples).
        'pulse_times_1' - Times of rising edges on digital input 1 (ms).
        'pulse_times_2' - Times of rising edges on digital input 2 (ms).
        'time'          - Time of each sample relative to start of recording (ms)
    """
    with open(file_path, "rb") as f:
        header_size = int.from_bytes(f.read(2), "little")
        data_header = f.read(header_size)
        data = np.frombuffer(f.read(), dtype=np.dtype("<u2"))
    # Extract header information
    header_dict = json.loads(data_header)
    volts_per_division = header_dict["volts_per_division"]
    sampling_rate = header_dict["sampling_rate"]
    # Extract signals.
    analog = data >> 1  # Analog signal is most significant 15 bits.
    digital = ((data & 1) == 1).astype(int)  # Digital signal is least significant bit.
    # Alternating samples are different signals.
    if "n_analog_signals" in header_dict.keys():
        n_analog_signals = header_dict["n_analog_signals"]
        n_digital_signals = header_dict["n_digital_signals"]
    else:  # Pre version 1.0 data file.
        n_analog_signals = 2
        n_digital_signals = 2
    
    
    if not n_analog_signals == 6:
        analog_1 = analog[::n_analog_signals] * volts_per_division[0]
        analog_2 = analog[1::n_analog_signals] * volts_per_division[1]
        analog_3 = analog[2::n_analog_signals] * volts_per_division[0] if n_analog_signals == 3 else None


        digital_1 = digital[::n_analog_signals]
        digital_2 = digital[1::n_analog_signals] if n_digital_signals == 2 else None
    else:
        # work with the bleedthrough channel
        analog_1 = analog[::n_analog_signals] * volts_per_division[0]
        analog_2 = analog[2::n_analog_signals] * volts_per_division[1]
        analog_3 = analog[4::n_analog_signals] * volts_per_division[0] 

        bleedthrough_ch2 = analog[1::n_analog_signals] * volts_per_division[0]
        bleedthrough_ch1 = analog[3::n_analog_signals] * volts_per_division[1]
        bleedthrough_isos = analog[5::n_analog_signals] * volts_per_division[0]

        digital_1 = digital[::n_analog_signals]
        digital_2 = digital[2::n_analog_signals]

    time = np.arange(analog_1.shape[0]) * 1000 / sampling_rate  # Time relative to start of recording (ms).
    # Filter signals with specified high and low pass frequencies (Hz).
    if low_pass and high_pass:
        b, a = butter(2, np.array([high_pass, low_pass]) / (0.5 * sampling_rate), "bandpass")
    elif low_pass:
        b, a = butter(2, low_pass / (0.5 * sampling_rate), "low")
    elif high_pass:
        b, a = butter(2, high_pass / (0.5 * sampling_rate), "high")
    if low_pass or high_pass:
        analog_1_filt = filtfilt(b, a, analog_1)
        analog_2_filt = filtfilt(b, a, analog_2)
        analog_3_filt = filtfilt(b, a, analog_3) if n_analog_signals == 3 else None
    else:
        analog_1_filt = analog_2_filt = analog_3_filt = None
    # Extract rising edges for digital inputs.
    pulse_inds_1 = 1 + np.where(np.diff(digital_1) == 1)[0]
    pulse_inds_2 = 1 + np.where(np.diff(digital_2) == 1)[0] if n_digital_signals == 2 else None
    pulse_times_1 = pulse_inds_1 * 1000 / sampling_rate
    pulse_times_2 = pulse_inds_2 * 1000 / sampling_rate if n_digital_signals == 2 else None
    # Return signals + header information as a dictionary.
    data_dict = {
        "filename": os.path.basename(file_path),
        "analog_1": analog_1,
        "analog_2": analog_2,
        "analog_1_filt": analog_1_filt,
        "analog_2_filt": analog_2_filt,
        "digital_1": digital_1,
        "digital_2": digital_2,
        "pulse_inds_1": pulse_inds_1,
        "pulse_inds_2": pulse_inds_2,
        "pulse_times_1": pulse_times_1,
        "pulse_times_2": pulse_times_2,
        "time": time,
    }

    if n_analog_signals == 3 or n_analog_signals == 6:
        data_dict.update(
            {
                "analog_3": analog_3,
                "analog_3_filt": analog_3_filt,
            }
        )
    
    if n_analog_signals ==6:
        data_dict.update(
            {
                'bleedthrough_ch1': bleedthrough_ch1,
                'bleedthrough_ch2': bleedthrough_ch2,
                'bleedthrough_isos': bleedthrough_isos
            }
        )

    data_dict.update(header_dict)
    return data_dict

#----------------------------------------------------------------------------------
# Load analog data
#----------------------------------------------------------------------------------

def import_ppd(file_path, data_format='v2'):
    '''Function to import pyPhotometry binary data files into Python. The high_pass 
    and low_pass arguments determine the frequency in Hz of highpass and lowpass 
    filtering applied to the filtered analog signals. To disable highpass or lowpass
    filtering set the respective argument to None.  Returns a dictionary with the 
    following items:
        'subject_ID'    - Subject ID
        'date_time'     - Recording start date and time (ISO 8601 format string)
        'mode'          - Acquisition mode
        'sampling_rate' - Sampling rate (Hz)
        'LED_current'   - Current for LEDs 1 and 2 (mA)
        'version'       - Version number of pyPhotometry
        'analog_1'      - Raw analog signal 1 (volts)
        'analog_2'      - Raw analog signal 2 (volts)
        'analog_1_filt' - Filtered analog signal 1 (volts)
        'analog_2_filt' - Filtered analog signal 2 (volts)
        'digital_1'     - Digital signal 1
        'digital_2'     - Digital signal 2
        'pulse_inds_1'  - Locations of rising edges on digital input 1 (samples).
        'pulse_inds_2'  - Locations of rising edges on digital input 2 (samples).
        'pulse_times_1' - Times of rising edges on digital input 1 (ms).
        'pulse_times_2' - Times of rising edges on digital input 2 (ms).
        'time'          - Time of each sample relative to start of recording (ms)
        
        data_format: v1 for hybrid recordig is GFP, isosbestic, RFP
         v2 for hybrid recording is GFP, RFP isosbestic
    '''
    with open(file_path, 'rb') as f:
        header_size = int.from_bytes(f.read(2), 'little')
        data_header = f.read(header_size)
        data = np.frombuffer(f.read(), dtype=np.dtype('<u2'))
    # Extract header information
    header_dict = json.loads(data_header)
    volts_per_division = header_dict['volts_per_division']
    sampling_rate = header_dict['sampling_rate']
    # Extract signals.
    analog  = data >> 1                     # Analog signal is most significant 15 bits.
    digital = ((data & 1) == 1).astype(int) # Digital signal is least significant bit.
    analog_3 = None

    
    if header_dict['mode'] == '1 colour continuous + 2 colour time div.' or header_dict['mode'] == '3 colour time div.':
        # there are 3 analog signals, try to make the name compatible with exisiting recordings
        if data_format == 'v1':
            analog_1 = analog[::3] * volts_per_division[0]  # GFP
            analog_3 = analog[1::3] * volts_per_division[0] # isosbestic
            analog_2 = analog[2::3] * volts_per_division[1] # RFP
        else:
            analog_1 = analog[::3] * volts_per_division[0]  # GFP
            analog_2 = analog[1::3] * volts_per_division[1] # RFP
            analog_3 = analog[2::3] * volts_per_division[0] # isosbestic
            
        digital_1 = digital[::3]
        digital_2 = digital[1::3]
        digital_3 = digital[2::3] #workaround for wrong data format in some recording
    else:
        # Alternating samples are signals 1 and 2.
        analog_1 = analog[::2] * volts_per_division[0]
        analog_2 = analog[1::2] * volts_per_division[1]
        digital_1 = digital[::2]
        digital_2 = digital[1::2]
        
    # Time relative to start of recording (ms).
    time = np.arange(analog_1.shape[0]).astype(np.int64)*1000/sampling_rate #warning: default data type np.int32 will lead to overflow
    # time = np.arange(analog_1.shape[0])*1000/sampling_rate #warning: default data type np.int32 will lead to overflow
   
    # Extract rising edges for digital inputs.
    pulse_inds_1 = 1+np.where(np.diff(digital_1) == 1)[0]
    pulse_inds_2 = 1+np.where(np.diff(digital_2) == 1)[0]
    pulse_times_1 = pulse_inds_1*1000/sampling_rate
    pulse_times_2 = pulse_inds_2*1000/sampling_rate
    # Return signals + header information as a dictionary.
    data_dict = {'analog_1'      : analog_1,
                 'analog_2'      : analog_2,
                 'digital_1'     : digital_1,
                 'digital_2'     : digital_2,
                 'pulse_inds_1'  : pulse_inds_1,
                 'pulse_inds_2'  : pulse_inds_2,
                 'pulse_times_1' : pulse_times_1,
                 'pulse_times_2' : pulse_times_2,
                 'time'          : time,
                 'data_format' : data_format}
    
    if analog_3 is not None:
        pulse_inds_3 = 1 + np.where(np.diff(digital_3)==1)[0]
        pulse_times_3  = pulse_inds_3*1000/sampling_rate
        data_dict.update({
            'analog_3': analog_3,
            'digital_3': digital_3,
            'pulse_times_3': pulse_times_3})
    
    
    # Add metadata to dictionary.
    data_dict.update(header_dict)
    
    return data_dict

#----------------------------------------------------------------------------------
# Rsync functions
#----------------------------------------------------------------------------------


def sync_photometry_file(
        session_file: str,
        photometry_file: str = None, 
        rsync_chan: int = 2,
        delete_unsynced: bool = True, 
        verbose: bool = False):
    """
    This function create a rsync aligment object into the corresponding
    session if the rsync pulses match betwwen pycontrol and pyphotometry files.

        Parameters:
            session_file (str): PyControl txt file path 
            photometry_file (str): PyPhotometry ppd file path 
            rsync_chan (int): Channel on which pulses have been
                recorded on the py_photometry device.
            delete_unsynced (bool): Delete the photometry file path in
                session.files['ppd'] if rsync does not match
            verbose (bool): display match/no match messages for each file

        Returns:
            None

        The warning:
            KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads...
        
        is due to rsync function.

        https://stackoverflow.com/questions/69596239/how-to-avoid-memory-leak-when-dealing-with-kmeans-for-example-in-this-code-i-am
        Follow the answer and set the einvironment variable OMP_NUM_THREADS to supress the warning.
                
    """
    
    session = Session(session_file, int_subject_IDs=True, verbose=False) 

    session.files = dict()
    session.files['ppd']  = [photometry_file] # list to make it backward compatible (implemented to allow for multiple matches [eg cameras])
    if photometry_file:
        # try to align times with rsync
        try:
            # Gives KeyError exception if no rsync pulses on pycontrol file
            pycontrol_rsync_times = session.times['rsync']
        
            photometry_dict = import_ppd(photometry_file)
            
            photometry_rsync_times = photometry_dict['pulse_times_' + str(rsync_chan)]

            photometry_rsync = Rsync_aligner(pulse_times_A= pycontrol_rsync_times, 
                pulse_times_B= photometry_rsync_times, plot=False)
            
            if verbose:
                print('pycontrol: ', session.subject_ID, session.datetime,
                '/ pyphotometry: ', photometry_file, ' : rsync does match')
            


        # if rsync aligner fails    
        except (RsyncError, ValueError, KeyError):
            photometry_rsync = None

            if verbose:
                print('pycontrol: ', session.subject_ID, session.datetime,
                '/ pyphotometry: ', photometry_file, ' : rsync does not match')

            if delete_unsynced:
                session.files['ppd'] = []

    # if there is no subject + date match in .ppd files
    else: 
        photometry_rsync = None

        if verbose:
            print('pycontrol: ', session.subject_ID, session.datetime,
            '/ pyphotometry: no file matching both subject and date')

    # for now return a session with embedded rsync object. 
    # Ouput will change when getting closer to fully functional implementation
    return photometry_rsync

#----------------------------------------------------------------------------------
# Peak / trough analyses
#----------------------------------------------------------------------------------


def get_trial_numbers_for_sessions(data_dir):
    """

    ## Output arguments
    df_trials              pd.DataFrame

    """ 

    dummy_dict = {'slope': np.nan, 'intercept': np.nan, 'r_value': np.nan, 'p_value': np.nan, 'std_err': np.nan}
    try:
        xr_photometry = xr.open_dataset(os.path.join(data_dir, 'xr_photometry.nc'))
    except Exception as e:
        #print(f"Caught an error: {e}")
        return [],  dummy_dict, dummy_dict, dummy_dict, False, str(e)

    try:
        xr_session = xr.open_dataset(os.path.join(data_dir, 'xr_session.nc'))
    except Exception as e:
        #print(f"Caught an error: {e}")
        return [],  dummy_dict, dummy_dict, dummy_dict, False, str(e)

    trial_nb_all = int(max(xr_session.trial_nb))

    # prepare df_trials for dip, rebound
    new_index = range(1, trial_nb_all+1)
    df_trials = pd.DataFrame({
        'trial_nb': list(range(1,trial_nb_all+1)),
        'outcome': xr_session['trial_outcome'].values.T.flatten(),  # flatten is used to convert (175, 1) to (175,)
        })
    df_trials

    return df_trials



def measure_DA_peak(data_dir):
    """
    #TODO refactro with measure_ACh_dip_rebound
    
    ## Output arguments
    df_trials              pd.DataFrame
    lin_regress_dip        dict
        dip vs trial_nb
    lin_regress_rebound    dict
        rebound vs trial_nb
    lin_regress_dip_rebound    dict
        dip vs rebound

        These dictionaries hold the outputs from linregress()
            'slope', 'intercept', 'r_value', 'p_value', 'std_err'
    is_success             bool 
    msg                    str
    """ 

    dummy_dict = {'slope': np.nan, 'intercept': np.nan, 'r_value': np.nan, 'p_value': np.nan, 'std_err': np.nan}
    try:
        xr_photometry = xr.open_dataset(os.path.join(data_dir, 'xr_photometry.nc'))
    except Exception as e:
        #print(f"Caught an error: {e}")
        return [],  dummy_dict, False, str(e)

    try:
        xr_session = xr.open_dataset(os.path.join(data_dir, 'xr_session.nc'))
    except Exception as e:
        #print(f"Caught an error: {e}")
        return [],  dummy_dict, False, str(e)

    trial_nb_all = int(max(xr_session.trial_nb))

    pk_values = []

    # Loop over trial numbers from 1 to trial_nb_all

    for k in range(1, trial_nb_all+1):
            try:
            # Calculate the mean over the specified event_time interval for reb
                pk = xr_photometry['hold_for_water_zscored_df_over_f'].sel(
                    trial_nb=k, event_time=slice(75, 250)).mean(dim='event_time')
            except Exception as e:
                #print(f"Caught an error: {e}")
                return [],  dummy_dict, False, str(e)
            # Append the value to the list
            pk_values.append(pk.values.max())

    # Convert lists to pandas Series
    pk_series = pd.Series(pk_values)

    # prepare df_trials for dip, rebound
    new_index = range(1, trial_nb_all+1)
    df_trials = pd.DataFrame({
        'trial_nb': list(range(1,trial_nb_all+1)),
        'peak': pk_series.reindex(new_index),
        'outcome': xr_session['trial_outcome'].values.T.flatten(),  # flatten is used to convert (175, 1) to (175,)
        })
    df_trials

    df_trials = df_trials.dropna(subset=['peak'])


    slope, intercept, r_value, p_value, std_err = linregress(df_trials['trial_nb'], df_trials['peak'])
    lin_regress_pk = {'slope': slope, 'intercept': intercept,
                      'r_value': r_value, 'p_value': p_value, 'std_err': std_err}

   
    return df_trials, lin_regress_pk, True, 'success'


#----------------------------------------------------------------------------------
# From here, legacy methods which will be probably deprecated in the future
#----------------------------------------------------------------------------------


def find_n_gaussians(
        data: np.ndarray,
        plot_results: bool = True,
        max_nb_gaussians: int = 4
    ) -> int: 
    '''
    Function to detect how many gaussians are needed to
    decribe a dataset.
    Re-use the original part to find M_best when there is
    more than 2-3 gaussians, and adjust the N range.
    Uncomment first original part to create artificial dataset

    Original author: Jake VanderPlas
    License: BSD
        The figure produced by this code is published in the textbook
        "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
        For more information, see http://astroML.github.com
        To report a bug or issue, use the following forum:
        https://groups.google.com/forum/#!forum/astroml-general
    
    '''
    #------------------------------------------------------------
    # Set up the dataset. 

    # median filter of raw red channel to remove small electric
    # data = medfilt(data,3)


    X = data.reshape(-1, 1)

    # (original code: We'll create our dataset by drawing samples from Gaussians)
    # random_state = np.random.RandomState(seed=1)

    # X = np.concatenate([random_state.normal(-1, 1.5, 350),
    #                     random_state.normal(0, 1, 500),
    #                     random_state.normal(3, 0.5, 150)]).reshape(-1, 1)

    #------------------------------------------------------------
    # Learn the best-fit GaussianMixture models
    #  Here we'll use scikit-learn's GaussianMixture model. The fit() method
    #  uses an Expectation-Maximization approach to find the best
    #  mixture of Gaussians for the data

    # fit models with 1-10 components
    N = np.arange(1, max_nb_gaussians)
    models = [None for i in range(len(N))]

    for i in range(len(N)):
        models[i] = GaussianMixture(N[i]).fit(X)

    # compute the AIC and the BIC
    AIC = [m.aic(X) for m in models]
    # BIC = [m.bic(X) for m in models]

    #------------------------------------------------------------
    # Plot the results
    #  We'll use three panels:
    #   1) data + best-fit mixture
    #   2) AIC and BIC vs number of components
    #   3) probability that a point came from each component

    # Original part, take the min of AIC to determine how many gaussians
    # M_best = models[np.argmin(AIC)]

    # Customized part tweaked to reduce the nb of gaussian used to the minimum
    diff_AIC = np.diff(np.diff(AIC))
    print(diff_AIC)  
    if diff_AIC[0] < 0:
        n_best = 0
    else:
        n_best = np.where(diff_AIC == min(diff_AIC))[0][0]+2
    

    M_best = models[n_best]
    # end of customized part

    p, bins = np.histogram(X, bins=np.arange(min(X),max(X),0.0002), density=True)
    print(len(bins))
    x = bins
    logprob = M_best.score_samples(x.reshape(-1, 1))
    # logprob = M_best.score_samples(x)

    responsibilities = M_best.predict_proba(x.reshape(-1, 1))
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    if plot_results:
        fig = plt.figure(figsize=(15, 5))
        # fig.subplots_adjust(left=0.12, right=0.97,
        #                     bottom=0.21, top=0.9, wspace=0.5)


        # plot 1: data + best-fit mixture
        ax = fig.add_subplot(121)

        ax.plot(x[:-1], p, 'r') # approximation
        ax.plot(x, pdf, '-k')
        ax.plot(x, pdf_individual, '--k')
        ax.text(0.04, 0.96, f'Best-fit Mixture n={n_best+1}',
                ha='left', va='top', transform=ax.transAxes)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$p(x)$')


        # plot 2: AIC and BIC
        ax = fig.add_subplot(122)
        ax.plot(N, AIC, '-k', label='AIC')
        # ax.plot(N, BIC, '--k', label='BIC')
        ax.set_xlabel('n. components')
        ax.set_ylabel('information criterion')
        ax.legend(loc=2)

        plt.show()

    return n_best+1

#----------------------------------------------------------------------------------
# Processing helper remaining from legacy
#----------------------------------------------------------------------------------

def compute_PCA(
        data: np.ndarray
    ):
    
    scaler = StandardScaler()
    pca = PCA(0.7, random_state=33)
    pca.fit(scaler.fit_transform(data))
    
    Xt = pca.inverse_transform(
        pca.transform(
            scaler.transform(data)
        ))


def dbscan_anomaly_detection(data):

    ### DBSCAN ANOMALY DETECTION ###

    network_ano = {}
    dbscan = DBSCAN(eps=0.6, min_samples=1, metric="precomputed")

    for trial_idx in range(data.shape[0]+1):
    
        trial = data[trial_idx, :,:].squeeze()
        preds = dbscan.fit_predict(
            pairwise_distances(trial, metric='correlation')
        )
        if (preds > 0).any():
            ano_features = list(X.columns[np.where(preds > 0)[0]])
            network_ano[past_id[-1]] = ano_features
        else:
            network_ano[past_id[-1]] = None

        

def photometry2xarray(data_photometry, skip_var=None):
    """
    Converts a pyphotometry dictionary into an xarray dataset. 
    
    Parameters
    ----------
    data_photometry : dict
        A pyphotometry dictionary containing data and associated time stamps.
    skip_var: list
        name of keyword in the data_photometry dict that you want to skip, mainly use to skip intermeidate variables
        
    Returns
    -------
    dataset : xarray.Dataset
        An xarray Dataset containing data and attributes associated with the 
        pyphotometry dictionary. 
    """ 
    
    data_list = {}
    attr_list = {}
    time = data_photometry['time'].astype(np.int64)
    
    if skip_var is None:
        skip_var = []

    for k, data in data_photometry.items():
        if not k in skip_var:
            
            if isinstance(data, (list,np.ndarray)) and len(data) == len(time):
                    array = xr.DataArray(data, coords={'time':time}, dims=['time'])
                    data_list[k] = array
            else:
                attr_list[k] = data

    dataset = xr.Dataset(data_list)
    dataset.attrs.update(attr_list)
    
    return dataset


def align_photometry_to_pycontrol(xr_photometry, df_event, pycontrol_aligner):
    # align the time coordinate
    new_time = pycontrol_aligner.A_to_B(xr_photometry.time)
    xr_photometry['time'] = new_time
    xr_photometry = xr_photometry.sel(time=xr_photometry.time.notnull()) #don't know why but dropna doesn't work here

    # interpolate and add in the trial_nb
    f = interp1d(df_event.time, df_event.trial_nb, kind = 'previous', 
            bounds_error=False, fill_value=-1)

    trial = f(xr_photometry.time)
    trial_xr = xr.DataArray(
        trial.astype(np.int16), coords={'time':xr_photometry.time}, dims=('time')
    )

    xr_photometry['trial'] = trial_xr
    
    return xr_photometry
        

def resample_event(aligner, ref_time, event_time, event_value, fill_value=-1):
    """
    Resample an event to a reference time.

    Parameters
    ----------
    aligner : object
        An instance of the Rsync_aligner class.
    ref_time : array-like
        Reference time points.
    event_time : array-like
        Event time points to align to the ref_time.
    event_value : array-like
        Event values corresponding to the event time points.
    fill_value : float, optional
        Value used to fill in for requested points outside of the range of event_time. The default is -1.

    Returns
    -------
    array-like
        Resampled event values corresponding to the reference time points.
        can contain NaN value if there is no overlap data
    """
    
    new_time = aligner.A_to_B(event_time)
    f = interp1d(new_time, event_value, kind = 'previous', 
                bounds_error=False, fill_value=fill_value)
    
    return f(ref_time)


def extract_event_data(trigger_timestamp, window, dataArray, sampling_rate, 
                       data_len =None, time_tolerance=10):
    '''
    Extract continous data around a timestamp. The original timestamp will be
    aligned to the coordinate of the dataArray with aligner
    
    Parameters:
        trigger_timestamp : float or int
            Timestamp around which data has to be extracted, in ms
        window : tuple
            Tuple containing minimum and maximum value of time window for which data is to be extracted
        aligner : object
            Object containing A_to_B() method for alignment of timestamp
        dataArray : array-like
            Array containing data from which required data is to be extracted. It it assumed to have a time coordinate in ms
        data_len : int, optional
            If provided, checks for length of output data
        time_tolerance: int, default=5
            the minimum time difference in ms that must be matched between the trigger stampstamp and the time coordinate of the dataArray
    Returns:
        data : numpy.ndarray
            Array of continuous data around the timestamp
        event_found : list
            List of boolean values indicating if any event was found around the given timestamp

    '''

    
    ts = trigger_timestamp
    ref_time = dataArray.time
    data = []
    event_found = []
    
    for t in ts:
        if t is not None:
            d = abs((ref_time-t).data)
            #Find the most close matched time stamp and extend it both ends 
            min_idx = np.argmin(d)
            min_time = d[min_idx]
            start_idx = min_idx +int(window[0]/1000*sampling_rate)
            end_idx = min_idx + int(window[1]/1000*sampling_rate)
            
            if min_time < time_tolerance and (start_idx>0) and (end_idx< len(dataArray.time)):
                min_idx = np.argmin(d)
                if dataArray.data.ndim == 1:
                    data.append(dataArray.data[start_idx:end_idx])
                else:
                    #TODO: work on the case for multi-dimensional data
                    data.append(dataArray.data[:,start_idx:end_idx])   
                event_found.append(True)
            else:
                # logger.debug(f'Warning: not enough data found {start_idx}:{end_idx}')
                x = np.zeros((int((window[1]-window[0])/1000*sampling_rate),))*np.nan
                data.append([x])
                event_found.append(False)
        else:
            x = np.zeros((int((window[1]-window[0])/1000*sampling_rate),))*np.nan
            data.append([x])
            event_found.append(False)
        
    # align to the longest element
    if len(data)>0:
        data  = np.vstack(data).astype(float)
    else:
        data = np.empty((0,int((window[1]-window[0])/1000*sampling_rate)))
    
    # if data_len is provide, perform additional check or correct the data length
    if data_len is not None:
        if not data.shape[0]==data_len:
            data = data[:data_len,:]
    
    return data,event_found #only float support NA

#%% Calulate the relative time
def get_rel_time(trigger_timestamp, window, aligner, ref_time):
    # Calculate the time relative to a trigger timestamp)
    ts = aligner.A_to_B(trigger_timestamp)
    time_relative = np.ones_like(ref_time)*np.nan
    
    for t in ts: 
        d = ref_time-t
        idx = (d>window[0]) & (d<window[1])
        time_relative[idx] = d[idx]
        
    return time_relative


def bin_rel_time(xr_dataset, bin_size):
    """Bins relative time in the input Xarray dataset to the given bin size.
    
    
    we need to do some special treatment to the relative time because the time stamp for that may not fall in the 
    same time bin, and hence the mean value of them will be different for different trial
    this will create problem with plotting and analysis later, so we need to fix it now

    Args:
        xr_dataset (xarray.core.dataset.Dataset): The input Xarray dataset.
        bin_size (float): The size of each bin.

    Returns:
        xarray.core.dataset.Dataset: The binned Xarray dataset.
    """
    for k in xr_dataset.data_vars.keys():
        if 'rel_time' in k:
            xr_dataset[k] = np.round(xr_dataset[k]/bin_size)*bin_size
            
    return xr_dataset


def bin_dataset(xr_dataset, bin_size, sampling_fs=1000):
    """
    Bin the input xarray dataset by grouping data within specified time intervals.
    
    Args:
    xr_dataset (xarray.Dataset): Input xarray dataset to be binned
    time_bin (float): Width of each time bin for grouping data, in ms
    
    Returns:
    dataset_binned (xarray.Dataset): Binned xarray dataset
    """
    
    ds_factor = int((bin_size/1000)*sampling_fs)
    logging.debug(f'Downsampling testby {ds_factor}')

    dataset_binned = xr_dataset.coarsen(time=ds_factor, boundary='trim').mean()

    dataset_binned = bin_rel_time(dataset_binned, bin_size)
    dataset_binned.attrs.update(xr_dataset.attrs)
    
    return dataset_binned


def make_condition_xarray(df_condition, dataset_binned):
    """
    Merge condition for each trial to the xarray
    
    Args:
    -----------------------------
    df_condition : pd.DataFrame
        Dataframe containing trial specific variables (e.g. condition to be analyzed)
    dataset_binned : xr.dataset
        Xarray dataset that has already been binned
    
    Returns:
    -----------------------------
    xr_condition : xr.dataset
        Dataset with the condition merged in as a new dimension
    
    """
    
    df_trial_nb = dataset_binned.trial_nb.to_dataframe()
    df_trial_nb['trial_nb'] = df_trial_nb['trial_nb'].astype(np.int16)
    df_trial_condition = df_trial_nb.merge(df_condition, on='trial_nb')
    df_trial_condition['time'] = dataset_binned.time
    df_trial_condition = df_trial_condition.set_index('time')
    xr_condition = df_trial_condition.to_xarray()
    
    return xr_condition



def make_rel_time_xr(event_time, windows, pyphoto_aligner, ref_time):
    time_rel = get_rel_time(event_time, windows, pyphoto_aligner, ref_time)

    rel_time = xr.DataArray(
        time_rel, coords={'time':ref_time}, dims=('time')
    )
    
    return rel_time

def make_event_xr(event_time, trial_window,
                  event_time_coordinate,  dataArray, sampling_rate, group='trial_nb', dim_name=None):
    '''
    Create xarray.DataArray object for the continuous data around provided timestamp. 

    Parameters:
        event_time : array_like
            List of timestamps around which continuous data is to be extracted
            it is assumed to have a index corresponds to the trial number
        trial_window : tuple
            Tuple containing minimum and maximum value of time window for which data is to be extracted
        event_time_coordinate : array_like
            List of trial numbers corresponding to each event_time
        dataArray : array_like
            Array containing time and data information. note, the time coordinate of the dataArray
            should be the same as in event_time
            
    Returns:
        da : xarray.DataArray
            DataArray object containing continuous data around the provided timestamp
    
    Note:
        It returns only data from extract_event_data() function ignoring the event_found.
    '''
    data, _ = extract_event_data(event_time, trial_window, dataArray, sampling_rate)
    if group =='trial_nb':
        assert event_time.index.name =='trial_nb', 'event_time should have a trial_nb index'
        da = xr.DataArray(
            data, coords={'event_time':event_time_coordinate, 
                        'trial_nb':event_time.index.values},
                            dims=('trial_nb','event_time'))
    else:
        # do not care about the trial structure, just extract everything

        if dim_name is None:
            raise ValueError('If not using trial_nb as group, you need to specify the group name for the datarray coordiante')
        da = xr.DataArray(
            data, 
            coords={'event_time':event_time_coordinate, 
                        f'{dim_name}_idx':np.arange(len(event_time))},
            dims=(f'{dim_name}_idx','event_time')
        )
    return da

def add_event_data(df_event, filter_func, trial_window,
                   dataset, event_time_coordinate, data_var_name, 
                   event_name, sampling_rate, groupby_col='trial_nb',filter_func_kwargs={}):
    '''
    Add continuous data around provided timestamp to a dataset.

    Parameters:
        df_event : pandas.DataFrame
            DataFrame containing event and timestamp information
        filter_func : function
            Function to filter particular events from the dataframe
        trial_window : tuple
            Tuple containing minimum and maximum value of time window for which data is to be extracted.
        aligner : Object
            Object containing A_to_B() method for alignment of timestamp
        dataset : xarray.Dataset
            Dataset in which data is to be added
        event_time_coordinate : array_like
            relative time coordinate in ms of the continous data
        data_var_name : str
            The variable name of the data in `dataset`.
        event_name : str
            name of the event
        sampling_rate: int
            sampling rate of the continous signal
        filter_func_args: dict
            argument that should be passed to the filter function
        
    Returns:
        None

    Note:
        It updates the given `dataset` with a new variable that contains continuous data around 
        each timestamp.  It requires two supporting functions make_event_xr() & extract_event_time().
    '''

    event_time = extract_event_time(df_event, filter_func, filter_func_kwargs, groupby_col=groupby_col)
    xr_event_data = make_event_xr(event_time, trial_window,
                                            event_time_coordinate,
                                            dataset[data_var_name], sampling_rate,
                                            group=groupby_col,
                                            dim_name=event_name)
    dataset[f'{event_name}_{data_var_name}'] = xr_event_data
# %%
def fit_exp_baseline(curve, sampling_rate, smooth_window=4001):

    def fitFn(x,a,b,c):
        return a+(b*np.exp(-(1/c)*x))
    
    p0 = [1,5,5]
    t = np.arange(len(curve))/sampling_rate
    smooth_curve = signal.savgol_filter(curve,smooth_window,3)
    popt,pcov = curve_fit(fitFn, t, smooth_curve,p0)
    baseline = fitFn(t,*popt)

    return baseline

def fit_a2b(a,b):
    # fit array a to b with linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x=a, y=b)
    output = intercept + slope * a
    return output

def remove_outliner_mad(x,thres_factor):
    '''
    Remove the shadow artifact from the signal when the signal is above median+ median_abs_deviation*thres_factor
    replace the signal with the median value 
    For best result, the signal needs to be first high-pass filtered e.g. by 0.1Hz
    '''
    x = x.copy()
    median_org = np.median(x)
    
    absx = np.abs(x)
    median = np.median(absx)
    mda = median_abs_deviation(absx)
    
    idx2remove = (absx>(median+mda*thres_factor))
    
    x[idx2remove] = median_org
    
    return x, idx2remove
    
def motion_correction_multicolor(photometry_dict, motion_smooth_win=1001, baseline_method='lowpass'):
    # analog1:  GFP
    # analog2: RFP
    # analog3: isosbestic
    '''
    Analysis notes:
    When the isosbestic channel is very noisy, and shadow artifact appears in the signal, it is be very difficult to cancel it.
    When we stretch the isosbestic signal to match the shadow artifact in the RFP channel, the noise from the isosbestic channel will be 
    stretched too. Substracting this amplified noise from the RFP will totally mask the original signal.
    We can probably smooth the signal to avoid the noise but that it will affect the results when there are real motion-artifact
    TODO: develop some specified way to remove the shadow artifact, probably by setting a threshold and just remove that data
    '''
    sampling_rate = photometry_dict['sampling_rate']
    
    if any(['analog_1_filt' not in photometry_dict, 'analog_2_filt' not in photometry_dict]):
        raise Exception('Analog 1 and Analog 2 must be filtered before motion correction')

    try:
        # method 1
        # slope, intercept, r_value, p_value, std_err = linregress(x=photometry_dict['analog_3_filt'], y=photometry_dict['analog_1_filt'])
        # photometry_dict['analog_1_est_motion'] = intercept + slope * photometry_dict['analog_3_filt']
        # photometry_dict['analog_1_corrected'] = photometry_dict['analog_1_filt'] - photometry_dict['analog_1_est_motion']
        
        # method 2: correct motion after baseline removed from both analog1 and isosbestic
        isos_bleach_baseline = lowpass_baseline(photometry_dict['analog_3_filt'], sampling_rate, 0.005) # low freq to only remove the baseline but not the motion artifact
        analog_1_bleach_baseline = lowpass_baseline(photometry_dict['analog_1_filt'], sampling_rate, 0.005) # low freq to only remove the baseline but not the motion artifact
        analog_1_detrend = photometry_dict['analog_1_filt'] - analog_1_bleach_baseline
        isos_detrend = photometry_dict['analog_3_filt'] - isos_bleach_baseline
        
        photometry_dict['isos_bleach_baseline'] = isos_bleach_baseline
        photometry_dict['analog_1_bleach_baseline'] = analog_1_bleach_baseline
        photometry_dict['analog_1_detrend'] = analog_1_detrend
        photometry_dict['isos_detrend'] = isos_detrend
        
        # photometry_dict['isos_scaled'] = fit_a2b(isos_detrend, analog_1_detrend) # match the scale
        photometry_dict['analog_1_est_motion'] = lowpass_baseline(isos_detrend, sampling_rate, 5) # only subtract the motion
        photometry_dict['analog_1_corrected'] = photometry_dict['analog_1_detrend'] - photometry_dict['analog_1_est_motion']
        
        # method 3
        # GFP_baseline = lowpass_baseline(photometry_dict['analog_1_filt'],sampling_rate)
        # photometry_dict['analog_1_corrected']  = photometry_dict['analog_1_filt'] - GFP_baseline
        
        # For RFP, remove baseline
        if baseline_method == 'lowpass':
            RFP_baseline =  lowpass_baseline(photometry_dict['analog_2_filt'],sampling_rate)
            isos_baseline = lowpass_baseline(photometry_dict['analog_3_filt'],sampling_rate)
        else:
            RFP_baseline = fit_exp_baseline(photometry_dict['analog_2_filt'], sampling_rate)
            isos_baseline = fit_exp_baseline(photometry_dict['analog_3_filt'],sampling_rate)
            
        photometry_dict['analog_2_detrended']  = photometry_dict['analog_2_filt'] - RFP_baseline
        photometry_dict['analog_3_detrended'] = photometry_dict['analog_3_filt'] - isos_baseline
        
        # we need to remove the shadow artifact first
        photometry_dict['analog_2_deshadow'],_ = remove_outliner_mad(photometry_dict['analog_2_detrended'],30)
        photometry_dict['analog_3_deshadow'],_ = remove_outliner_mad(photometry_dict['analog_3_detrended'],30)

        
        analog_2_est_motion, analog_2_corrected = window_subtraction(photometry_dict['analog_2_deshadow'],
                                                                     photometry_dict['analog_3_deshadow'],
                                                                     sampling_rate)
        
        photometry_dict['analog_2_est_motion'] = analog_2_est_motion
        photometry_dict['analog_2_corrected'] = analog_2_corrected

        photometry_dict['motion_corrected'] = 1
        
        return photometry_dict
    
    except ValueError as e:
        print(e)
        print('Motion correction failed. Skipping motion correction')
        # probably due to saturation , do not do motion correction
        photometry_dict['analog_1_corrected'] = photometry_dict['analog_1_filt']
        photometry_dict['motion_corrected'] = 0

    return photometry_dict

def lowpass_baseline(curve, sampling_rate, corner_freq = 0.02):
    # use an aggressive lowpass filter to find the baseline
    b,a = get_filt_coefs(low_pass=corner_freq, sampling_rate=sampling_rate)
    return signal.filtfilt(b,a,curve)

def baseline_correction_multicolor(photometry_dict,baseline_method='lowpass'):
    sampling_rate = photometry_dict['sampling_rate']
    if baseline_method =='lowpass':
        RFP_baseline = lowpass_baseline(photometry_dict['analog_2_filt'],sampling_rate)
        GFP_baseline = lowpass_baseline(photometry_dict['analog_1_filt'],sampling_rate)
    else:
        RFP_baseline =  fit_exp_baseline(photometry_dict['analog_2_filt'],sampling_rate, sampling_rate*2)
        GFP_baseline =  fit_exp_baseline(photometry_dict['analog_1_filt'],sampling_rate, sampling_rate*2)

    photometry_dict['analog_2_corrected']  = photometry_dict['analog_2_filt'] - RFP_baseline
    photometry_dict['analog_1_corrected']  = photometry_dict['analog_1_filt'] - GFP_baseline
    
    return photometry_dict

    
def preprocess_photometry(data_photometry, df_pycontrol):
    # Perform preprocessing on the photometry data e.g. bleach and motion correcction etc.
    data_photometry = denoise_filter(data_photometry, 20)
    
    # determine how to do motion correction
    animal_info = pd.read_csv('params/animal_info.csv',index_col='animal_id')
    if 'subject_id' in df_pycontrol.attrs:
        animal_id = df_pycontrol.attrs['subject_id'] 
    else:
        animal_id = df_pycontrol.attrs['Subject ID'] 
    
    animal_id = animal_id.replace('-','') #clean up the animal name
        
    if animal_id in animal_info.index:
        injection = animal_info.loc[animal_id].injection.strip().split(';')
        if len(injection) == 1:
            #single injection, skip all motion correction for now
            # TODO do motion correction for tdtomato
            data_photometry['analog_1_corrected'] = data_photometry['analog_1_filt']
            data_photometry['analog_2_corrected'] = data_photometry['analog_2_filt']
            data_photometry['motion_corrected'] = 0

        else:
            if set(['Rdlight', 'rDA','tdTomato']) & set(injection):
                logger.debug('Processing multicolor photometry')
                if not 'analog_3' in data_photometry:
                    baseline_correction_multicolor(data_photometry)
                    data_photometry['motion_corrected'] = 1

                else:
                    # Do multicolor correction
                    data_photometry = motion_correction_multicolor(data_photometry)
            else:    
                data_photometry = motion_correction_win(data_photometry)
    else:
        # Canont find animal, emit error and stop
        raise ValueError(f'Error: cannot find animal info for {animal_id}. Aborting.') 
    
    
    data_photometry = compute_df_over_f(data_photometry, low_pass_cutoff=0.01)
    data_photometry = compute_zscore(data_photometry)
    return data_photometry
    
