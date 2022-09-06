# Utility functions for pycontrol and pyphotometry files processing

import shutil
import json

from os.path import join
from os import walk
from datetime import datetime, date
from re import search

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, medfilt
from pandas import Timestamp
from rsync import *

#----------------------------------------------------------------------------------
# Plotting
#----------------------------------------------------------------------------------

def plot_longitudinal(results, plot_individuals=True):
    fontsize = 12
    condition_IDs = results['condition_ID'].unique()
    metric_names = results['metric_name'].unique()
    group_IDs = results['group_ID'].unique()

    fig, axs = plt.subplots(len(metric_names), len(condition_IDs), sharex= 'all', sharey = 'row', squeeze=False, figsize=(10,10))
    
    mean_df = results.groupby(['metric_name','condition_ID', 'group_ID', 'session_ID']).mean()
    sem_df = results.groupby(['metric_name','condition_ID', 'group_ID', 'session_ID']).sem()

    for col, cond_ID in enumerate(condition_IDs):
        for row, metric_name in enumerate(metric_names):
            
            mean_df = results.groupby(['metric_name','condition_ID', 'group_ID', 'session_ID']).mean()
            sem_df = results.groupby(['metric_name','condition_ID', 'group_ID', 'session_ID']).sem()
            
            for group_ID in group_IDs:

                subset_mean = mean_df.loc[(metric_name,cond_ID,group_ID)]
                subset_sem = sem_df.loc[(metric_name,cond_ID,group_ID)]

                axs[row, col].plot(subset_mean.index.values, subset_mean.metric.values)
                axs[row, col].fill_between(subset_mean.index.values, subset_mean.metric.values - subset_sem.metric.values,
                    subset_mean.metric.values + subset_sem.metric.values, color='gray', alpha=0.5)

                axs[row, col].set_xlabel('session nb', fontsize=fontsize)
                axs[row, col].set_ylabel(metric_name, fontsize=fontsize)
                axs[row, col].set_title('Title', fontsize=fontsize)

#----------------------------------------------------------------------------------
# Helpers
#----------------------------------------------------------------------------------

def find_matching_files(subject_ID, datetime_to_match, files_df, ext):
    '''
    Helper function for match_sessions_to_files, find files with
    the same subject_ID in the filename, and taken the same day
    as the pycontrol session, return the file(s) with the shortest
    timedelta compared to the start of the session.
    
            Parameters:
                    subject_ID (int): from session.subject_ID (need to be converted
                        from string to int if int_subject_IDs=False at Session object creation)
                    datetime_to_match (datetime): from session.datetime
                    files_df (pd.Dataframe): Created from a list of files in 
                        match_sessions_to_files function
                    ext (str): extension used to filter files within a folder
                        do not include the dot. e.g.: "mp4"

            Returns:
                    match_df (pd.Dataframe): containing filenames of matching files
    ''' 

    if ext not in ['nwb','h5']:
        # for videos, avoid integrating DeepLabCut labelled videos
        match_df = files_df[(files_df['datetime'].apply(lambda x: Timestamp.date(x)) == datetime_to_match.date()) &
            (files_df['filename'].str.contains(str(subject_ID))) &
            ~(files_df['filename'].str.contains('DLC'))]

        # will not avoid DLC-containing filenames in case of searching DLC .nwb data files
    else:
        match_df = files_df[(files_df['datetime'].apply(lambda x: Timestamp.date(x)) == datetime_to_match.date()) &
             (files_df['filename'].str.contains(str(subject_ID)))]

    
    # match_df = match_df.to_frame(name='matching_filename')
    if ~match_df.empty:
      
        match_df['timedelta'] = match_df['datetime'].apply(lambda x: abs(datetime_to_match-x))
        match_df = match_df[match_df['timedelta'] == match_df['timedelta'].min()]
        #print(match_df['timedelta'])
        match_df['timedelta'] = match_df['timedelta'].apply(lambda x: x.seconds)
    
    return match_df
    
def get_datetime_from_datestr(datestr):
    # here is the order and format of Year(2 or 4 digits)-months-day and time which matters.
    # All possibilities present in a folder should be listed
    # for an exhaustive list of possibilities see:
    # https://www.programiz.com/python-programming/datetime/strptime 
    #     
    date_patterns = ["%Y-%m-%d_%H-%M-%S", "%y-%m-%d_%H-%M-%S", "%m-%d-%y_%H-%M-%S", "%Y-%m-%d-%H%M%S"]

    for pattern in date_patterns:
        try:
            datetime_match = datetime.strptime(datestr, pattern)
            #print(s_date,datetime_match)
            return datetime_match
        except:
            #print(s_date,'exception')
            continue

def get_datestr_from_filename(filename):
    
    # list all the possible decimal format for date strings
    # here the order of year month and date doesn't matter
    # datestring will be later processed as a datetime
    re_patterns = ['\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', '\d{2}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}']
    for idx, redate in enumerate(re_patterns):
        # print(idx)
        match_str = search(redate, filename)
        if match_str:
            continue
    
    datestr = match_str.group(0)
    return datestr


def find_if_event_within_timelim(df_item, timelim):
    if isinstance(df_item,list):
        within_lim = any(ele >= timelim[0] and ele <= timelim[1] for ele in df_item)
    else:
        within_lim = False

    return within_lim

def time_delta_by_row(df_events_row, col_idx_start, col_idx_end):
    #print(df_events_row)
    start_time = min([i for i in df_events_row[col_idx_start] if i > 0], default=np.NaN)
    if isinstance(start_time, float):
        return np.NaN
    else:
        end_time = min([i for i in df_events_row[col_idx_end] if i > start_time], default=np.NaN)
        if isinstance(end_time, int):
            return end_time - start_time
        else:
            return np.NaN

def normalize_coords(coord_dict, normalize_betwen=['Left_paw','spout'], bins_nb=200):
    '''
    Get the coordinates of maximum density of two regions in order to normalize trajectories.
    Only for 2D for now.
    coord_dict is a dictionary which keys are regions computed, and values are X-Y ndarrays
    return the coordinates normalized between the coords of max density of two regions.
    normalize_betwen is a 2 items list which state the start and stop region
    to normalize between.
    bins_nb is the number of bins used to compute the np.histogram2d functions.
    The trade-off for bins_nb: too high value will only have a few timestamps
    in a bin, leading to poor aggregation and then random-ish maximum coord.
    Values too low will lead to a good aggregation but much less pixel-wise
    precision.
    Used by session.get_deeplabcut_trials()
    '''
    if len(normalize_betwen) != 2:
        raise Exception('normalize_betwen must be a list of two regions (str)')
    
    min_max_coord = np.ndarray((2,2))
    for idx_r, region in enumerate(normalize_betwen):
        nan_free_coords = np.delete(coord_dict[region], np.isnan(coord_dict[region][:,0]),0)
        xmin = nan_free_coords[:,0].min()
        xmax = nan_free_coords[:,0].max()
        ymin = nan_free_coords[:,1].min()
        ymax = nan_free_coords[:,1].max()

        H, xedges, yedges = np.histogram2d(coord_dict[region][:,0],coord_dict[region][:,1], 
            bins=bins_nb , range=[[xmin, xmax], [ymin, ymax]])

        ind = np.unravel_index(np.argmax(H, axis=None), H.shape)
        min_max_coord[idx_r,:] = [xedges[ind[0]],yedges[ind[1]]]

    rangeXY = [min_max_coord[1,0] - min_max_coord[0,0], min_max_coord[1,1] - min_max_coord[0,1]]

    norm_coord_dict = dict()
    for region in coord_dict.keys():
        norm_coord_dict[region] = np.ndarray(shape=coord_dict[region].shape)
        norm_coord_dict[region][:,0] = (coord_dict[region][:,0]-min_max_coord[0,0]) / rangeXY[0]
        norm_coord_dict[region][:,1] = (coord_dict[region][:,1]-min_max_coord[0,1]) / rangeXY[1]

    return norm_coord_dict

#----------------------------------------------------------------------------------
# Data reorganization
#----------------------------------------------------------------------------------

def copy_files_to_horizontal_folders(root_folders, horizontal_folder_pycontrol, horizontal_folder_photometry):
    '''
    Browse sub-folders (in a single root folder or within a list of root folder)
    and copy them in a separate horizontal folders (no subfolders). The main
    purpose is for easier match between pycontrol and photometry files
    '''
    
    if isinstance(root_folders, str):
        root_folders = [root_folders]

    for root in root_folders:
        for path, subdirs, files in walk(root):
            for name in files:
                if name[-4:] == '.txt':
                    print(join(path, name))
                    shutil.copyfile(join(path, name),join(horizontal_folder_pycontrol, name))
                elif name[-4:] == '.ppd':
                    shutil.copyfile(join(path, name),join(horizontal_folder_photometry, name))

#----------------------------------------------------------------------------------
# Load analog data
#----------------------------------------------------------------------------------

def load_analog_data(file_path):
    '''Load a pyControl analog data file and return the contents as a numpy array
    whose first column is timestamps (ms) and second data values.'''
    with open(file_path, 'rb') as f:
        return np.fromfile(f, dtype='<i').reshape(-1,2)

def import_ppd(file_path, low_pass=20, high_pass=0.01, median_filt=None):
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
    # Alternating samples are signals 1 and 2.
    analog_1 = analog[ ::2] * volts_per_division[0]
    analog_2 = analog[1::2] * volts_per_division[1]
    digital_1 = digital[ ::2]
    digital_2 = digital[1::2]
    time = np.arange(analog_1.shape[0])*1000/sampling_rate # Time relative to start of recording (ms).
    # Filter signals with specified high and low pass frequencies (Hz).
    if low_pass and high_pass:
        b, a = butter(2, np.array([high_pass, low_pass])/(0.5*sampling_rate), 'bandpass')
    elif low_pass:
        b, a = butter(2, low_pass/(0.5*sampling_rate), 'low')
    elif high_pass:
        b, a = butter(2, high_pass/(0.5*sampling_rate), 'high')
      
    if low_pass or high_pass:
        if median_filt:
            analog_1_filt = medfilt(analog_1, kernel_size=median_filt)
            analog_2_filt = medfilt(analog_2, kernel_size=median_filt)    
            analog_1_filt = filtfilt(b, a, analog_1_filt)
            analog_2_filt = filtfilt(b, a, analog_2_filt)
        else:
            analog_1_filt = filtfilt(b, a, analog_1)
            analog_2_filt = filtfilt(b, a, analog_2)
    else:
        if median_filt:
            analog_1_filt = medfilt(analog_1, kernel_size=median_filt)
            analog_2_filt = medfilt(analog_2, kernel_size=median_filt)   
        else:
            analog_1_filt = analog_2_filt = None
    # Extract rising edges for digital inputs.
    pulse_inds_1 = 1+np.where(np.diff(digital_1) == 1)[0]
    pulse_inds_2 = 1+np.where(np.diff(digital_2) == 1)[0]
    pulse_times_1 = pulse_inds_1*1000/sampling_rate
    pulse_times_2 = pulse_inds_2*1000/sampling_rate
    # Return signals + header information as a dictionary.
    data_dict = {'analog_1'      : analog_1,
                 'analog_2'      : analog_2,
                 'analog_1_filt' : analog_1_filt,
                 'analog_2_filt' : analog_2_filt,
                 'digital_1'     : digital_1,
                 'digital_2'     : digital_2,
                 'pulse_inds_1'  : pulse_inds_1,
                 'pulse_inds_2'  : pulse_inds_2,
                 'pulse_times_1' : pulse_times_1,
                 'pulse_times_2' : pulse_times_2,
                 'time'          : time}
    data_dict.update(header_dict)
    return data_dict