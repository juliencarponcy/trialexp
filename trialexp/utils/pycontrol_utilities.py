# Utility functions for pycontrol and pyphotometry files processing

import shutil

from os.path import join, isfile
from os import walk
from datetime import datetime
from re import search

import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
from pandas import Timestamp

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

def blank_spurious_detection(df_item, blank_timelim):
    '''
    Delete events within blank_timelim, which are suspected to be caused by
    unwanted contacts (e.g. drop to whisker), or artifacts in the detections.
    '''
    if isinstance(df_item, list):
        tlist = [t for t in df_item if (t < 0 or t > 60)]
        return tlist

def find_last_time_before_list(x):
    if len(x) >= 1:
        min_time = min([i for i in x if i>0], default=np.NaN)
    elif isinstance(x, int) and x > 0:
        min_time = x
    elif len(x) == 0:
        min_time = np.NaN
    else:
        print(x,type(x))
    return min_time


def find_min_time_list(x):
    if len(x) >= 1:
        min_time = min([i for i in x if i>0], default=np.NaN)
    elif isinstance(x, int) and x > 0:
        min_time = x
    elif len(x) == 0:
        min_time = np.NaN
    else:
        print(x,type(x))
    return min_time

def find_if_event_within_timelim(df_item, timelim):
    if isinstance(df_item, list):
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

def cmap10():
    """
    Default plot colors of matplotlib.pyplot.plot, turned into colormap
    """
    cmap = (mpl.colors.ListedColormap([u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
        u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'])
        ) # default 10 colors

    return cmap

def get_methods(obj):
    object_methods = [method_name for method_name in dir(obj)
                      if callable(getattr(obj, method_name))]
    print(object_methods)
    return object_methods

def get_attributes(obj):
    lst = list(obj.__dict__.keys())
    print(lst)
    return lst

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
                    if not isfile(join(horizontal_folder_pycontrol,name)):
                        print(join(path, name))
                        shutil.copyfile(join(path, name),join(horizontal_folder_pycontrol, name))
                elif name[-4:] == '.ppd':
                    if not isfile(join(horizontal_folder_photometry, name)):
                        print(join(path, name))
                        shutil.copyfile(join(path, name),join(horizontal_folder_photometry, name))

#----------------------------------------------------------------------------------
# Load analog data
#----------------------------------------------------------------------------------

def load_analog_data(file_path):
    '''Load a pyControl analog data file and return the contents as a numpy array
    whose first column is timestamps (ms) and second data values.'''
    with open(file_path, 'rb') as f:
        return np.fromfile(f, dtype='<i').reshape(-1,2)
