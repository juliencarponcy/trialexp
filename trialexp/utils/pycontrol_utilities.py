# Utility functions for pycontrol and pyphotometry files processing

import shutil

from os.path import join, isfile, isdir
from os import walk, listdir
from datetime import datetime
from re import search

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

#----------------------------------------------------------------------------------
# Plotting
#----------------------------------------------------------------------------------
# should move in a plotting module
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

def match_sessions_to_files(experiment, files_dir, ext='mp4', verbose=False) -> str:
    '''
    Take an experiment instance and look for files within a directory
    taken the same day as the session and containing the subject_ID,
    store the filename(s) with the shortest timedelta compared to the
    start of the session in exp.sessions[x].files["ext"] as a list
    
            Parameters:
                    file_name (str): name of the file to look for
                    files_dir (str): path of the directory to look into to find a match
                    ext (str): extension used to filter files within a folder
                        do not include the dot. e.g.: "mp4"

            Returns:
                    str (store list in sessions[x].file["ext"])
    ''' 

    # subject_IDs = [session.subject_ID for session in self.sessions]
    # datetimes = [session.datetime for session in self.sessions]
    files_list = [f for f in listdir(files_dir) if isfile(
        join(files_dir, f)) and ext in f]

    if len(files_list) == 0:
        raise Exception(f'No files with the .{ext} extension where found in the following folder: {files_dir}')

    files_df = pd.DataFrame(columns=['filename','datetime'])

    files_df['filename'] = pd.DataFrame(files_list)
    files_df['datetime'] = files_df['filename'].apply(lambda x: get_datetime_from_datestr(get_datestr_from_filename(x)))
    # print(files_df['datetime'])
    for s_idx, session in enumerate(experiment.sessions):
        match_df = find_matching_files(session.subject_ID, session.datetime, files_df, ext)
        if verbose:
            print(session.subject_ID, session.datetime, match_df['filename'].values)
        
        if not hasattr(experiment.sessions[s_idx], 'files'):
            experiment.sessions[s_idx].files = dict()
        
        experiment.sessions[s_idx].files[ext] = [join(files_dir, filepath) for filepath in match_df['filename'].to_list()]

def find_matching_files(subject_ID, datetime_to_match, files_df, ext):
    """
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
    """ 

    if ext not in ['nwb','h5']:
        # for videos, avoid integrating DeepLabCut labelled videos "['filename'].str.contains('DLC')"
        #TODO match_df is not a view or copy
        match_df = files_df.loc[(files_df['datetime'].apply(lambda x: pd.Timestamp.date(x)) == datetime_to_match.date()) &
            (files_df['filename'].str.contains(str(subject_ID))) &
            ~(files_df['filename'].str.contains('DLC'))].copy() 

        # will not avoid DLC-containing filenames in case of searching DLC .nwb data files
    else:
        match_df = files_df.loc[(files_df['datetime'].apply(lambda x: pd.Timestamp.date(x)) == datetime_to_match.date()) &
                                (files_df['filename'].str.contains(str(subject_ID)))].copy() #TODO match_df is not a view or copy

    # match_df = match_df.to_frame(name='matching_filename')
    if ~match_df.empty:
      
        # Compute time difference between the files
        match_df['timedelta'] = match_df['datetime'].apply(
            lambda x: abs(datetime_to_match-x))

        # Take the file with the minimum time difference
        match_df = match_df[match_df['timedelta'] == match_df['timedelta'].min()]
        #print(match_df['timedelta'])
        match_df['timedelta'] = match_df['timedelta'].apply(lambda x: x.seconds)
    
    return match_df
    
def get_datetime_from_datestr(datestr: str):
    '''
    here is the order and format of Year(2 or 4 digits)-months-day and time which matters.
    All possibilities present in a folder should be listed
    for an exhaustive list of possibilities see:
    https://www.programiz.com/python-programming/datetime/strptime 
    '''

    date_patterns = ["%Y-%m-%d_%H-%M-%S", "%y-%m-%d_%H-%M-%S", "%m-%d-%y_%H-%M-%S", "%Y-%m-%d-%H%M%S"]

    for pattern in date_patterns:
        try:
            datetime_match = datetime.strptime(datestr, pattern)
            #print(s_date,datetime_match)
            return datetime_match
        except:
            #print(s_date,'exception')
            continue

def get_datestr_from_filename(filename: str):
    '''   
        list all the possible decimal format for date strings
        here the order of year month and date doesn't matter
        datestring will be later processed as a datetime
        
        Add more patterns as needed

        Should be recoded with strftime possibly
    '''

    re_patterns = [
        '\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', 
        '\d{2}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}',
        '\d{4}-\d{2}-\d{2}-\d{6}' # ppd files date format
        ]
    
    # loop over all the date patterns to find a match
    for idx, redate in enumerate(re_patterns):
        # print(idx)
        match_str = search(redate, filename)
        if match_str:
            break
    
    # return datestr if match
    if match_str is not None:
        datestr = match_str.group(0)
    # or empty str if no match
    else:
        datestr = ''
    
    return datestr


def blank_spurious_detection(df_item, blank_timelim):
    '''
    Delete events within blank_timelim, which are suspected to be caused by
    unwanted contacts (e.g. drop to whisker), or artifacts in the detections.
    '''
    if isinstance(df_item, list):
        tlist = [t for t in df_item if (t < blank_timelim[0] or t > blank_timelim[1])]
        return tlist

def find_last_time_before_list(list_ev, list_lim):
    '''
    Utility function to use as apply to DataFrame in order
    to find the last event occuring before another event,
    like the last bar_off occrurign before a spout touch
    list_ev is the list of events to detect (contained in a cell of dataframe)
    list_lim is the list of events to use as limit
    '''
    if len(list_ev) >= 1 and len(list_lim) >= 1:
        last_time = max([i for i in list_ev if i < find_min_time_list(list_lim)], default=np.NaN)
    
    # TODO check implementation for limit cases (when no lim events found)
    elif isinstance(list_ev, int) and find_min_time_list(list_lim) is not np.NaN:
        
        if find_min_time_list(list_lim) > list_ev:
            last_time = list_ev
        else:
            last_time = np.NaN

    elif len(list_ev) == 0 or len(list_lim) == 0:
        last_time = np.NaN
    else:
        print(list_ev,type(list_ev))
    return last_time


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

def find_max_time_list(x):
    if len(x) >= 1:
        max_time = max([i for i in x if i>0], default=np.NaN)
    elif isinstance(x, int) and x > 0:
        max_time = x
    elif len(x) == 0:
        max_time = np.NaN
    else:
        print(x,type(x))

    return max_time

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
    """
    obj.__dict__
    vars(obj)
    These return attribute names and values.

    dir(obj)
    returns both attributes and methods

    See also:
    get_methods(get_attributes_and_properties)

    https://stackoverflow.com/questions/34439/finding-what-methods-a-python-object-has
    """

    spacing=20
    methodList = []
    for method_name in dir(object):
        try:
            if callable(getattr(object, method_name)):
                methodList.append(str(method_name))
        except Exception:
            methodList.append(str(method_name))
    processFunc = (lambda s: ' '.join(s.split())) or (lambda s: s)
    for method in methodList:
        try:
            print(str(method.ljust(spacing)) + ' ' +
                processFunc(str(getattr(object, method).__doc__)[0:90]))
        except Exception:
            print(method.ljust(spacing) + ' ' + ' getattr() failed')

        object_methods = [method_name for method_name in dir(obj)
                if callable(getattr(obj, method_name))]
        print(object_methods)
        return object_methods

def get_attributes_and_properties(obj):
    """
    obj.__dict__
    vars(obj)
    These return attribute names and values.

    dir(obj)
    returns both attributes and methods

    See also:
    get_methods(obj)
    """
    attrnames = list(obj.__dict__.keys())

    propnames = [p for p in dir(obj) if isinstance(getattr(obj, p), property)]

    print('Attributes:')
    print(attrnames)
    print('Properties:')
    print(propnames)

    return [attrnames, propnames]

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
