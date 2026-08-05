'''
These utilities contained originally much more methods, which have been moved
to more appropriate modules during refactoring.

It still contains mostly date matching patterns methods and helper methods
to find events timestamps for differential trial alignment.

It is possibly best to store the remaining methods in appropriate modules
'''
from pathlib import Path
import shutil

from tqdm import tqdm
import numpy as np
import warnings
from datetime import datetime, timedelta
from re import search
import pandas as pd
from trialexp.process.pycontrol.data_import import session_dataframe
from trialexp.process.pycontrol.utils import parse_session_dataframe
from trialexp.process.pyphotometry.utils import import_ppd_auto
from trialexp.utils.ephys_utilities import create_ephys_rsync, get_recordings_properties
from trialexp.utils.pyphotometry_utilities import create_photo_sync
from loguru import logger



def parse_pycontrol_fn(fn):
    pattern = r'(\w+)-(.*)\.[txt|tsv]'
    m = search(pattern, fn.name)
    
    if m:
        subject_id = m.group(1)
        date_string = m.group(2)
        expt_datetime = datetime.strptime(date_string, "%Y-%m-%d-%H%M%S")
        
        try:
            df = session_dataframe(fn) #note: this may run into error
            df = parse_session_dataframe(df)

            session_length = df.time.iloc[-1]
            task_name = df.attrs['task_name']
            
            return { 'subject_id': subject_id,
                    'path': fn,                 
                    'session_id': fn.stem,
                    'filename': fn.stem, 
                    'timestamp': expt_datetime,
                    'session_length': session_length,
                    'task_name': task_name}
        except Exception as e:
            print(f'Error while import file {fn}')
            print(e)
            return { 'subject_id': subject_id,
                    'path': fn,                 
                    'session_id': fn.stem,
                    'filename': fn.stem, 
                    'timestamp': expt_datetime,
                    'session_length': 0,
                    'task_name': 'unknown'}
    else:
        print('Error for ', fn)
'''
following is depracted until possible re-use elsewhere
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

'''
    
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
        Should probably be refactored with strftime

        PyControl and other files like videos and DLC data may
        have different date formats in their filenames.
        The purpose of this function is to reconcile these differences
        
        list all the possible decimal format for date strings
        here the order of year month and date doesn't matter
        datestring will be later processed as a datetime
        
        Add more patterns as needed

    '''

    re_patterns = [
        '\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', 
        '\d{2}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}',
        '\d{4}-\d{2}-\d{2}-\d{6}' # ppd files date format
        ]
    
    # loop over all the date patterns to find a match
    for re_date in re_patterns:
        # print(idx)
        match_str = search(re_date, filename)
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
        last_time = max([i for i in list_ev if i < find_min_time_list(list_lim)], default=np.nan)
    
    # TODO check implementation for limit cases (when no lim events found)
    elif isinstance(list_ev, int) and find_min_time_list(list_lim) is not np.nan:
        
        if find_min_time_list(list_lim) > list_ev:
            last_time = list_ev
        else:
            last_time = np.nan

    elif len(list_ev) == 0 or len(list_lim) == 0:
        last_time = np.nan
    else:
        print(list_ev,type(list_ev))
    return last_time


def find_min_time_list(x):
    if len(x) >= 1:
        min_time = min([i for i in x if i>0], default=np.nan)
    elif isinstance(x, int) and x > 0:
        min_time = x
    elif len(x) == 0:
        min_time = np.nan
    else:
        print(x,type(x))

    return min_time

def find_max_time_list(x):
    if len(x) >= 1:
        max_time = max([i for i in x if i>0], default=np.nan)
    elif isinstance(x, int) and x > 0:
        max_time = x
    elif len(x) == 0:
        max_time = np.nan
    else:
        print(x,type(x))

    return max_time

# Make use of timelim former param for success definition (if event found within specified limits), 
# expected to possibly be legacy/deprecated method. But could be useful for some cases?
def find_if_event_within_timelim(df_item, timelim):
    if isinstance(df_item, list):
        within_lim = any(ele >= timelim[0] and ele <= timelim[1] for ele in df_item)
    else:
        within_lim = False

    return within_lim

def time_delta_by_row(df_events_row, col_idx_start, col_idx_end):
    #print(df_events_row)
    start_time = min([i for i in df_events_row[col_idx_start] if i > 0], default=np.nan)
    if isinstance(start_time, float):
        return np.nan
    else:
        end_time = min([i for i in df_events_row[col_idx_end] if i > start_time], default=np.nan)
        if isinstance(end_time, int):
            return end_time - start_time
        else:
            return np.nan

'''
# Kouichi's helper, appears not to be used
def cmap10():
    """
    Default plot colors of matplotlib.pyplot.plot, turned into colormap
    """
    cmap = (mpl.colors.ListedColormap([u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
        u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'])
        ) # default 10 colors

    return cmap
'''
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
# Load analog data
#----------------------------------------------------------------------------------

def load_analog_data(file_path):
    '''
    Load a pyControl analog data file and return the contents as a numpy array
    whose first column is timestamps (ms) and second data values.
    
    This is to be used for example to import treadmill or piezzo sensor data 
    acquired by pyControl.
    '''
    with open(file_path, 'rb') as f:
        return np.fromfile(f, dtype='<i').reshape(-1,2)


def parse_video_fn(fn):
    # parse the filename of video file and extract its timestamp and direction
    pattern = r'(.+?)_([\w\d+]+)_Rig_(\d+)_(\w+).mp4'
    date_format = '%m-%d-%y_%H-%M-%S.%f'    
    
    m = search(pattern, fn.name)
    if m:
        date_string = m.group(1)
        subject_id = m.group(2)
        rig = m.group(3)
        camera = m.group(4)

        data = {
            'subject_id': subject_id,
                'path':fn, 
                'filename':fn.stem, 
                'rig' : int(rig),
                'camera': camera,
                'start_time': date_string
            }
        
        try:
            expt_datetime = datetime.strptime(date_string, date_format)
        
            data['timestamp']  = expt_datetime
        except ValueError:
            data['timestamp']  = None
    else:
        data = {
        'subject_id': None,
            'path':None, 
            'filename':None, 
            'rig' : None,
            'camera': None,
            'timestamp': None,
            'start_time': None
        } 
        
    return data

def get_df_video(video_folder):
    video_files = list(video_folder.glob('*.mp4'))
    df_video = pd.DataFrame(list(map(parse_video_fn, video_files)))
    df_video = df_video.dropna()
    return df_video

def get_matched_timestamp(df, df_pycontrol_row, camera_no=2, min_minute=3):
    df = df[df.subject_id == df_pycontrol_row.subject_id]
    # find the closet match in time
    if not df.empty:
        min_td = np.min(abs(df_pycontrol_row.timestamp - df.timestamp))
        idx = np.argmin(abs(df_pycontrol_row.timestamp - df.timestamp))
        if min_td < timedelta(minutes=min_minute):
            #Find videos from other cameras
            cameras =  df[df.iloc[idx].timestamp == df.timestamp]
            return cameras
    else:
        return None

def match_video(df_pycontrol, df_video):
    matched_video_names = []
    # Matching videos
    for _, row in df_pycontrol.iterrows():

        matched_videos = get_matched_timestamp(df_video,row)
        if matched_videos is not None:
            matched_video_names.append(matched_videos.filename.values)
        else:
            matched_video_names.append(None)
            
    df_pycontrol['video_names'] = matched_video_names

def copy_if_not_exist(src, dest):
    if not (dest/src.name).exists():
        shutil.copy(src, dest)
    
def move_folders(df_pycontrol, export_base_path, ephys_base_path):
    for i in tqdm(range(len(df_pycontrol))):
        try:
            row = df_pycontrol.iloc[i]
            session_id = row.session_id
            subject_id = row.subject_id
            task_name = row.task_name
            
            if '\\' in task_name:
                #skip copying for incorrect task name
                print(f'{task_name} is not a proper task name. Skipping')
                continue
            
            target_pycontrol_folder = Path(export_base_path,task_name, session_id, 'pycontrol')
            target_pyphoto_folder = Path(export_base_path, task_name, session_id, 'pyphotometry')
            target_ephys_folder = Path(export_base_path,  task_name, session_id, 'ephys')
            target_video_folder = Path(export_base_path, task_name, session_id, 'video')
            
            if not target_pycontrol_folder.exists():
                # create the base folder
                target_pycontrol_folder.mkdir(parents=True)
                
            if not target_pyphoto_folder.exists():
                target_pyphoto_folder.mkdir(parents=True)

            if not target_ephys_folder.exists():
                target_ephys_folder.mkdir(parents=True)
                
            if not target_video_folder.exists():
                target_video_folder.mkdir(parents=True)
                
            pycontrol_file = row.path
            pyphotometry_file = row.pyphoto_path
            video_files = row.video_names

            #copy the pycontrol files
            # print(pycontrol_file, target_pycontrol_folder)
            copy_if_not_exist(pycontrol_file, target_pycontrol_folder)
            
            #copy all the analog data
            analog_files = list(pycontrol_file.parent.glob(f'{session_id}*.pca')) + list(pycontrol_file.parent.glob(f'{session_id}*.npy'))
            for f in analog_files:
                copy_if_not_exist(f, target_pycontrol_folder) 
                
            #Copy pyphotometry file if they match
            if pyphotometry_file is not None:
                data_pycontrol = session_dataframe(pycontrol_file)
                data_pyphotmetry = import_ppd_auto(pyphotometry_file)
                if create_photo_sync(data_pycontrol, data_pyphotmetry) is not None:
                    copy_if_not_exist(pyphotometry_file, target_pyphoto_folder)
                else:
                    logger.debug(f'Cannot sync photometry data for {pyphotometry_file.name}')

                    
            # write down the filename of the video file
            video_list_file = target_video_folder/'video_list.txt'
            if row.video_names is not None and not video_list_file.exists():
                np.savetxt(video_list_file, row.video_names, '%s')


            #write information about ephys recrodings in the ephys folder
            if row.ephys_folder_name:

                recordings_properties = get_recordings_properties(ephys_base_path, row.ephys_folder_name)
                # try to sync ephys recordings
                recordings_properties['syncable'] = False
                recordings_properties['longest'] = False
                sync_paths = recordings_properties.sync_path.unique()
                sync_file_found = False
                for sync_path in sync_paths:
                    # copy syncing files in 
                    if create_ephys_rsync(str(pycontrol_file), sync_path) is not None:
                        recordings_properties.loc[recordings_properties.sync_path == sync_path, 'syncable'] = True
                        sync_file_found = True
                
                if not sync_file_found:    
                    print(f'Cannot find ephys data for {row.ephys_folder_name}')
                    
                longest_syncable = recordings_properties.loc[recordings_properties.syncable == True, 'duration'].max()
                recordings_properties.loc[(recordings_properties.duration == longest_syncable) & (recordings_properties.syncable == True), 'longest'] = True

                sync_path = recordings_properties.loc[recordings_properties.longest == True, 'sync_path'].unique()
                
                if len(sync_path) > 1:
                    raise NotImplementedError(f'multiple valids sync_path for the session, something went wrong: {row.ephys_folder_name}')
                
                # copy sync files from the longest syncable recording
                elif len(sync_path) == 1:

                    copy_if_not_exist(sync_path[0] / 'states.npy', target_ephys_folder)
                    copy_if_not_exist(sync_path[0] / 'timestamps.npy', target_ephys_folder)

                else:
                    # no syncable recordings
                    ...


                recordings_properties.to_csv(target_ephys_folder / 'rec_properties.csv')
        except Exception as e:
            print(f'Error copying files for {row.session_id}')
            print(e)