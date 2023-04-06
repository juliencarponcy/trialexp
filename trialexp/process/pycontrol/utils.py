# Utility functions for pycontrol and pyphotometry files processing

import shutil
from collections import defaultdict, namedtuple
from datetime import datetime
from os import walk
from os.path import isfile, join
from re import search

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas import Timestamp
from plotly.subplots import make_subplots
from plotly.validators.scatter.marker import SymbolValidator

from trialexp.process.pycontrol.spike2_export import Spike2Exporter
from pathlib import Path

Event = namedtuple('Event', ['time','name'])
State = namedtuple('State', ['time','name'])

######## Analyzing event data

def parse_session_dataframe(df_session):
    # parse and format the session dataframe imported from pycontrol
    df_events = df_session[(df_session.type!='info')]
    info = df_session[df_session.type=='info']
    info = dict(zip(info.name, info.value))
    
    # error correction for some task name
    if 'pycontrol_share' in info['Task name']:
        #the full path is stored, only take the last bit
        # taskname = Path(info['Task name']).parts[-1]
        taskname = (info['Task name']).split('\\')[-1]
        info['Task name'] = taskname
    
    df_events = df_events.drop(columns='duration')
    df_events.attrs.update(info)

    return df_events

def print2event(df_events, conditions):
    
    df = df_events.copy()
    
    #Extract print event matched by conditions and turn them into events for later analysis
    idx = (df.type=='print') & (df.value.isin(conditions))
    df.loc[idx, 'name'] = df.loc[idx,'value'] 
    
    return df   

def parse_events(session, conditions):
    #parse the event and state information and return it as a dataframe

    #parse the events list to distinguish between state and event
    state_names = session.state_IDs.keys()
    events = session.events

    for i, evt in enumerate(events):
        if session.events[i].name in state_names:
            events[i] = State(evt.time, evt.name)

    #parse the print line and turn them into events
    print_evts = []
    for ln in session.print_lines:
        s = ln.split()
        # s[0] is the time, s[1] is the print statement
        time = s[0]
        if s[1:] in conditions:
            # treat print as another event
            print_evts.append(
                Event(int(s[0]), s[1:])) 

    # merge the print list and event list and sort them by timestamp
    all_events = events+print_evts
    all_events = sorted(all_events, key=lambda x:x.time)

    #convert events into a data frame
    # state change is regarded as a special event type
    evt_list = []
    last_state = None
    for evt in all_events:
        if type(evt) is State:
            last_state = evt.name
            event = {
               'state':last_state,
                'event_name': last_state,
                'time':evt.time,
            }
        else:
            event = {
                'state':last_state,
                  'event_name':evt.name,
                    'time':evt.time,
            }

        evt_list.append(event)


    df_events = pd.DataFrame(evt_list)

    # remove rsync
    df_events = df_events[df_events.event_name!='rsync'].copy()
    return df_events


def add_trial_number(df_events, trigger):
    # trigger is a tuple containing the state and event_name e.g. ('waiting_for_spout','state_change')
    # I really liked that
    df = df_events.copy()


    df['trial_number'] = 0

    df.loc[df.event_name==trigger, 'trial_number'] = 1
    df.trial_number = df.trial_number.cumsum()
    
    return df

def plot_session(df:pd.DataFrame, keys: list = None, state_def: list = None, print_expr: list = None, 
                    event_ms: list = None, export_smrx: bool = False, smrx_filename: str = None, verbose :bool = False,
                    print_to_text: bool = True):
        """
        Visualise a session using Plotly as a scrollable figure

        keys: list
            subset of self.times.keys() to be plotted as events
            Use [] to plot nothing

        state_def: dict, list, or None = None
            must be None (default)
            or dictionary of 
                'name' : str
                    Channel name
                'onset' : str 
                    key for onset 
                'offset' : str
                    key for offset
            or list of such dictionaries

            eg. dict(name='trial', onset='CS_Go', offset='refrac_period')
            eg. {'name':'trial', 'onset':'CS_Go', 'offset':'refrac_period'}

            For each onset, find the first offset event before the next onset 

        event_ms: list of dict
                'name':'name of something'
                'time_ms': X
            allow plotting timestamps as an event

        state_ms: list of dict #TODO

        verbose :bool = False


        """

        # see  \Users\phar0528\Anaconda3\envs\trialexp\Lib\site-packages\sonpy\MakeFile.py
        #NOTE cannot put file path in the pydoc block

        raw_symbols  = SymbolValidator().values
        symbols = [raw_symbols[i+2] for i in range(0, len(raw_symbols), 12)]
        # 40 symbols

        fig = go.Figure()
        if keys is None:
            keys = df.event_name.unique()
        else:
            for k in keys: 
               assert k in df.event_name.unique(), f"{k} is not found in self.time.keys()"
        
        
        if export_smrx:
            if smrx_filename is None:
                raise ValueError('You must specify the smrx_filename filename if you want to export file')
            else:
                spike2exporter = Spike2Exporter(smrx_filename, df.time.max()*1000, verbose)

        def find_states(state_def_dict: dict):
            """
            state_def: dict, list, or None = None
            must be None (default)
            or dictionary of 
                'name' : str
                    Channel name
                'onset' : str 
                    key for onset 
                'offset' : str
                    key for offset
            or list of such dictionaries

            eg. dict(name='trial', onset='CS_Go', offset='refrac_period')
            eg. {'name':'trial', 'onset':'CS_Go', 'offset':'refrac_period'}

            For each onset, find the first offset event before the next onset 
            """
            if state_def_dict is None:
                return None

            all_on_sec = df[(df.event_name == state_def_dict['onset'])].time.values
            all_off_sec = df[(df.event_name == state_def_dict['offset'])].time.values
            # print(all_on_sec)

            onsets_sec = [np.NaN] * len(all_on_sec)
            offsets_sec = [np.NaN] * len(all_on_sec)

            for i, this_onset in enumerate(all_on_sec):  # slow
                good_offset_list_ms = []
                for j, _ in enumerate(all_off_sec):
                    if i < len(all_on_sec)-1:
                        if all_on_sec[i] < all_off_sec[j] and all_off_sec[j] < all_on_sec[i+1]:
                            good_offset_list_ms.append(all_off_sec[j])
                    else:
                        if all_on_sec[i] < all_off_sec[j]:
                            good_offset_list_ms.append(all_off_sec[j])

                if len(good_offset_list_ms) > 0:
                    onsets_sec[i] = this_onset
                    offsets_sec[i] = good_offset_list_ms[0]
                else:
                    ...  # keep them as nan

            onsets_sec = [x for x in onsets_sec if not np.isnan(x)]  # remove nan
            offsets_sec = [x for x in offsets_sec if not np.isnan(x)]
            # print(onsets_sec)

            state_sec = map(list, zip(onsets_sec, offsets_sec,
                           [np.NaN] * len(onsets_sec)))
            # [onset1, offset1, NaN, onset2, offset2, NaN, ....]
            state_sec = [item for sublist in state_sec for item in sublist]
            # print(state_sec)

            return state_sec

        y_index = 0
        
        for kind, k in enumerate(keys):
            y_index += 1
            df_evt2plot = df[df.event_name==k]
            line1 = go.Scatter(x=df_evt2plot.time, y=[k]
                        * len(df_evt2plot), name=k, mode='markers', marker_symbol=symbols[y_index % 40])
            fig.add_trace(line1)
            
            if export_smrx:
                spike2exporter.write_event(df_evt2plot.time.values, k, y_index)
                
                


        if event_ms is not None:
            if isinstance(event_ms, dict):
                event_ms = [event_ms]
            
            for dct in event_ms:
                y_index += 1
                line3 = go.Scatter(
                    x=[t/1000 for t in dct['time_ms']],
                    y=[dct['name']] * len(dct['time_ms']),
                    name=dct['name'], mode='markers', marker_symbol=symbols[y_index % 40])
                fig.add_trace(line3)


        if state_def is not None:
            # Draw states as gapped lines
            # Assuming a list of lists of two names

            if isinstance(state_def, list):# multiple entry
                state_sec = None
                for state in state_def:
                    assert isinstance(state, dict)
                    
                    y_index +=1
                    state_sec = find_states(state)
                    
                    line1 = go.Scatter(x=[x for x in state_sec], y=[state['name']] * len(state_sec), 
                        name=state['name'], mode='lines', line=dict(width=5))
                    fig.add_trace(line1)
                    
                    if export_smrx:
                        spike2exporter.write_marker_for_state(state_sec, state['name'], y_index)

            else:
                state_sec = None
        else:
            state_sec = None
             

        fig.update_xaxes(title='Time (s)')
        fig.update_yaxes(fixedrange=True) # Fix the Y axis

        # fig.update_layout(
            
        #     title =dict(
        #         text = f"{self.task_name}, {self.subject_ID} #{self.number}, on {self.datetime_string} via {self.setup_ID}"
        #     )
        # )
        
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20) )

        fig.show()


def export_session(df:pd.DataFrame, keys: list = None, export_state=True, print_expr: list = None, 
                    event_ms: list = None, smrx_filename: str = None, verbose :bool = False,
                    print_to_text: bool = True):
        """
        Visualise a session using Plotly as a scrollable figure

        keys: list
            subset of self.times.keys() to be plotted as events
            Use [] to plot nothing

        state_def: dict, list, or None = None
            must be None (default)
            or dictionary of 
                'name' : str
                    Channel name
                'onset' : str 
                    key for onset 
                'offset' : str
                    key for offset
            or list of such dictionaries

            eg. dict(name='trial', onset='CS_Go', offset='refrac_period')
            eg. {'name':'trial', 'onset':'CS_Go', 'offset':'refrac_period'}

            For each onset, find the first offset event before the next onset 

        event_ms: list of dict
                'name':'name of something'
                'time_ms': X
            allow plotting timestamps as an event

        state_ms: list of dict #TODO

        verbose :bool = False


        """

        # see  \Users\phar0528\Anaconda3\envs\trialexp\Lib\site-packages\sonpy\MakeFile.py
        #NOTE cannot put file path in the pydoc block

        raw_symbols  = SymbolValidator().values
        symbols = [raw_symbols[i+2] for i in range(0, len(raw_symbols), 12)]
        # 40 symbols

        if keys is None:
            keys = df.name.unique()
        else:
            for k in keys: 
               assert k in df.name.unique(), f"{k} is not found in self.time.keys()"
        
        
        if smrx_filename is None:
            raise ValueError('You must specify the smrx_filename filename if you want to export file')
        else:
            spike2exporter = Spike2Exporter(smrx_filename, df.time.max(), verbose)
            
        
        def extract_states(df_pycontrol):
            # extract the onset and offset of state automatically
            df_states = df_pycontrol[df_pycontrol.type=='state']

            states_dict = defaultdict(list)

            #extract the starting and end point of stats
            if len(df_states)>2:
                curState  = df_states.iloc[0]['name']
                start_time = df_states.iloc[0]['time']
                
                for _, row in df_states.iloc[1:].iterrows():
                    if not row.name == curState:
                        states_dict[curState].extend([start_time, row.time])
                        start_time = row['time']
                        curState = row['name']
                        
            return states_dict  

        y_index = 0
        
        for kind, k in enumerate(keys):
            y_index += 1
            df_evt2plot = df[df.name==k]
            spike2exporter.write_event(df_evt2plot.time.values, k, y_index)

        if event_ms is not None:
            if isinstance(event_ms, dict):
                event_ms = [event_ms]
            
        if export_state:
            # Draw states as gapped lines
            state_dict = extract_states(df)
            
            for state, time_ms in state_dict.items():
                y_index += 1
                spike2exporter.write_marker_for_state(time_ms, state, y_index)




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
        match_df = files_df.loc[(files_df['datetime'].apply(lambda x: Timestamp.date(x)) == datetime_to_match.date()) &
            (files_df['filename'].str.contains(str(subject_ID))) &
            ~(files_df['filename'].str.contains('DLC'))].copy() 

        # will not avoid DLC-containing filenames in case of searching DLC .nwb data files
    else:
        match_df = files_df.loc[(files_df['datetime'].apply(lambda x: Timestamp.date(x)) == datetime_to_match.date()) &
                                (files_df['filename'].str.contains(str(subject_ID)))].copy() #TODO match_df is not a view or copy

    # match_df = match_df.to_frame(name='matching_filename')
    if ~match_df.empty:
      
        # A value is trying to be set on a copy of a slice from a DataFrame.
        # Try using .loc[row_indexer, col_indexer] = value instead
        match_df['timedelta'] = match_df['datetime'].apply(
            lambda x: abs(datetime_to_match-x))

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
    
    # Make sure the input are lists
    if not isinstance(list_ev, list):
        list_ev = [list_ev]
    
    if not isinstance(list_lim, list):
        list_lim =[list_lim]
    
    last_time = max([i for i in list_ev if i < find_min_time_list(list_lim)], default=np.NaN)

        
    return last_time


def find_min_time_list(x):
    
    if isinstance(x, list):
        if len(x) == 0:
            min_time = np.NaN
        else:
            min_time = min([i for i in x if i>0], default=np.NaN)
    else:
        min_time = x

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
