# Utility functions for pycontrol and pyphotometry files processing

import shutil
from collections import defaultdict, namedtuple
from datetime import datetime
from os import walk
from os.path import isfile, join
from re import search, match, DOTALL
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas import Timestamp
from plotly.subplots import make_subplots
from plotly.validators.scatter.marker import SymbolValidator
import warnings
import platform
from dotenv import load_dotenv
import os
from loguru import logger
import scipy
from scipy.stats import norm
import xarray as xr

Event = namedtuple('Event', ['time','name'])
State = namedtuple('State', ['time','name'])


def auto_load_dotenv(workspace_dir='env'):
    #automatically local the correct dotenv file based on the platform
    # load_dotenv will search for the .env file incrementally higher up
    
    if workspace_dir != '':
        workspace_dir += '/'
    
    if platform.system() == 'Windows':
        load_dotenv(workspace_dir+'windows.env')
    else:
        # determine whether we are in WSL
        try: 
            with open('/proc/version','r') as f:
                version_info = f.read().lower()
                if 'microsoft' in version_info or 'wsl' in version_info:
                    print('Loading config for wsl')
                    load_dotenv(workspace_dir+'wsl.env')
                    return
        except:
            pass                    
        
        # print(platform.node())
        if platform.node() == 'lapis':    
            load_dotenv(workspace_dir+'linux_lapis.env')
        else:
            # print('Loading linux .env')     
            load_dotenv(workspace_dir+'linux.env')
        

######## Analyzing event data

def parse_session_dataframe(df_session):
    # parse and format the session dataframe imported from pycontrol
    df_events = df_session[(df_session.type!='info')]
    info = df_session[df_session.type=='info']
    info = dict(zip(info.subtype, info.content))
    
    #correct for naming error in info
    if 'task_name' in info and 'pycontrol_share' in info['task_name']:
        task_name = info['task_name'].split('\\')[-1]
        info['task_name'] = task_name
                                            
    df_events.attrs.update(info)

    return df_events

def parse_trial_param(s):
    pattern = r'([a-zA-Z_ ]+):\s*([\d.]+|\w+)'
    d={}
    
    try:
        for m in re.finditer(pattern, s):
            d.update({m.group(1):m.group(2)})
    except TypeError:
        logger.debug(f'Cannot parse print line for {s}')
    return d

def print2event(df_events, conditions, trial_parameters):
    
    df = df_events.copy()
    extra_row_count = 0
    
    #Extract print event matched by conditions and turn them into events for later analysis
    for idx, row in df.iterrows():
        if row.type == 'print':
            # check if the value column match with condition
            # if so then assign the condition to the event column
            matched_con = [c for c in conditions if c == row.content]
            if len(matched_con)>0:
                if len(matched_con)>1:
                    warnings.warn(f'Warning: more than one conditionas found {matched_con}')  
                df.loc[idx,'name'] = matched_con[0]
                
            # Note: row and index are fixed during iterrows(), they will only be updated outside the
            # loop
            param_dict = parse_trial_param(row.content)
            param_dict = {k:v for k,v in param_dict.items() if k in trial_parameters}
            for k,v in param_dict.items():
                # add extract row to the dataframe per paramter
                df.loc[df.index[-1]+extra_row_count] = {
                    'type': 'trial_param',
                    'subtype': k,
                    'time':row.time,
                    'content': v
                }
                extra_row_count += 1
                
    
    df = df.sort_values('time').reset_index(drop=True) #make sure the order is correct
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

            onsets_sec = [np.nan] * len(all_on_sec)
            offsets_sec = [np.nan] * len(all_on_sec)

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
                           [np.nan] * len(onsets_sec)))
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

def extract_v_line_v2(df_pycontrol, print_char_limit=20):
    # for pyControl > v2
    
    # extract variable change
    df = df_pycontrol[df_pycontrol.type=='variable']

    v_list = []
    for _, row in df.iterrows():
        if type(row.content) is dict:
            for k,v in row.content.items():
                v_list.append(
                    {
                        'time':row.time,
                        'key': k,
                        'value': v,
                        'type': 'variable'
                    }
                )
    df_v = pd.DataFrame(v_list)
    if len(df_v)>0:
        df_v = df_v[~df_v.key.str.endswith('__')]

    # extract print lines
    df = df_pycontrol[df_pycontrol.type=='print']

    p_list = []
    for _, row in df.iterrows():
        if len(row.content)<print_char_limit:    
            # skip the long print line, they are prbably just status udpate
            p_list.append(
                {
                    'time':row.time,
                    'content': row.content,
                    'type': 'print'
                }
        )
    df_p = pd.DataFrame(p_list)
    return df_v, df_p

def extract_v_line(pycontrol_path):

    # # assuming just one txt file
    # pycontrol_txt = list(Path(sinput.pycontrol_folder).glob('*.txt'))

    with open(pycontrol_path, 'r') as f:
        all_lines = [line.strip() for line in f.readlines() if line.strip()]

    count = 0
    print_lines = []
    while count < len(all_lines):
        # all_lines[count][0] == 'P'
        if bool(match('P\s\d+\s', all_lines[count])):
            print_lines.append(all_lines[count][2:])
            count += 1
            while (count < len(all_lines)) and not (bool(match('[PVD]\s\d+\s', all_lines[count]))):
                print_lines[-1] = print_lines[-1] + \
                    "\n" + all_lines[count]
                count += 1
        else:
            count += 1

    v_lines = [line[2:] for line in all_lines if line[0] == 'V']
    
    return v_lines, print_lines


def export_session(df:pd.DataFrame, keys: list = None, export_state=True, print_expr: list = None, 
                    event_ms: list = None, smrx_filename: str = None, verbose :bool = False,
                    print_lines : list = None,
                    v_lines : list = None,
                    data_photometry: dict = None, photometry_times_pyc: np.ndarray = None,
                    photometry_keys: list = None):
        """
        Visualise a session using Spike2

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

        smrx_filename: str = None
            The output filename (*.smrx)

        print_to_text: Bool = True
            print_lines will be converted to text (and TextMark channel in Spike2)

        vchange_to_text: Bool = True
            Variable changes during the session, eg. "V 12560 windor_dur_ms 3000", will be converted to text (and TextMark channel in Spike2)

        data_photometry: dict = None
            Holding photometry data
            If None, the photometry channels will be skipped
        
        photometry_times_pyc: np.ndarray
            Rsync-ed pyphotometry time stamps in pycontrol time in ms
        
        photometry_keys: list = None
            Specify what channels to export.

        verbose :bool = False


        """

        # see  \Users\phar0528\Anaconda3\envs\trialexp\Lib\site-packages\sonpy\MakeFile.py
        #NOTE cannot put file path in the pydoc block

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
            # allow exporting time stamps into event channels
            for dct in event_ms:
                y_index += 1
                spike2exporter.write_event(dct['time_ms'], dct['name'], y_index)

        if export_state:
            # Draw states as gapped lines
            state_dict = extract_states(df)
            
            for state, time_ms in state_dict.items():
                y_index += 1
                spike2exporter.write_marker_for_state(time_ms, state, y_index)
        
        #TODO accept custom state defintions?

        if print_lines:

            # NOTE . doesn't capture \n and re.DOTALL is required below
            EXPR = '^(\d+)\s(.+)'
            list_of_match = [match(
                EXPR, L, DOTALL) for L in print_lines if match(EXPR, L) is not None]
            ts_ms = [int(m.group(1)) for m in list_of_match]
            txt = [m.group(2) for m in list_of_match]

            # df_print = pd.DataFrame(list(zip(ts_ms, txt)), columns=['ms', 'text'])

            if txt:
                y_index += 1
                spike2exporter.write_textmark(ts_ms, 'print lines', y_index, txt) 
            
        if v_lines:
            EXPR = '^([1-9]\d*)\s(.+)' #NOTE Need to ignore the defaults (V 0 ****)
            list_of_match = [match(EXPR, L) for L in v_lines if match(EXPR, L) is not None]
            ts_ms = [int(m.group(1)) for m in list_of_match]
            txt = [m.group(2) for m in list_of_match]
  
            # df_print = pd.DataFrame(list(zip(ts_ms, txt)), columns=['ms', 'text'])
            if txt:
                y_index += 1
                spike2exporter.write_textmark(ts_ms, 'V changes', y_index, txt)

        if (data_photometry is not None) and (photometry_times_pyc is not None) \
            and (photometry_keys is not None) :

            multiplier = int((1/1000) / spike2exporter.dTimeBase) #NOTE sampling_rate was originally 1000, and we assume that it is unchanged
            T = photometry_times_pyc
            nan_indices = np.argwhere(np.isnan(T))
            T_no_nan = np.delete(T, nan_indices)

            new_T = np.arange(0, df.time.max(), 1/1000*1000) #NOTE sampling_rate was originally 1000

            def write_photometry(name):
                Y = data_photometry[name]
                Y_no_nan = np.delete(Y, nan_indices)  # []

                if len(Y_no_nan) == len(T_no_nan):
                    new_Y = np.interp(new_T, T_no_nan, Y_no_nan)

                    spike2exporter.write_waveform(new_Y, name, y_index, multiplier)
                else:
                    # the length mismatch, maybe a wrong data type (not a time series?)
                    #TODO issue a warning or raise an error? 
                    ...

            for name in photometry_keys:
                y_index += 1
                if name in data_photometry:
                    write_photometry(name)
                else:
                    # name is not found in data_photometry
                    #TODO issue a warning or error or nothing? 
                    ...

        elif (data_photometry is None) :
            ...
            # skip exporting photometry
            
 

def export_session_v2(df:pd.DataFrame, keys: list = None, export_state=True, print_expr: list = None, 
                    event_ms: list = None, smrx_filename: str = None, 
                    verbose :bool = False,
                    df_print=None, 
                    df_variable=None,
                    data_photometry: dict = None, photometry_times_pyc: np.ndarray = None,
                    photometry_keys: list = None):
        """
        Visualise a session using Spike2, support for pycontrol >v2

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

        smrx_filename: str = None
            The output filename (*.smrx)

        print_to_text: Bool = True
            print_lines will be converted to text (and TextMark channel in Spike2)

        vchange_to_text: Bool = True
            Variable changes during the session, eg. "V 12560 windor_dur_ms 3000", will be converted to text (and TextMark channel in Spike2)

        data_photometry: dict = None
            Holding photometry data
            If None, the photometry channels will be skipped
        
        photometry_times_pyc: np.ndarray
            Rsync-ed pyphotometry time stamps in pycontrol time in ms
        
        photometry_keys: list = None
            Specify what channels to export.

        verbose :bool = False


        """

        # see  \Users\phar0528\Anaconda3\envs\trialexp\Lib\site-packages\sonpy\MakeFile.py
        #NOTE cannot put file path in the pydoc block

        if keys is None:
            keys = df.name.unique()
        else:
            for k in keys: 
               assert k in df.content.unique(), f"{k} is not found in self.time.keys()"
        
        
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
                curState  = df_states.iloc[0]['content']
                start_time = df_states.iloc[0]['time']
                
                for _, row in df_states.iloc[1:].iterrows():
                    if not row.name == curState:
                        states_dict[curState].extend([start_time, row.time])
                        start_time = row['time']
                        curState = row['content']
                        
            return states_dict  

        y_index = 0
        
        for kind, k in enumerate(keys):
            y_index += 1
            df_evt2plot = df[df.content==k]
            spike2exporter.write_event(df_evt2plot.time.values, k, y_index)

        if event_ms is not None:
            if isinstance(event_ms, dict):
                event_ms = [event_ms]
            # allow exporting time stamps into event channels
            for dct in event_ms:
                y_index += 1
                spike2exporter.write_event(dct['time_ms'], dct['name'], y_index)

        if export_state:
            # Draw states as gapped lines
            state_dict = extract_states(df)
            
            for state, time_ms in state_dict.items():
                y_index += 1
                spike2exporter.write_marker_for_state(time_ms, state, y_index)
        
        #TODO accept custom state defintions?

        if df_print is not None:

            ts_ms = df_print.time.values
            txt = df_print.content.values

            # df_print = pd.DataFrame(list(zip(ts_ms, txt)), columns=['ms', 'text'])

            if len(txt)>0:
                y_index += 1
                spike2exporter.write_textmark(ts_ms, 'print lines', y_index, txt) 
            
        if df_variable is not None and len(df_variable)>0:
            ts_ms = df_variable.time.values
            txt = [f'{r.key}:{r.value}' for _,r in df_variable.iterrows()]
            # df_print = pd.DataFrame(list(zip(ts_ms, txt)), columns=['ms', 'text'])
            if len(txt)>0:
                y_index += 1
                spike2exporter.write_textmark(ts_ms, 'V changes', y_index, txt)

        if (data_photometry is not None) and (photometry_times_pyc is not None) \
            and (photometry_keys is not None) :

            multiplier = int((1/1000) / spike2exporter.dTimeBase) #NOTE sampling_rate was originally 1000, and we assume that it is unchanged
            T = photometry_times_pyc
            nan_indices = np.argwhere(np.isnan(T))
            T_no_nan = np.delete(T, nan_indices)

            new_T = np.arange(0, df.time.max(), 1/1000*1000) #NOTE sampling_rate was originally 1000

            def write_photometry(name):
                Y = data_photometry[name]
                Y_no_nan = np.delete(Y, nan_indices)  # []

                if len(Y_no_nan) == len(T_no_nan):
                    new_Y = np.interp(new_T, T_no_nan, Y_no_nan)

                    spike2exporter.write_waveform(new_Y, name, y_index, multiplier)
                else:
                    # the length mismatch, maybe a wrong data type (not a time series?)
                    #TODO issue a warning or raise an error? 
                    ...

            for name in photometry_keys:
                y_index += 1
                if name in data_photometry:
                    write_photometry(name)
                else:
                    # name is not found in data_photometry
                    #TODO issue a warning or error or nothing? 
                    ...

        elif (data_photometry is None) :
            ...
            # skip exporting photometry

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
    
    last_time = max([i for i in list_ev if i < find_min_time_list(list_lim)], default=np.nan)

        
    return last_time


def find_min_time_list(x):
    
    if isinstance(x, list):
        if len(x) == 0:
            min_time = np.nan
        else:
            min_time = min([i for i in x if i>0], default=np.nan)
    else:
        min_time = x

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


def get_sync_time(df_pycontrol):
    # return the rsync time
    if df_pycontrol.attrs['framework_version'] == '1.8.1':
        return df_pycontrol[df_pycontrol.content=='rsync'].time
    else:
        return df_pycontrol[df_pycontrol.subtype=='sync'].time


####### Lick detection


def analyze_lick_signal(lick_signal_path, lick_ts_path, lick_fs=200):
    
    lick_signal = np.load(lick_signal_path).astype(float)
    lick_ts = np.load(lick_ts_path).astype(float)
    
    if len(lick_signal)< lick_fs*60:
        print('Warning: Lick signal is too short')
        return ([],[],[])

    [b,a]  = scipy.signal.butter(5, 10/(lick_fs/2))
    lick = scipy.signal.filtfilt(b,a,lick_signal)
    lick = scipy.signal.savgol_filter(lick, 51,1)
    
    
    # check if the signal is inverted or not
    mean_lick = np.mean(lick)
    pos_peaks_ratio = np.mean(lick > mean_lick+2*np.std(lick))
    neg_trough_ratio = np.mean(lick < mean_lick-2*np.std(lick))
    
    if neg_trough_ratio> pos_peaks_ratio:
        # signal is probably inverted, convert trough to peak
        lick = -(lick - mean_lick)
    else:
        lick = lick - mean_lick
        
    thres = 2*np.std(lick)
    lick_on = lick>thres
    
    lick_on_t = lick_ts[np.diff(lick_on, append=[0])==1] *1000 # be consistent in ms unit
    lick_off_t = lick_ts[np.diff(lick_on, append=[0])==-1] *1000
    
    # plt.plot(lick_ts*1000, lick)
    # plt.plot(lick_on_t, np.ones((len(lick_on_t),))*100, '^')
    # plt.plot(lick_off_t, np.ones((len(lick_off_t),))*100, 'rv')
    if len(lick_on_t) == len(lick_off_t):
        print(f'Warning: lick_on and lick_off does not match {len(lick_on_t)} vs {len(lick_off_t)}')

    return (lick_on_t, lick_off_t, lick)

def add_lick_events(df_pycontrol, lick_on, lick_off):
    df_lick_on = pd.DataFrame({
        'time':lick_on,
        'type': 'event',
        'subtype': 'task',
        'content': 'lick'
    })
    
    df_lick_off = pd.DataFrame({
        'time':lick_off,
        'type': 'event',
        'subtype': 'task',
        'content': 'lick_off'
    })
    
    attrs = df_pycontrol.attrs # for some reason concat does not perserve the attrs
    
    df_pycontrol = pd.concat([df_pycontrol, df_lick_on, df_lick_off]).sort_values('time').reset_index()
    df_pycontrol.attrs = attrs
    
    return df_pycontrol

def get_discrminability_score(df_pycontrol, on_state, off_state, event='spout'):
    # Calculate the discrminability score between action in the two states
    cue_on = df_pycontrol[df_pycontrol.content==on_state]
    touch_rate = []

    # find the on state touch rate
    for _, row in cue_on.iterrows():
        start_time = row.time
        end_time = row.time + row.duration

        df_states = df_pycontrol[(df_pycontrol.time>start_time) & (df_pycontrol.time<=end_time)]
        num_events = len(df_states[df_states.content==event])
        if row.duration>0:
            touch_rate.append(num_events/(row.duration/1000)) # in event per second

    touch_rate = np.array(touch_rate)
    cue_off = df_pycontrol[df_pycontrol.content==off_state]


    touch_rate_off = []

    # find the off state touch rate
    for _, row in cue_off.iterrows():
        start_time = row.time
        end_time = row.time + row.duration

        df_states = df_pycontrol[(df_pycontrol.time>start_time) & (df_pycontrol.time<=end_time)]
        num_events = len(df_states[df_states.content==event])
        if row.duration>0:
            touch_rate_off.append(num_events/(row.duration/1000)) # in event per second

    touch_rate_off = np.array(touch_rate_off[:len(touch_rate)])
    
    all_rate = np.concatenate([touch_rate, touch_rate_off])
    rate_mean = np.mean(all_rate)
    rate_std = np.std(all_rate)

    # transform the rate into z score
    ztouch_rate = (touch_rate-rate_mean)/rate_std
    ztouch_rate_off = (touch_rate_off-rate_mean)/rate_std

    H = np.mean(ztouch_rate)
    F = np.mean(ztouch_rate_off)

    return H-F, H, F

def get_windowed_discriminability_score(df_pycontrol, window_sec=3*60):
    # window should be in ms
    # calculate the discriminability score in each window
    window = window_sec*1000
    df_pycontrol = df_pycontrol.copy() 
    bins = np.arange(df_pycontrol.time.min(), df_pycontrol.time.max()+window, window)
    df_pycontrol['discrim_bin'] = pd.cut(df_pycontrol.time, bins)
    dprime = df_pycontrol.groupby('discrim_bin').apply(lambda df: get_discrminability_score(df, 'busy_win','short_break')[0])
    t = [idx.left/1000 for idx in dprime.index]

    return pd.DataFrame({'time': t, 'dprime':dprime.values})
 

def discrminability_before_after(df_pycontrol, windows=(-1000,1000), event_name='busy_win'):
    # use the touch rate before and after cue as the discrminability score
    busy_win_time = df_pycontrol[df_pycontrol.content==event_name].time
    spout_time = df_pycontrol[df_pycontrol.content == 'spout']
    
    before_reach = np.zeros_like(busy_win_time)
    after_reach = np.zeros_like(busy_win_time)
    for i,t in enumerate(busy_win_time):
      before_reach[i] = len(spout_time[(spout_time.time<=t) & (spout_time.time> (t+windows[0]))])
      after_reach[i] = len(spout_time[(spout_time.time>=t) & (spout_time.time< (t+windows[1]))])

    return pd.DataFrame({
      'cue_time':busy_win_time,
      'before_cue_reach': before_reach,
      'after_cue_reach': after_reach 
    })


def calculate_lick_rate(df_event, dataset, lick_bin_size=0.2):
    """
    Calculate lick rate from event data and return as xarray DataArray.
    
    Parameters:
    -----------
    df_event : pd.DataFrame
        Event dataframe containing lick events
    dataset : xr.Dataset
        Dataset containing time coordinates and sampling rate
    lick_bin_size : float, optional
        Window size for rolling mean in seconds (default: 0.2)
    
    Returns:
    --------
    xr.DataArray
        Lick rate data array with time coordinates
    """
    lick_on = df_event[df_event.content=='lick'].time
    
    lick_rate,_ = np.histogram(lick_on, dataset.time)
    
    # calculate the rolling mean lick rate
    win_size = int(lick_bin_size*dataset.attrs['sampling_rate'])
    lick_rate = np.convolve(lick_rate, np.ones(win_size)/win_size, mode='same')*dataset.attrs['sampling_rate']
    
    xa_lick_rate = xr.DataArray(lick_rate,
                                coords={'time':dataset.time[:-1]},dims=['time'])
    return xa_lick_rate



def add_events_to_time_series(df_pycontrol, df):
    """
    Adds event and state indicators to a time series DataFrame based on pyControl event/state logs.
    Parameters
    ----------
    df_pycontrol : pandas.DataFrame
        DataFrame containing pyControl logs with columns 'type', 'content', 'time', and 'duration'
    df : pandas.DataFrame
        Time series DataFrame with a 'time' column representing time points to annotate.
    Returns
    -------
    pandas.DataFrame
        The input time series DataFrame with additional columns for each event and state.
        Event columns are named 'event:<event_name>' and contain 1 at time points where the event occurred, 0 otherwise.
        State columns are named 'state:<state_name>' and contain 1 for time intervals where the state is active, 0 otherwise.
    """
    events = df_pycontrol[(df_pycontrol.type=='event') & (df_pycontrol.content!='rsync')]
    states = df_pycontrol[df_pycontrol.type=='state']

    # events
    for i, evt in enumerate(events.content.unique()):
        evt_data = np.zeros((len(df),))
        evt_time = events[events.content==evt].time
        idx = np.searchsorted(df.time, evt_time)
        evt_data[idx-1] = i
        df['event:'+evt] = evt_data

    # states
    for sts in states.content.unique():
        state_data = np.zeros((len(df),))
        state_start_time = states[states.content==sts].time
        state_end_time = state_start_time + states[states.content==sts].duration 
        state_start_idx = np.searchsorted(df.time, state_start_time)-1
        state_end_idx = np.searchsorted(df.time, state_end_time)-1

        for start, end in zip(state_start_idx, state_end_idx):
            state_data[start:end] = 1
        
        df['state:'+sts] = state_data
    
    return df 