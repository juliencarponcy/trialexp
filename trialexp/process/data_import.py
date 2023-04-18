# Python classes for importing pyControl data files and representing pyControl 
# sessions and experiments.  Dependencies: Python 3.5+, Numpy.
import datetime
import itertools
import os
import pickle
import re
import warnings
from cmath import isnan, nan
from collections import namedtuple
from itertools import compress
from math import ceil
from operator import itemgetter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.validators.scatter.marker import SymbolValidator
from scipy.signal import butter, decimate, filtfilt
from scipy.stats import linregress, zscore

from trialexp.dataset_classes.trial_dataset_classes import *

from trialexp.process.pyphotometry.photometry_functional import *
from trialexp.process.pyphotometry.utils import *
from trialexp.process.pycontrol.utils import *

from trialexp.utils.data_organisation import *
from trialexp.utils.DLC_utilities import *
from trialexp.utils.pycontrol_utilities import *
from trialexp.utils.rsync import *



# Custom type, assess usefulness
NoneType = type(None)

# Custom exception, put elsewhere
class DeepLabCutFileError(Exception):
    def __init__(self, subject_ID, datetime, camera_keyword):
        self.message = f'No DLC file matching {subject_ID} at {datetime} \
            on camera {camera_keyword}'
        super().__init__(self.message)


#----------------------------------------------------------------------------------
# Session class
#----------------------------------------------------------------------------------


class Session():
    """
    Import data from a pyControl file and represent it as an object 
    
    Attributes
    ----------
    file_name : str
        .txt file name
    experiment_name : str
    task_name : str
    setup_ID : str
        The COM port of the computer used (can be useful when multiple rigs on one computer)
    subject_ID : integer or str
        If argument int_subject_IDs is True, suject_ID is stored as an integer,
        otherwise subject_ID is stored as a string.
    datetime
        The date and time that the session started stored as a datetime object.
    datetime_string
        The date and time that the session started stored as a string of format 'YYYY-MM-DD HH:MM:SS'
    events : list of namedtuple
        A list of all framework events and state entries as objects in the order they occurred. 
        Each entry is a namedtuple with fields 'time' & 'name', such that you can get the 
        name and time of event/state entry x with x.name and x.time respectively.
    times : dict
        A dictionary with keys that are the names of the framework events and states and 
        corresponding values which are Numpy arrays of all the times (in milliseconds since the
        start of the framework run) at which each event/state entry occurred.
    print_lines : list
        A list of all the lines output by print statements during the framework run, each line starts 
        with the time in milliseconds at which it was printed.
    number : scalar integer
    analyzed : bool
    trial_window : list
        eg [-2000, 6000]
        The time window relative to triggers used for trial-based data fragmentation in Trial_Dataset class.
        Set by Session.get_task_specs() 
        cf. timelim
    triggers : list
        eg ['CS_Go']
        Set by Session.get_task_specs() depending on tasksfile `trialexp\params\tasks_params.csv`
    events_to_process : list
        Set by Session.get_task_specs() depending on tasksfile `trialexp\params\tasks_params.csv`
    conditions : list
        Set by Session.get_task_specs() depending on tasksfile `trialexp\params\tasks_params.csv`
    timelim : list
        eg [0, 2000]
        The time window used to detect successful trials. Set bySession.get_task_specs()
        cf. compute_success()
        cf. trial_window
    df_events : DataFrame
        DataFrame with rows for all the trials. 
    df_conditions : DataFrame
        DataFrame with rows for all the trials and columns for 'trigger', 'valid' and the keys of conditions_list passed to
        Experiment.behav_events_to_dataset. df_conditions is a subset of df_events with fewer columns (maybe)
    photometry_rsync
    files : dict
    """

    def __init__(self, file_path, int_subject_IDs=True, verbose=False):

        # Load lines from file.

        with open(file_path, 'r') as f:
            if verbose:
                print('Importing data file: '+os.path.split(file_path)[1])
            all_lines = [line.strip() for line in f.readlines() if line.strip()]

        # Extract and store session information.

        self.file_name = os.path.split(file_path)[1]

        info_lines = [line[2:] for line in all_lines if line[0]=='I']

        self.experiment_name = next(line for line in info_lines if 'Experiment name' in line).split(' : ')[1]
        self.task_name       = next(line for line in info_lines if 'Task name'       in line).split(' : ')[1]
        self.setup_ID        = next(line for line in info_lines if 'Setup ID'        in line).split(' : ')[1]
        subject_ID_string    = next(line for line in info_lines if 'Subject ID'      in line).split(' : ')[1]
        datetime_string      = next(line for line in info_lines if 'Start date'      in line).split(' : ')[1]

        # take only filename of the task (without .py), not the folder
        # plan for Unix acquisition (parsing task path)
        # appears not functional
        self.task_name = os.path.basename(self.task_name)


        if int_subject_IDs: # Convert subject ID string to integer.
            self.subject_ID = int(''.join([i for i in subject_ID_string if i.isdigit()]))
        else:
            self.subject_ID = subject_ID_string

        self.datetime = datetime.strptime(datetime_string, '%Y/%m/%d %H:%M:%S')
        self.datetime_string = self.datetime.strftime('%Y-%m-%d %H:%M:%S')

        # Extract and store session data.

        state_IDs = eval(next(line for line in all_lines if line[0]=='S')[2:])
        event_IDs = eval(next(line for line in all_lines if line[0]=='E')[2:])

        ID2name = {v: k for k, v in {**state_IDs, **event_IDs}.items()}

        data_lines = [line[2:].split(' ') for line in all_lines if line[0]=='D']

        self.events = [Event(int(dl[0]), ID2name[int(dl[1])]) for dl in data_lines]

        self.times = {event_name: np.array([ev.time for ev in self.events if ev.name == event_name])  
                      for event_name in ID2name.values()}

        # self.print_lines = [line[2:] for line in all_lines if line[0]=='P']

        # capture multiple line print messages as self.print_lines
        count = 0
        self.print_lines = []
        while count < len(all_lines):
            if bool(re.match('P\s\d+\s',all_lines[count])):  # all_lines[count][0] == 'P'
                self.print_lines.append(all_lines[count][2:])
                count += 1
                while (count < len(all_lines)) and not(bool(re.match('[PVD]\s\d+\s', all_lines[count]))):
                    self.print_lines[-1] = self.print_lines[-1] + \
                        "\n" + all_lines[count]
                    count += 1
            else:
                count += 1

        self.v_lines = [line[2:] for line in all_lines if line[0]=='V']
        
        self.state_IDs = state_IDs
        self.event_IDs = event_IDs
    
    def get_task_specs(self, tasksfile, trial_window, timelim):
        """
        All the df columns named in this function, events and opto_categories must 
        follow columns of the indicated tasksfile
        
        This function sets the values of 
            self.triggers
            self.events_to_process
            self.conditions
            self.trial_window
            self.timelim

        """

        tasks_trig_and_events = pd.read_csv(tasksfile)
        # match triggers (events/state used for t0)
        self.triggers = np.array2string(tasks_trig_and_events['triggers'][tasks_trig_and_events['task'] == 
            self.task_name].values).strip("'[]").split('; ')
                
        # events to extract
        self.events_to_process = np.array2string(tasks_trig_and_events['events'][tasks_trig_and_events['task'] ==
            self.task_name].values).strip("'[]").split('; ')
        # printed line in task file indicating
        # the type of optogenetic stimulation
        # used to group_by trials with same stim/sham
        self.conditions = np.array2string(tasks_trig_and_events['conditions'][tasks_trig_and_events['task'] ==
            self.task_name].values).strip("'[]").split('; ')
        
        # REMOVED, now only at Experiment level to avoid inconsistencies
        # define trial_window parameter for extraction around triggers
        # self.trial_window = trial_window
        self.timelim = timelim
        
        return self

    def extract_data_from_session(self):
        """
        The two attributes
            self.df_events
            self.df_conditions
        are assigned by looking into a session data
        """

        df_events = pd.DataFrame(self.events, columns=['timestamp', 'event'])
        
        df_events['timestamp'] = df_events['timestamp'].astype('int')

        # parsing timestamps and events from print lines
        print_events = [line.split() for line in self.print_lines]
        print_ts = [int(line[0]) for line in print_events]
        print_text = [' '.join(line[1:]) for line in print_events]
        
        # put timestamp and text of print lines in a dataframe, and make the timestamp as index
        df_print_events = pd.DataFrame({'timestamp':print_ts,'event':print_text})
        
        # keep print_lines that are relevant to task analysis
        df_print_events = df_print_events.loc[df_print_events['event'].isin(
            self.triggers + self.events_to_process + self.conditions
        )]
 
        # keep events in df if any event is relevant for behaviour        
        df_events = df_events.loc[df_events['event'].isin(
            self.triggers + self.events_to_process + self.conditions
        )]

        # Merge print and events which are relevant to conditions of trials (time insensitive)
        df_conditions = pd.concat([df_print_events.loc[df_print_events['event'].isin(self.conditions)],
            df_events.loc[df_events['event'].isin(self.conditions)]]
        , ignore_index=False)

        # Merge print and events which are relevant to events of trials (time sensitive)
        df_events = pd.concat([df_print_events.loc[df_print_events['event'].isin(self.triggers + self.events_to_process)],
            df_events.loc[df_events['event'].isin(self.triggers + self.events_to_process)]]
        , ignore_index=True)

        # Turn into events/conditions string into categorical variables      
        df_events['event'] = df_events['event'].astype('category')
        df_conditions['event'] = df_conditions['event'].astype('category')

        # use timestamp as index
        df_events.set_index('timestamp',inplace=True, drop=True)
        df_conditions.set_index('timestamp',inplace=True, drop=True)

        self.df_events = df_events
        self.df_conditions = df_conditions
        
        # return session objet
        return self

    # VERY UGLY AND SLOW:
    # compute trial nb and triggering events types to aggegate and index on them
    # TODO: optimize with itertuples or apply 
    def compute_trial_nb(self, trial_window):

        # just reassigning object attributes to new variables to improve readibility
        # almost certainly at the expense of performance
        
        df_events = self.df_events
        df_conditions = self.df_conditions

        # initializing trial_nb and trial_time column and casting as float,
        # float allows use of NaN values and discarding events outside trials
        # at the expense of memory (probably very mild burden)

        df_events['trial_nb'] = nan
        df_events['trial_nb'] = df_events['trial_nb'].astype(float)

        df_events['trigger'] = ''
        df_events['trial_time'] = nan
        df_events['trial_time'] = df_events['trial_time'].astype(float)

        df_conditions['trial_nb'] = nan
        df_conditions['trial_nb'] = df_conditions['trial_nb'].astype(float)

        df_conditions['trigger'] = ''
        df_conditions['trial_time'] = nan
        df_conditions['trial_time'] = df_conditions['trial_time'].astype(float)

        # for each trigger type extract and concatenate trial times and trigger type
        for tidx, trig in enumerate(self.triggers):
            if tidx == 0:
                all_trial_times = [t for t in self.times[trig]]
                all_trial_triggers = [trig for i in range(len(self.times[trig]))]
            else:
                all_trial_times = all_trial_times + [t for t in self.times[trig]]
                all_trial_triggers = all_trial_triggers + [trig for i in range(len(self.times[trig]))]
        # sort trial times and triggers in chronological order
        indices, all_trial_times_sorted = zip(*sorted(enumerate(all_trial_times), key=itemgetter(1)))
        #print(indices, all_trial_times_sorted)
        all_trial_triggers_sorted = [all_trial_triggers[i] for i in indices]
        #print(all_trial_triggers_sorted)
        
        for ntrial, trigtime in enumerate(list(all_trial_times_sorted)):
                        
            # attribute trial_nb to event/conditions occuring around trigger (event/state)
            # if the loop is at the last trial
            if ntrial == len(all_trial_times_sorted)-1:
                #print('last trial: ', ntrial)
                df_events['trial_nb'].mask(
                    (df_events.index >= trigtime + trial_window[0])
                , ntrial+1, inplace=True)

                df_conditions['trial_nb'].mask(
                    (df_conditions.index >= trigtime + trial_window[0])
                , ntrial+1, inplace=True)

                df_events['trial_time'].mask(
                    (df_events.index >= trigtime + trial_window[0])
                , df_events.index[
                    (df_events.index >= trigtime + trial_window[0])
                ] - trigtime, inplace=True)

                # determine triggering event
                df_conditions['trigger'].mask(
                    (df_conditions.index >= trigtime + trial_window[0])
                , all_trial_triggers_sorted[ntrial], inplace=True)

                # compute trial relative time
                df_conditions['trial_time'].mask(
                    (df_conditions.index >= trigtime + trial_window[0])
                , df_conditions.index[
                    (df_conditions.index >= trigtime + trial_window[0])
                ] - trigtime, inplace=True)

            # for every trial except last
            else: 
                #print('all but last trial: ', ntrial, 'over', len(all_trial_times_sorted),  all_trial_times_sorted[ntrial+1])

                df_events['trial_nb'].mask(
                    (df_events.index >= trigtime + trial_window[0]) &
                    (df_events.index <= all_trial_times_sorted[ntrial+1] + trial_window[0])
                , ntrial+1, inplace=True)

                df_conditions['trial_nb'].mask(
                    (df_conditions.index >= trigtime + trial_window[0]) &
                    (df_conditions.index <= all_trial_times_sorted[ntrial+1] + trial_window[0])
                , ntrial+1, inplace=True)

                df_events['trial_time'].mask(
                    (df_events.index >= trigtime + trial_window[0]) &
                    (df_events.index <= all_trial_times_sorted[ntrial+1] + trial_window[0])
                , df_events.index[
                    (df_events.index >= trigtime + trial_window[0]) &
                    (df_events.index <= all_trial_times_sorted[ntrial+1] + trial_window[0])
                ] - trigtime, inplace=True)

                # determine triggering event
                df_conditions['trigger'].mask(
                    (df_conditions.index >= trigtime + trial_window[0]) &
                    (df_conditions.index <= all_trial_times_sorted[ntrial+1] + trial_window[0])
                , all_trial_triggers_sorted[ntrial], inplace=True)

                # compute trial relative time
                df_conditions['trial_time'].mask(
                    (df_conditions.index >= trigtime + trial_window[0]) &
                    (df_conditions.index <= all_trial_times_sorted[ntrial+1] + trial_window[0])
                , df_conditions.index[
                    (df_conditions.index >= trigtime + trial_window[0]) &
                    (df_conditions.index <= all_trial_times_sorted[ntrial+1] + trial_window[0])
                ] - trigtime, inplace=True)
        
        # drop events without trial_nb attributed (normally first events only)
        df_events.dropna(axis=0, subset=['trial_nb'], inplace=True)
        df_conditions.dropna(axis=0, subset=['trial_nb'], inplace=True)

        # reconvert trial_nb and trial_time to int since now there is no more need for nan values
        df_events['trial_nb'] = df_events['trial_nb'].astype(int)
        df_conditions['trial_nb'] = df_conditions['trial_nb'].astype(int)

        df_events['trial_time'] = df_events['trial_time'].astype(int)
        df_conditions['trial_time'] = df_conditions['trial_time'].astype(int)

        # aggregate events trial_time by trial_nb and event
        df_events = df_events.groupby(['trial_nb','event']).agg(list)

        # construct a new dataframe to flattened multi-indexed based events
        # and introduce trigger, valid and success columns
        ev_col_list = [i + '_trial_time' for i in self.events_to_process]

        columns=['timestamp', 'trigger', 'valid', 'success', 'uid']
        columns = columns + ev_col_list

        new_df = pd.DataFrame(index= range(1,len(all_trial_times_sorted)+1),
            columns=columns)

        # Create unique identifiers for trials
        new_df['trial_nb'] = new_df.index.values
        new_df['uid'] = new_df['trial_nb'].apply(
            lambda x: f'{self.subject_ID}_{self.datetime.date()}_{self.datetime.time()}_{x}')
        new_df['trial_nb'].drop(columns='trial_nb', inplace=True)

        # fill new <event>_trial_time columns
        for ev in self.events_to_process:
            try:
                new_df.loc[df_events.index.get_level_values('trial_nb').unique(), str(ev + '_trial_time')] = df_events.loc[
                    (df_events.index.get_level_values('trial_nb'), ev), 'trial_time'].values
            except KeyError as ke:
                print('No event ', ke, ' found: ',
                    self.subject_ID, self.datetime_string, self.task_name)

        # fill the new_df with timestamps of trigger and trigger types
        new_df['timestamp'] = all_trial_times_sorted
        new_df['trigger'] = all_trial_triggers_sorted

        # validate trials in function of the time difference between trials (must be > at length of trial_window)
        new_df['valid'] = new_df['timestamp'].diff() > trial_window[0]
        
        # validate first trial except if too early in the session
        if new_df['timestamp'].iloc[0] > abs(trial_window[0]):
           new_df.loc[1, 'valid'] = True
        
        # assing the newly built dataframe into the session object
        self.df_events = new_df

        # replace NaN by an empty lists for trial without events
        for ev_times_col in ev_col_list:
            self.df_events.loc[(self.df_events[ev_times_col].isnull()), ev_times_col] = \
                self.df_events.loc[(self.df_events[ev_times_col].isnull()), ev_times_col].apply(lambda x: [])
            

        #self.df_events = new_df
        self.df_conditions = df_conditions


        # check if df_conditions and df_events have same number of rows
        # assert(self.df_events['uid'] == self.df_conditions['uid'])

        return self

    def compute_conditions_by_trial(self):

        vars = self.df_conditions[['trigger','trial_nb']]
        
        # One-hot encoding of conditions to detect :
        # turn a column of string into as much columns as conditions to detect
        # and fill values with 0 and 1
        dummies = pd.get_dummies(self.df_conditions['event'])

        # join numeric variables (timestamp, trial_nb, trial_time) with dummies (one-hot encoded) variables
        self.df_conditions = vars.join(dummies, on='timestamp')


        # do this only if conditions have been defined, e.g. opto protocols, free reward occurences...
        # df_conditions will be used to separate trials with the same trigger into different sub-categories depending
        # on what occured during this trial, or the specific variation of that trial, the idea is to use
        # conditions as a categorization for trial sub-types, with time insensitive information.

        # df_events on the other hands, serves to store time sensitive information, and multiple similar
        # events could occur during the same trial. Timestamps of each trial (relative or absolute)
        # will be then aggregated and used to compute nb of occurences or mean interval between events
        if 'nan' not in self.conditions:
            #print(self.df_conditions)
            # new variable for readability
            df_conditions = self.df_conditions

            # check for condition(s) not present in file
            absent_conditions = set(self.conditions).difference(df_conditions.columns)
            # add the corresponding columns and fill with 0 to facilitate further analyses
            # especially in cases where a condition is present in some sessions and not in others
            for cond in absent_conditions:
                df_conditions[cond] = 0

            # group by trigger and trial nb
            df_conditions_summed = self.df_conditions[self.conditions + ['trial_nb']].groupby(['trial_nb'], as_index=True).agg(lambda x: sum(x)) 
            # Replicate the index of df_events to complete for trials where no condition were detected
            df_conditions_summed = df_conditions_summed.reindex(index=self.df_events.index)
            df_conditions_summed.loc[:,['trigger','valid','success']] = self.df_events.loc[:,['trigger','valid','success']]
            
            self.df_conditions = df_conditions_summed
            self.df_conditions.loc[:,self.conditions] = self.df_conditions[self.conditions].astype(bool)

        else:
            self.df_conditions = self.df_events.iloc[:,1:4] # TODO: check and/or remove
            df_conditions_summed = self.df_conditions

        ###################################################################################
        # Compute if trials are cued or uncued for this specific task, to be removed asap #
        ###################################################################################
        if self.task_name == 'reaching_go_spout_cued_uncued':
            # FORCED: timestamp threshold, assume that blocks are 20min long, adapt otherwise
            block_lim = 20*60*1000

            # extract starting_block
            try:
                starting_block = [p for p in self.print_lines if 'Starting block' in p][0].split(': ')[1]
            # if starting block if not defined, the first block is Cued
            except:
                starting_block = 'Cued'
            
            # assign trial conditions to cued or uncued depending on timestamps
            if starting_block == 'Cued':               
                self.df_conditions.loc[(self.df_events.timestamp < block_lim),'cued'] = True
                self.df_conditions.loc[(self.df_events.timestamp > block_lim),'cued'] = False
            # else:
                # self.df_conditions.loc[(self.df_events.timestamp > block_lim),'cued'] = True
                # self.df_conditions.loc[(self.df_events.timestamp < block_lim),'cued'] = False
            self.df_conditions['cued'] = self.df_conditions['cued'].astype('bool') 
            # to avoid FurureWarning
            # FutureWarning: In a future version, object-dtype columns with all-bool values will not be 
            # included in reductions with bool_only=True. Explicitly cast to bool dtype instead.

            # change triggers name for this task to cued and uncued
            self.triggers = ['cued', 'uncued']
            self.df_conditions.loc[(self.df_conditions.cued == True),['trigger']] = self.triggers[0]
            self.df_conditions.loc[(self.df_conditions.cued == False),['trigger']] = self.triggers[1]
        #     self.df_events.loc[(self.df_conditions.cued == True),['trigger']] = self.triggers[0]
        #     self.df_events.loc[(self.df_conditions.cued == False),['trigger']] = self.triggers[1]
        # # print(self.df_events.shape, self.df_conditions.shape)
                    
        ##############################################
        # End of block to remove (for specific task) #
        ##############################################

        self.df_conditions['trial_nb'] = self.df_conditions.index.values
        
        # Create unique identifiers for trials
        self.df_conditions['uid'] = self.df_conditions['trial_nb'].apply(
                lambda x: f'{self.subject_ID}_{self.datetime.date()}_{self.datetime.time()}_{x}')
        self.df_conditions.drop(columns = 'trial_nb', inplace=True)
        # check if df_conditions and df_events have same number of rows
        assert(self.df_events['uid'].equals(self.df_conditions['uid']))

        
        return self

    def create_metadata_dict(self, trial_window):
        metadata_dict = {
            'subject_ID' : self.subject_ID,
            'datetime' : self.datetime,
            'task' : self.task_name,
            'trial_window' : trial_window,
            'success_time_lim' : self.timelim,
            'com_port' : self.setup_ID
        }
        return metadata_dict

    # Perform all the pretreatments to analyze behavioural file by trials
    def get_session_by_trial(
            self, 
            trial_window: list, 
            timelim: list = None,
            tasksfile: str = None, 
            blank_spurious_event: list = None, 
            blank_timelim: list = None, 
            verbose: bool = False):
        """

        """

        # set a minimum nb of events to not process aborted files
        min_ev_to_process = 30
        # Do not process files with less than min_ev_to_process
        if len(self.events) < min_ev_to_process:
            print(f'file too short to process (likely aborted session): \
                {self.subject_ID} {self.datetime_string} {self.task_name}') 
            self.analyzed = False          
            return self
        else:
            try:
                self.analyzed = False
                # self.trial_window removed from Session, only in Experiment
                # to avoid inconsistencies
                # self.trial_window = trial_window 
                # get triggers and events to analyze, set trial_window to be extracted
                # and timelim for which trials are considered success/fail
                self = self.get_task_specs(tasksfile,trial_window, timelim)
                # get triggers and events to analyze
                if verbose:
                    print(f'processing by trial: {self.file_name} task: {self.task_name}')

                self = self.extract_data_from_session()
                self = self.compute_trial_nb(trial_window)
                
                
                if blank_spurious_event is not None:
                    self.df_events[blank_spurious_event + '_trial_time'] = \
                        self.df_events[blank_spurious_event + '_trial_time'].apply(lambda x: blank_spurious_detection(x, blank_timelim))

                self = self.compute_conditions_by_trial()
                self = self.compute_success()

                def tidy_up_duplicate_columns(df_events: pd.DataFrame, df_conditions: pd.DataFrame):
                    '''
                    Temporary helper method to clean duplicated columns (due to previous implementation)
                    between df_events and df_conditions dataframes.

                    Possibly due to the following line:
                    self.df_conditions = self.df_events.iloc[:,1:4] (in compute_conditions_by_trial)

                    This should allow cleaner joining/merging for related functions
                    It keeps 'uid' columns for merging on data that includes more than one subject/session
                    Also check if df_events and df_conditions are still matching on 'uid'
                    '''
                    df_events.drop(columns=['trigger','valid','success','trial_nb'], inplace=True)
                    
                    assert(df_events.uid.equals(df_conditions.uid))

                    return df_events

                self.df_events = tidy_up_duplicate_columns(self.df_events, self.df_conditions)    
                self.metadata_dict = self.create_metadata_dict(trial_window)
                self.analyzed = True
                return self
                #pycontrol_utilities method

            # TODO: These exceptions are useful to deal with too short files or missing a trial type
            # but are are obfuscating code bugs when they happen somewhere. Improve exception handling
            except KeyError as ke:
                if verbose:
                    print(f'No trial {ke} found: {self.file_name} task: {self.task_name}')

                self.analyzed = False
                return self
            except ValueError as ve:
                if verbose:
                    print(f'No trial {ve} found: {self.file_name} task: {self.task_name}')
                self.analyzed = False
                return self

    #----------------------------------------------------------------------------------
    # The following function will be highly customized for each task as it needs to take
    # into accounts specific criterion for a trial to be considered as successful
    #----------------------------------------------------------------------------------

    # TODO: consider putting list of tasks elsewhere,
    # or separate entirely in a more versatile function
    # TODO: identify the most common/likely patterns
    def compute_success(self):
        """computes success trial numbers

        This methods includes task_name-specific definitions of successful trials.
        The results are stored in the 'success' columns of self.df_events and self.df_conditions as bool (True or False).
        """
        self.df_conditions['success'] = False
        # self.df_events['success'] = False
        #print(self.task_name)
        
        
        # To perform for all Go-NoGo variants of the task (list below)
        if self.task_name in ['reaching_go_nogo', 'reaching_go_nogo_jc', 'reaching_go_nogo_opto_continuous',
            'reaching_go_nogo_opto_sinusoid' , 'reaching_go_nogo_opto_sinusoid_spout', 
            'reaching_go_nogo_reversal', 'reaching_go_nogo_reversal_incentive',
            'reaching_go_nogo_touch_spout']:
            # self.triggers[0] refers to CS_Go triggering event most of the time whereas self.triggers[1] refers to CS_NoGo
            # find if spout event within timelim for go trials
            go_success = self.df_events.loc[
                (self.df_events[self.df_events.trigger == self.triggers[0]].index),'spout_trial_time'].apply(
                    lambda x: find_if_event_within_timelim(x, self.timelim))
            go_success_idx = go_success[go_success == True].index
            #print(go_success_idx)
            # categorize successful go trials which have a spout event within timelim
            self.df_conditions.loc[(go_success_idx),'success'] = True
            # self.df_events.loc[(go_success_idx),'success'] = True
            # find if no bar_off event within timelim for nogo trials
            nogo_success = ~self.df_events.loc[
                (self.df_events[self.df_events.trigger == self.triggers[1]].index),'bar_off_trial_time'].apply(
                    lambda x: find_if_event_within_timelim(x, self.timelim))
            nogo_success_idx = nogo_success[nogo_success == True].index
            #print(go_success_idx, nogo_success_idx)
            # categorize as successful trials which contains no bar_off but are not Go trials
            # nogo_success_idx = nogo_success_idx.get_level_values('trial_nb').difference(
            #     self.df_conditions[self.df_conditions['trigger'] == self.triggers[0]].index.get_level_values('trial_nb'))
            self.df_conditions.loc[(nogo_success_idx),'success'] = True
            # self.df_events.loc[(nogo_success_idx),'success'] = True

        # To perform for simple pavlovian Go task, 
        elif self.task_name in ['train_Go_CS-US_pavlovian','reaching_yp', 'reaching_test','reaching_test_CS',
            'train_CSgo_US_coterminated','train_Go_CS-US_pavlovian', 'train_Go_CS-US_pavlovian_with_bar', 'pavlovian_nobar_nodelay']:

            # self.triggers[0] refers to CS_Go triggering event most of the time whereas self.triggers[1] refers to CS_NoGo
            # find if spout event within timelim for go trials
                go_success = self.df_events.loc[
                    (self.df_events[self.df_events.trigger == self.triggers[0]].index),'spout_trial_time'].apply(
                    lambda x: find_if_event_within_timelim(x, self.timelim))
                go_success_idx = go_success[go_success == True].index
                # categorize successful go trials which have a spout event within timelim
                self.df_conditions.loc[(go_success_idx),'success'] = True

            # self.df_events.loc[(go_success_idx),'success'] = True

        # To perform for cued-uncued version of the go task
        elif self.task_name in ['reaching_go_spout_cued_uncued', 'cued_uncued_oct22']:
            # reformatting trigger name for that one task, with lower case
            if self.task_name in ['cued_uncued_oct22']:
                self.df_conditions.trigger = self.df_conditions.trigger.str.lower()
                self.df_events.trigger = self.df_events.trigger.str.lower()

            # for cued trials, find if spout event within timelim           
            cued_success = self.df_events.loc[
                (self.df_events[self.df_conditions.trigger == 'cued'].index),'spout_trial_time'].apply(
                lambda x: find_if_event_within_timelim(x, self.timelim))
            cued_success_idx = cued_success[cued_success == True].index

            # for uncued trials, just check if there is a spout event after trial start
            uncued_success = self.df_events.loc[
                (self.df_events[self.df_conditions.trigger == 'uncued'].index),'spout_trial_time'].apply(
                lambda x: x[-1] > 0 if len(x) > 0 else False)
            uncued_success_idx = uncued_success[uncued_success == True].index
            
            # categorize successful go trials
            self.df_conditions.loc[np.hstack((cued_success_idx.values, uncued_success_idx.values)), 'success'] = True
            # self.df_events.loc[np.hstack((cued_success_idx.values, uncued_success_idx.values)),'success'] = True
            print(self.task_name, self.subject_ID, self.datetime_string, len(cued_success_idx), len(uncued_success_idx))


        elif self.task_name in ['reaching_go_spout_nov22']:
            reach_time_before_reward = self.df_events.loc[:,['spout_trial_time','US_end_timer_trial_time']].apply(
                lambda x: find_last_time_before_list(x['spout_trial_time'], x['US_end_timer_trial_time']), axis=1)    
            # select only trials with a spout event before a US_end_timer event
            reach_bool = reach_time_before_reward.notnull()
            # select trial where the hold time was present (not aborted)
            reach_success_bool = reach_bool & self.df_conditions.busy_win
            # set these trials as successful
            self.df_conditions.loc[(reach_success_bool), 'success'] = True

        # To perform for delayed tasks (check whether a US_end_timer was preceded by a spout)
        elif self.task_name in ['reaching_go_spout_bar_nov22','reaching_go_spout_bar_mar23']:

            reach_time_before_reward = self.df_events.loc[:,['spout_trial_time','US_end_timer_trial_time']].apply(
                    lambda x: find_last_time_before_list(x['spout_trial_time'], x['US_end_timer_trial_time']), axis=1)    
            # select only trials with a spout event before a US_end_timer event
            reach_bool = reach_time_before_reward.notnull()
            # select trial where the hold time was present (not aborted)
            reach_success_bool = reach_bool & self.df_conditions.waiting_for_spout
            # set these trials as successful
            self.df_conditions.loc[(reach_success_bool), 'success'] = True
        
        elif self.task_name in ['reaching_go_spout_incr_break2_nov22']:

            reach_time_before_reward = self.df_events.loc[:,['spout_trial_time','US_end_timer_trial_time']].apply(
                    lambda x: find_last_time_before_list(x['spout_trial_time'], x['US_end_timer_trial_time']), axis=1)    
            # select only trials with a spout event before a US_end_timer event
            reach_bool = reach_time_before_reward.notnull()

            # set these trials as successful
            self.df_conditions.loc[(reach_bool), 'success'] = True

        elif self.task_name in ['reaching_go_spout_bar_dual_all_reward_dec22', 
            'reaching_go_spout_bar_dual_dec22']:

            reach_time_before_reward = self.df_events.loc[:,['spout_trial_time','US_end_timer_trial_time']].apply(
                    lambda x: find_last_time_before_list(x['spout_trial_time'], x['US_end_timer_trial_time']), axis=1)    
            # select only trials with a spout event before a US_end_timer event
            reach_bool = reach_time_before_reward.notnull()
            # select trial where the hold time was present (not aborted)
            reach_success_bool = reach_bool & self.df_conditions.Go_to_get_water
            # set these trials as successful
            self.df_conditions.loc[(reach_success_bool), 'success'] = True

        # # Reorder columns putting trigger, valid and success first for more clarity
        # col_list = list(self.df_conditions.columns.values)
        # col_to_put_first = ['trigger', 'success','valid']
        # for c in col_to_put_first:
        #     col_list.remove(c)
        # col_list = ['trigger', 'success','valid'] + col_list
        # self.df_conditions = self.df_conditions[col_list]

        

        return self

    def get_deeplabcut_trials(
            self,
            conditions_list = None,
            cond_aliases = None,
            camera_fps = 100,
            camera_keyword: str = None,
            dlc_scorer: str = None,
            bodyparts_to_ave = None, 
            names_of_ave_regions = None,
            normalize_between = None,
            bins_nb = 200,
            p_thresh = 0.6, 
            bodyparts_to_store = None, 
            trig_on_ev = None, 
            three_dims = False, 
            return_full_session = False, 
            verbose = False):
        
        if not isinstance(conditions_list, list):
            conditions_list= [conditions_list] 
        
        if not hasattr(self, 'files') or 'h5' not in self.files:
            raise Exception('The session has not been matched with a deeplabcut (.h5 / .nwb / .csv) file, \
                build an experimental object, then run <Exp_name>.match_sessions_to_files(files_dir, ext=''h5'')')

        # Check consistency of names_of_ave_regions and bodyparts_to_ave:
        if names_of_ave_regions or bodyparts_to_ave:
            if names_of_ave_regions and bodyparts_to_ave:
                if len(names_of_ave_regions) != len(bodyparts_to_ave):
                    raise Exception('bodyparts_to_ave and names_of_ave_regions must be lists of the same length')
            elif names_of_ave_regions and not bodyparts_to_ave:
                raise Exception('You need indicate the bodyparts_to_ave -> List of str')
            elif not names_of_ave_regions and bodyparts_to_store:
                raise Exception('You need indicate names of the bodyparts_to_ave (names_of_ave_regions arguments -> List of str)')

        dlc_file_to_read = []
        for dlc_file in self.files['h5']:
            if search(camera_keyword, dlc_file):
                dlc_file_to_read.append(dlc_file)
        
        # Check how many DLC file are matching the video
        if len(dlc_file_to_read) == 0:
            raise DeepLabCutFileError(
                self.subject_ID, self.datime, camera_keyword)

        # TODO: implement a DLC network keyword argument 
        # for when there is more than one DLC file per vid 
        elif len(dlc_file_to_read) > 1:
            dlc_file_to_read = os.path.realpath(dlc_file_to_read[0])
            print(f' Warning: multiple DLC files matching {self.subject_ID} at {self.datetime} on camera {camera_keyword}: \n\r \
                will use {dlc_file_to_read}')
        
        # when one single DLC file match the video (normal case)
        else:
            dlc_file_to_read = dlc_file_to_read[0]

        # normalize file path format
        dlc_file_to_read = os.path.realpath(dlc_file_to_read)

        # load DLC data
        df_dlc = pd.read_hdf(dlc_file_to_read)
        if verbose:
            print(f'Successfully loaded DLC file: {os.path.split(dlc_file_to_read)[1]}')
        scorer = df_dlc.columns.get_level_values(0).unique().values[0]
        bodyparts = df_dlc.columns.get_level_values(1).unique().values.tolist()
        len_dlc = df_dlc.shape[0]

        for b in bodyparts:
            df_dlc.loc[:, (scorer, b, 'x')].mask(df_dlc.loc[:, (scorer, b, 'likelihood')] < p_thresh, inplace=True)
            df_dlc.loc[:, (scorer, b, 'y')].mask(df_dlc.loc[:, (scorer, b, 'likelihood')] < p_thresh, inplace=True)
            if three_dims:
                df_dlc.loc[:, (scorer, b, 'z')].mask(df_dlc.loc[:, (scorer, b, 'likelihood')] < p_thresh, inplace=True)

        coord_dict = dict()

        regions_to_store = get_regions_to_store(bodyparts_to_ave, names_of_ave_regions, bodyparts_to_store)
                
        if bodyparts_to_ave:
            for r_idx, r in enumerate(names_of_ave_regions):
                if three_dims:
                    coord_dict[r] = np.ndarray(shape=(len_dlc,3))
                else:
                    coord_dict[r] = np.ndarray(shape=(len_dlc,2))
                
                coord_dict[r][:,0] = df_dlc.loc[:, (scorer, bodyparts_to_ave[r_idx], 'x')].mean(axis=1).values
                coord_dict[r][:,1] = df_dlc.loc[:, (scorer, bodyparts_to_ave[r_idx], 'y')].mean(axis=1).values
                if three_dims:
                    coord_dict[r][:,2] = df_dlc.loc[:, (scorer, bodyparts_to_ave[r_idx], 'z')].mean(axis=1).values

        if bodyparts_to_store:
            for r_idx, r in enumerate(bodyparts_to_store):
                if three_dims:
                    coord_dict[r] = np.ndarray(shape=(len_dlc,3))
                else:
                    coord_dict[r] = np.ndarray(shape=(len_dlc,2))
                
                coord_dict[r][:,0] = df_dlc.loc[:, (scorer, r, 'x')].values
                coord_dict[r][:,1] = df_dlc.loc[:, (scorer, r, 'y')].values
                if three_dims:
                    coord_dict[r][:,2] = df_dlc.loc[:, (scorer, r, 'z')].values


        del df_dlc
     
        # store only the regions of interest
        coord_dict = {region: coord_dict[region] for region in regions_to_store}   

        if normalize_between:
            coord_dict = normalize_coords(coord_dict=coord_dict, normalize_betwen=['Left_paw','spout'], bins_nb=bins_nb)

        df_meta_dlc = pd.DataFrame(columns=['subject_ID', 'datetime', 'task_name', 'condition_ID', 'trial_nb'])
        
        col_names_numpy = [region + ax for region in regions_to_store for ax in ['_x','_y']]
        # Prepare dictionary output with keys are variable names and values are columns index
        col_names_numpy = {col_name: col_idx for col_idx, col_name in enumerate(col_names_numpy)}

        for condition_ID, conditions_dict in enumerate(conditions_list):
            
            
            trials_idx, trial_times = self.get_trials_times_from_conditions(
                conditions_dict=conditions_dict, trig_on_ev=trig_on_ev)
            idx_dlc = (trial_times / (1000/camera_fps)).round().astype(int) # in ms
            
            # TODO: Test when no trials in first or nth condition
            if trials_idx == []:
                continue

            # check if enough data left and right
            complete_mask = (idx_dlc + int(ceil(self.trial_window[0]/(1000/camera_fps))) >= 0) & \
                (idx_dlc + int(ceil(self.trial_window[1]/(1000/camera_fps))) < len_dlc)

            trials_idx = np.array(trials_idx)[complete_mask] 
            idx_dlc = np.array(idx_dlc)[complete_mask]

            if verbose:
                print(f'condition {condition_ID} trials: {len(trials_idx)}')

            if len(idx_dlc) == 0 :
                continue

            # Craft the ranges on which extract the data
            idx_dlc = [range(idx + round(self.trial_window[0]/ (1000/camera_fps)) ,
                idx + round(self.trial_window[1] / (1000/camera_fps))) for idx in idx_dlc]
            
            if condition_ID == 0:
                for r_idx, region in enumerate(coord_dict.keys()):
                    if r_idx == 0:
                        # print(f'condition {condition_ID} region {r_idx} shape {np.take(coord_dict[region], idx_dlc, axis=0).shape}')
                        dlc_array = np.take(coord_dict[region], idx_dlc, axis=0)
                    else:
                        # print(f'condition {condition_ID} region {r_idx} shape {np.take(coord_dict[region], idx_dlc, axis=0).shape}')
                        dlc_array = np.concatenate((dlc_array, np.take(coord_dict[region], idx_dlc, axis=0)), axis=2)

                df_meta_dlc['trial_nb'] = trials_idx
                df_meta_dlc['subject_ID'] = self.subject_ID
                df_meta_dlc['datetime'] = self.datetime
                df_meta_dlc['task_name'] = self.task_name
                df_meta_dlc['condition_ID'] = condition_ID

            else:
                
                for r_idx, region in enumerate(coord_dict.keys()):
                    if r_idx == 0:
                        # print(f'condition {condition_ID} region {r_idx} shape {np.take(coord_dict[region], idx_dlc, axis=0).shape}')
                        dlc_array_temp = np.take(coord_dict[region], idx_dlc, axis=0)
                    else:
                        # print(f'condition {condition_ID} region {r_idx} shape {np.take(coord_dict[region], idx_dlc, axis=0).shape}')
                        dlc_array_temp = np.concatenate((dlc_array_temp, np.take(coord_dict[region], idx_dlc, axis=0)), axis=2)

                df_meta_dlc_temp = pd.DataFrame(columns=['subject_ID', 'datetime', 'task_name', 'condition_ID', 'trial_nb'])

                df_meta_dlc_temp['trial_nb'] = trials_idx
                df_meta_dlc_temp['subject_ID'] = self.subject_ID
                df_meta_dlc_temp['datetime'] = self.datetime
                df_meta_dlc_temp['task_name'] = self.task_name
                df_meta_dlc_temp['condition_ID'] = condition_ID      
                # concatenate with previous dataframe
                df_meta_dlc = pd.concat([df_meta_dlc, df_meta_dlc_temp], axis=0, ignore_index=True)
                # concatenate with previous numpy array

                dlc_array = np.concatenate((dlc_array, dlc_array_temp), axis=0)
        
        if 'dlc_array' in locals():
            dlc_array = dlc_array.swapaxes(2,1)
        else:
            # would occur anyway without the previous check, 
            # avoid it happening spontaneously on return.
            # useless but could be use to convey extra information to calling method
            raise UnboundLocalError()

        return df_meta_dlc, col_names_numpy, dlc_array
                 
    def get_photometry_trials(self,
            conditions_list: list = None,
            cond_aliases: list = None,
            trial_window: list = None,
            trig_on_ev: str = None,
            last_before: str = None,
            high_pass: float = None, 
            low_pass: int = None, 
            median_filt: int = None, 
            motion_corr: bool = False, 
            df_over_f: bool = False,
            z_score: bool = False, # To be implemented
            downsampling_factor: int = None,
            return_full_session: bool = False, 
            export_vars: list = ['analog_1','analog_2'],
            verbose: bool = False):
            
        """
        Returns a dictionary containing photometry data for each trial that satisfies the specified conditions.

        
        Parameters
        ----------
        conditions_list : list, optional
            A list of conditions (dictionaries) used to filter the trials. A trial must satisfy all conditions to be selected. 
            Default is None.
        cond_aliases : list, optional
            A list of strings to rename the conditions of the experiment.
            Default is None.
        trial_window : list, optional
            A list of two floats specifying the time window to select relative to the trial onset.
            Default is None.
        trig_on_ev : str, optional
            The name of the event to use for trial alignment. 
            Default is None, which aligns trials to the trial onset.
        last_before : str, optional
            If specified, the last event with the given name before `trig_on_ev` is used for trial alignment.
            Default is None.
        high_pass : float, optional
            The high-pass filter cutoff frequency to use for the motion correction. 
            Default is None.
        low_pass : int, optional
            The low-pass filter cutoff frequency to use for the motion correction. 
            Default is None.
        median_filt : int, optional
            The size of the median filter to use for the motion correction. 
            Default is None.
        motion_corr : bool, optional
            If True, motion correction is performed using the high-pass, low-pass, and median filters.
            Default is False.
        df_over_f : bool, optional
            If True, computes the dF/F and exports the analog_1_df_over_f variable.
            Default is False.
        z_score : bool, optional (not implemented yet)
            If True, z-scores the data.
            Default is False.
        downsampling_factor : int, optional
            The factor by which to downsample the data.
            Default is None.
        return_full_session : bool, optional
            If True, returns the full session.
            Default is False.
        export_vars : list, optional
            A list of variables to export in photo_array
            Default is ['analog_1', 'analog_2'].
        verbose : bool, optional
            If True, prints the number of trials selected for each condition.
            Default is False.

        Returns
        -------
        df_meta_photo : pandas.DataFrame
            metadata dataframe containing info about each trial
        col_names_numpy : list
            list of strings containing the names of the variables in the numpy array
        photo_array : numpy.ndarray
            numpy array containing the photometry data for each trial 
        photometry_dict : dict, optional (if return_full_session == True)
            dictionary containing all the photometry data for the entire session

        Raises
        ------
        Exception
            If the session has not been matched with a .ppd file, or if the session has no matching .ppd file, or if no alignment 
            could be performed between rsync pulses.
        Exception
            If motion_corr is True and high_pass, low_pass, or median_filt is None.
        Exception
            If df_over_f is True and motion_corr is False.
        """

        if not isinstance(conditions_list, list):
            conditions_list= [conditions_list] 
        
        if not isinstance(export_vars, list):
            export_vars= [export_vars] 

        if not hasattr(self, 'photometry_rsync'):
            raise Exception('The session has not been matched with a .ppd file, \
                please run experiment.match_to_photometry_files(kvargs)')
        elif self.photometry_rsync == None:
            raise Exception('The session has no matching .ppd file, or no alignment \
                could be performed between rsync pulses')
        
        if motion_corr == True and high_pass == None and low_pass == None and median_filt == None:
            raise Exception('You need to high_pass and/or low_pass and/or median_filt the signal for motion correction')
        if df_over_f == True and motion_corr == False:
            raise Exception('You need motion correction to compute dF/F')
    
        # import of raw and filtered data from full photometry session

        photometry_dict = import_ppd(self.files['ppd'][0], high_pass=high_pass, low_pass=low_pass, medfilt_size=median_filt)

        #----------------------------------------------------------------------------------
        # Filtering / Motion correction / resampling block below
        #----------------------------------------------------------------------------------

        # TODO: verify/improve/complement the implementation of the following:
        # https://github.com/ThomasAkam/photometry_preprocessing/blob/master/Photometry%20data%20preprocessing.ipynb
        
        if motion_corr == True:
            photometry_dict = motion_correction(photometry_dict)

            if df_over_f == False:
                export_vars.append('analog_1_corrected')

            elif df_over_f == True:
                photometry_dict = compute_df_over_f(photometry_dict, low_pass_cutoff=0.01)
                export_vars.append('analog_1_df_over_f')
                if z_score == True:
                    photometry_dict['zscored_df_over_f'] = zscore(photometry_dict['analog_1_df_over_f'])
                    export_vars.append('zscored_df_over_f')

        elif high_pass or low_pass:
            export_vars.append('analog_1_filt')
        else:
            export_vars.append('analog_1')

        # only keep unique items (keys for the photometry_dict)
        export_vars = list(set(export_vars))

        if downsampling_factor:
            # downsample
            for k in export_vars:
                photometry_dict[k] = decimate(photometry_dict[k], downsampling_factor)
            # adjust sampling rate accordingly (maybe unnecessary)
            photometry_dict['sampling_rate'] = photometry_dict['sampling_rate'] / downsampling_factor

        fs = photometry_dict['sampling_rate']

        df_meta_photo = pd.DataFrame(columns=['subject_ID', 'datetime', 'task_name', 'condition_ID', 'trial_nb'])
        
        # Prepare dictionary output with keys are variable names and values are columns index
        col_names_numpy = {var: var_idx for var_idx, var in enumerate(export_vars)}
        
        # TODO: must be more effective to do that process a single time and then sort by / attribute conditions
        for condition_ID, conditions_dict in enumerate(conditions_list):
            # TEST the option of triggering on the first event of a trial

            trials_idx, timestamps_pycontrol = self.get_trials_times_from_conditions(conditions_dict, trig_on_ev=trig_on_ev, last_before=last_before)
            
            if len(trials_idx) == 0 :
                continue
            # assumes that sync between pycontrol and photometry has been performed in previous step
            timestamps_photometry = self.photometry_rsync.A_to_B(timestamps_pycontrol)
            photometry_idx = (timestamps_photometry / (1000/photometry_dict['sampling_rate'])).round().astype(int)

            # retain only trials with enough values left and right
            complete_mask = (photometry_idx + trial_window[0]/(1000/photometry_dict['sampling_rate']) >= 0) & (
                photometry_idx + trial_window[1] < len(photometry_dict[export_vars[0]])) 

            # complete_idx = np.where(complete_mask)
            trials_idx = np.array(trials_idx)
            photometry_idx = np.array(photometry_idx)

            trials_idx = trials_idx[complete_mask]           
            photometry_idx = photometry_idx[complete_mask]
            
            if verbose:
                print(f'condition {condition_ID} trials: {len(trials_idx)}')
            
            if len(trials_idx) == 0 :
                continue

            # Construct ranges of idx to get chunks (trials) of photometry data with np.take method 
            photometry_idx = [range(idx + int(trial_window[0]/(1000/photometry_dict['sampling_rate'])) ,
                idx + int(trial_window[1]/(1000/photometry_dict['sampling_rate']))) for idx in photometry_idx]

            if condition_ID == 0:
                # initialization of 3D numpy arrays (for all trials)
                # Dimensions are M (number of trials) x N (nb of samples by trial) x P (number of variables,
                # e.g.: analog_1_filt, analog_1_df_over_f, etc)

                photo_array = np.ndarray((len(trials_idx), len(photometry_idx[0]),len(export_vars)))

                for var_idx, photo_var in enumerate(export_vars):
                    # print(f'condition {condition_ID} var: {var_idx} shape {np.take(photometry_dict[photo_var], photometry_idx).shape}')
                    photo_array[:,:,var_idx] = np.take(photometry_dict[photo_var], photometry_idx)


                df_meta_photo['trial_nb'] = trials_idx
                df_meta_photo['subject_ID'] = self.subject_ID
                df_meta_photo['datetime'] = self.datetime
                df_meta_photo['task_name'] = self.task_name
                df_meta_photo['condition_ID'] = condition_ID

            else:
                # initialization of temp 3D numpy arrays (for subset of trials by conditions)
                # Dimensions are M (number of trials) x N (nb of samples by trial) x P (number of variables,
                # e.g.: analog_1_filt, analog_1_df_over_f, etc)

                photo_array_temp = np.ndarray((len(trials_idx), len(photometry_idx[0]),len(export_vars)))

                for var_idx, photo_var in enumerate(export_vars):
                    # print(f'condition {condition_ID} var: {var_idx} shape {np.take(photometry_dict[photo_var], photometry_idx).shape}')
                    photo_array_temp[:,:,var_idx]  = np.take(photometry_dict[photo_var], photometry_idx)


                df_meta_photo_temp = pd.DataFrame(columns=['subject_ID', 'datetime', 'task_name', 'condition_ID', 'trial_nb'])

                df_meta_photo_temp['trial_nb'] = trials_idx
                df_meta_photo_temp['subject_ID'] = self.subject_ID
                df_meta_photo_temp['datetime'] = self.datetime
                df_meta_photo_temp['task_name'] = self.task_name
                df_meta_photo_temp['condition_ID'] = condition_ID      
                # concatenate with previous dataframe
                df_meta_photo = pd.concat([df_meta_photo, df_meta_photo_temp], axis=0, ignore_index=True)
                # concatenate with previous numpy array
                
                # Take into account when there is no trial in first condition
                if 'photo_array' in locals():
                    photo_array = np.concatenate((photo_array, photo_array_temp), axis=0)
                else:
                    photo_array = photo_array_temp



        if 'photo_array' in locals():
            photo_array = photo_array.swapaxes(2,1)
        else:
            # This occurs when no photometry data is recored at all for the session
            # would occur anyway without the previous check, 
            # avoid it happening spontaneously on return.
            # useless but could be use to convey extra information to calling method
            
            if verbose:
                print(f'No photometry data to collect for subject ID:{self.subject_ID}\
                    \nsession: {self.datetime}')

            raise UnboundLocalError()

            # Trying to implement empty arrays and dataframe when nothing to return
            # df_meta_photo = pd.DataFrame(columns=['subject_ID', 'datetime', 'task_name', 'condition_ID', 'trial_nb'])
            # ra
            # photo_array = np.ndarray((len(trials_idx), len(photometry_idx),len(export_vars)))

        if return_full_session == False:
            return df_meta_photo, col_names_numpy, photo_array, fs
        else:
            return df_meta_photo, col_names_numpy, photo_array, photometry_dict

    # TODO: implement for analog data like piezzo, threadmill, etc.
    def get_analog_trial():
        ...
    
    def get_trials_times_from_conditions(
            self,
            conditions_dict: dict = None, 
            trig_on_ev: str = None,
            last_before: str = None, 
            output_first_ev: bool = False):
        '''
        Get the indices and timestamps of the trials matching a set dict of conditions,
        offsetted (or not) by the first (or last before) occurence of a particular event
        '''
        if conditions_dict is not None:
            cond_pairs = list(conditions_dict.items())
            idx_rows = []

            for idx_cond, cond in enumerate(cond_pairs):
                # create a list of set of index values for each conditions
                idx_rows.append(set(self.df_conditions.index[self.df_conditions[cond_pairs[idx_cond][0]] == cond_pairs[idx_cond][1]].values))

            # compute the intersection of the indices of all conditions requested
            idx_joint = list(set.intersection(*idx_rows))
            idx_joint.sort()
        else:
            idx_joint = self.df_conditions.index.values

        if output_first_ev and not isinstance(trig_on_ev, str):
            raise NotPermittedError('get_trials_times_from_conditions',
                ['trig_on_ev','output_first_ev'], [trig_on_ev,output_first_ev])

        # if no trig_on_ev specified, return the original trigger timestamp
        if trig_on_ev == None:
            trials_times = self.df_events.loc[(idx_joint),'timestamp'].values
        elif trig_on_ev not in self.events_to_process:
            raise Exception('trig_on_ev not in events_to_process')
        # Otherwise offset trigger timestamp by occurence of first event (TODO: or last before)
        else:    
            trials_times = self.df_events.loc[(idx_joint), 'timestamp'].copy()
            
            df_ev_copy = self.df_events.copy()
            # TODO: continue implementation
            if last_before is not None and last_before in set(self.events_to_process):

                if len(idx_joint) == 0: # Added because find_last_time_before_list did not do well with empty Df
                    ev_times = pd.DataFrame()
                else:
                    ev_col = trig_on_ev + '_trial_time'
                    before_col = last_before + '_trial_time'

                    ev_times = df_ev_copy.loc[(idx_joint), [ev_col, before_col]].apply(
                        lambda x: find_last_time_before_list(x[ev_col], x[before_col]), axis=1)               
                
            # If last_before is not requested
            else:

                ev_times = df_ev_copy.loc[(idx_joint), trig_on_ev + '_trial_time'].apply(
                    lambda x: find_min_time_list(x))
                

            #keep ony trial_times with a first event
            ev_times_nona = ev_times.dropna(inplace=False)
            
            # Add time of the first event to trial onset
            trials_times = trials_times.loc[ev_times_nona.index] + ev_times_nona
            # trials_times.dropna(inplace=True)
            # Keep only the index where events (trig_on_ev) were found
            idx_joint = trials_times.index.values.astype(int)
            # retransmorm trial_times as an array
            trials_times = trials_times.values.astype(int)
            # ev_times = ev_times_nona.astype(int).values
        
            #print(idx_joint.shape,first_ev_times_nona.shape, first_ev_times.shape)
        # if output_first_ev:
        #     return idx_joint, trials_times
        # else:    
        return idx_joint, trials_times

    def compute_behav_metrics(self, conditions_dict, events=None):
        ''' 
        Early function, decide whether to keep or not
        Most likely to deprecate

        if two events are passed, time_delta between the first occurence of events[0]
        "t0(events[0])" and the first occurence of events[1] *after t0(events[0])*
        will be computed. Results will be stored in mean_timedelta and std_timedelta

        '''

        if events == 'all':
            events = self.events_to_process

        if events:
            if isinstance(events,str):
                events = [events]
        
            if any([ev not in self.events_to_process for ev in events]):
                raise Exception('One or more requested events are not belonging to the "events_to_process"')

        idx_cond, trial_times = self.get_trials_times_from_conditions(conditions_dict=conditions_dict)

        # if behaviour file too short or not analyzed
        if self.analyzed == False:
            print('behavioural metrics impossible to calculate, file is too short or not extracted by trials:', 
                self.subject_ID, self.datetime_string, self.task_name)
            pass
        else:
            # Create a 'behav' dictionary to store behavioural metrics
            self.behav = dict()
            self.behav['condition'] = conditions_dict
            self.behav['num_trials'] = len(idx_cond)

            try:
                if 'trigger' in list(conditions_dict.keys()):
                    success_rate = len(self.df_conditions[(self.df_conditions['valid'] == True) & (self.df_conditions['success'] == True) & (self.df_conditions['trigger'] == conditions_dict['trigger'])].index) \
                        / len(self.df_conditions[(self.df_conditions['valid'] == True) & (self.df_conditions['trigger'] == conditions_dict['trigger'])].index)
                else:
                                success_rate = len(self.df_conditions[(self.df_conditions['valid'] == True) & (self.df_conditions['success'] == True)].index) \
                        / len(self.df_conditions[(self.df_conditions['valid'] == True)].index)
            except (ZeroDivisionError):
                success_rate = np.NaN

            # append value to behav dict
            self.behav['success_rate'] = success_rate
            
            if events is not None:
                # event specific computations
                for ev in events:

                    # compute the mean and std of the event times AFTER (>0) trial initiation
                    mean_min_time = self.df_events[ev + '_trial_time'].loc[idx_cond].apply(
                        lambda x: min([i for i in x if i>0], default=np.NaN)).mean()
                    std_min_time = self.df_events[ev + '_trial_time'].loc[idx_cond].apply(
                        lambda x: min([i for i in x if i>0], default=np.NaN)).std()
                    # for EACH trial, count how many events occurs after t0
                    num_min_time = self.df_events[ev + '_trial_time'].loc[idx_cond].apply(
                        lambda x: np.count_nonzero(~isnan([i for i in x if i>0])))
                    # HOW MANY trial have at least one events after t0
                    num_min_time = np.count_nonzero(num_min_time)
                    # I expect to see RuntimeWarnings in this block
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning) 
                        self.behav['mean_min_time_'+ ev] = mean_min_time
                        self.behav['std_min_time_'+ ev] = std_min_time
                        self.behav['num_min_time_'+ ev] = num_min_time

                # compute timedelta measurements between first occurences of first and second events
                if len(events) == 2:

                    col_idx_start = list(self.df_events.columns.values).index(events[0] + '_trial_time') + 1 # +1 since index is the first value of the row from itertuples()
                    col_idx_end = list(self.df_events.columns.values).index(events[1] + '_trial_time') + 1 # +1 since index is the first value of the row from itertuples()
                    time_intervals = []
                    for row in self.df_events.loc[idx_cond].itertuples():
                        time_intervals.append(time_delta_by_row(row, col_idx_start, col_idx_end))
                    
                    # I expect to see RuntimeWarnings in this block
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)    
                        self.behav['mean_timedelta'] = np.nanmean(time_intervals)
                        self.behav['std_timedelta'] = np.nanstd(time_intervals)
                        self.behav['num_timedelta'] = np.count_nonzero(time_intervals)
        
                    # Make a single list of all events to compute distribution
                    # list_of_list = self.df_events.loc[(self.df_conditions['valid'] == True) & (self.df_conditions['trigger'] == trig), ev + '_trial_time'].values
                    # flat_list_times = [item for sublist in list_of_list for item in sublist]
        return self

    def plot_session(self, keys: list = None, state_def: list = None, print_expr: list = None, 
                    event_ms: list = None, export_smrx: bool = False, smrx_filename: str = None, verbose :bool = False,
                    print_to_text: bool = True, vchange_to_text: bool = True):
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
                'onset' : str | list of str 
                    key for onset 
                'offset' : str | list of str 
                    key for offset
            or list of such dictionaries

            eg. dict(name='trial', onset='CS_Go', offset='refrac_period')
            eg. {'name':'trial', 'onset':'CS_Go', 'offset':'refrac_period'}
            eg. {'name':'trial', 'onset':'CS_Go', 'offset': ['refrac_period', 'break_after_abortion']}


            For each onset, finds the first offset event before the next onset
            You can use multiple definitions with OR operation, eg. 'offset' determined by 'abort' or 'success', whichever comes first
        
        print_expr: list of dict #TODO need more testing
            'name':'name of channel'
            'expr': The expression '^\d+(?= ' + expr + ')' will be used for re.match()
            list of regular expressions to be searched for self.print_lines and shown as an event channel

            eg. {
                'name':'water success',
                'expr':'.?water success' # .? is needed if it is unknown whether there is any character ahead        
            }

        event_ms: list of dict
                'name':'name of something'
                'time_ms': X
            allow plotting timestamps as an event

        state_ms: list of dict #TODO

        export_smrx: Bool = False
            Save the plotted channels to a Spike 2 .smrx file.
            An event channel and a state channel will be represetnted as an event and marker channels.
            For the latter, onset and offset of a state is coded by 1 and 0 for code0.
            Use
                pip install sonpy
            to install the sonpy module.

            This metthod seems unstable. Tha same session data may fail ot succedd to export Spike2 file. Try restating kernel a few times. 
            Apparently addition of time.sleep(0.05) helped to make this more stable.
            Use verbose option to see what's going on.
            When failed, the file size tends to be 11KB. Verbose will show [-1].
            Restart the kernel to delete the corrupeted .smrx file.

            Stylise the Spike2 display using notebooks|noncanonical|display_style.s2s

        smrx_filename: str = None

        verbose :bool = False

        print_to_text: Bool = True
            print_lines will be converted to text (and TextMark channel in Spike2)

        vchange_to_text: Bool = True
            Variable changes during the session, eg. "V 12560 windor_dur_ms 3000", will be converted to text (and TextMark channel in Spike2)

        """

        # see  \Users\phar0528\Anaconda3\envs\trialexp\Lib\site-packages\sonpy\MakeFile.py
        #NOTE cannot put file path in the pydoc block

        raw_symbols  = SymbolValidator().values
        symbols = [raw_symbols[i+2] for i in range(0, len(raw_symbols), 12)]
        # 40 symbols

        fig = go.Figure()
        if keys is None:
            keys = self.times.keys()
        else:
            for k in keys: 
               assert k in self.times.keys(), f"{k} is not found in self.time.keys()"

        if export_smrx:
            import time

            from sonpy import lib as sp
            if smrx_filename is None:
                raise Exception('smrx_filename is required')
            #TODO assert .smlx

            mtc = re.search('\.smrx$', smrx_filename)
            if mtc is None:
                raise Exception('smrx_filename has to end with .smrx')

            MyFile = sp.SonFile(smrx_filename, nChans = int(400)) #NOTE int() is required
            #NOTE nChans = ctypes.c_uint16(400) # TypeError
            #NOTE nChans = 400 # MyFile.MaxChannels() = -1
            #NOTE sonpy 1.9.5 works with nChans (1.8.5. doesn't)
            CurChan = 0
            UsedChans = 0
            Scale = 65535/20
            Offset = 0
            ChanLow = 0
            ChanHigh = 5
            tFrom = 0
            tUpto = sp.MaxTime64()         # The maximum allowed time in a 64-bit SON file
            dTimeBase = 1e-6               # s = microseconds
            x86BufSec = 2.
            EventRate = 1/(dTimeBase*1e3)  # Hz, period is 1000 greater than the timebase
            SubDvd = 1                     # How many ticks between attached items in WaveMarks

            times_ = [np.max(self.times[k]) for k in keys if any(self.times[k])]
            if times_ == []:
                raise Exception('No time stamp found: Cannot determine MaxTime()')

            else:
                max_time_ms1 = np.max(times_) #TODO ValueError when np.max([]) 

                list_of_match = [re.match('^\d+', L) for L in self.print_lines if re.match('^\d+', L) is not None]
                max_time_ms2 = np.max([int(m.group(0)) for m in list_of_match])

                max_time_ms = np.max([max_time_ms1, max_time_ms2])
                time_vec_ms = np.arange(0, max_time_ms, 1000/EventRate)
                # time_vec_micros = np.arange(0, max_time_ms*1000, 10**6 * 1/EventRate)

                samples_per_s = EventRate
                interval = 1/samples_per_s

                samples_per_ms = 1/1000 * EventRate
                interval = 1/samples_per_s

                MyFile.SetTimeBase(dTimeBase)  # Set timebase


        def write_event(MyFile, X_ms, title, y_index, EventRate, time_vec_ms):
            (hist, ___) = np.histogram(X_ms, bins=time_vec_ms) # time is 1000 too small

            eventfalldata = np.where(hist)

            MyFile.SetEventChannel(y_index, EventRate)
            MyFile.SetChannelTitle(y_index, title)
            if eventfalldata[0] is not []:
                MyFile.WriteEvents(int(y_index), eventfalldata[0]*1000) #dirty fix but works
                time.sleep(0.05)# might help?

            if verbose:
                print(f'{y_index}, {title}:')
                nMax = 10
                # nMax = int(MyFile.ChannelMaxTime(int(y_index))/MyFile.ChannelDivide(int(y_index))) 
                print(MyFile.ReadEvents(int(y_index), nMax, tFrom, tUpto)) #TODO incompatible function arguments.
                # [-1] when failed

                # ReadEvents(self: sonpy.amd64.sonpy.SonFile, 
                #     chan: int, 
                #     nMax: int, # probably the end of the range to read in the unit of number of channel divide
                #     tFrom: int, 
                #     tUpto: int = 8070450532247928832, 
                #     Filter: sonpy.amd64.sonpy.MarkerFilter = <sonpy.MarkerFilter> in mode 'First', with trace column -1 and items
                #     Layer 1 [
                #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        def write_marker_for_state(MyFile,X_ms, title, y_index, EventRate, time_vec_ms):

            # remove NaN
            X_notnan_ms = [x for x in X_ms if not np.isnan(x)]

            (hist, ___) = np.histogram(X_notnan_ms, bins=time_vec_ms) # time is 1000 too small

            eventfalldata = np.where(hist)

            nEvents = len(eventfalldata[0])

            MarkData = np.empty(nEvents, dtype=sp.DigMark)
            for i in range(nEvents):
                if (i+1) % 2 == 0:
                    MarkData[i] = sp.DigMark(eventfalldata[0][i]*1000, 0) #offset
                elif (i+1) % 2 == 1:
                    MarkData[i] = sp.DigMark(eventfalldata[0][i]*1000, 1) #onset
                else:
                    raise Exception('oh no')
            MyFile.SetMarkerChannel(y_index, EventRate)
            MyFile.SetChannelTitle(y_index, title)
            if eventfalldata[0] is not []:
                MyFile.WriteMarkers(int(y_index), MarkData)
                time.sleep(0.05)# might help?

            if verbose:             
                print(f'{y_index}, {title}:')
                print(MyFile.ReadMarkers(int(y_index), nEvents, tFrom, tUpto)) #TODO failed Tick = -1

        def write_textmark(MyFile, X_ms, title, y_index, txt, EventRate, time_vec_ms):

            (hist, ___) = np.histogram(X_ms, bins=time_vec_ms) # time is 1000 too small

            eventfalldata = np.where(hist)

            nEvents = len(eventfalldata[0])

            MarkData = np.empty(nEvents, dtype=sp.DigMark)

            TMrkData = np.empty(nEvents, dtype=sp.TextMarker)

            for i in range(nEvents):
                if (i+1) % 2 == 0:
                    MarkData[i] = sp.DigMark(eventfalldata[0][i]*1000, 0) #offset
                elif (i+1) % 2 == 1:
                    MarkData[i] = sp.DigMark(eventfalldata[0][i]*1000, 1) #onset
                else:
                    raise Exception('oh no')
                
                #NOTE Spike2 truncates text longer than 79 characters???
                TMrkData[i] = sp.TextMarker(re.sub('\n', '', txt[i]), MarkData[i])

            if len(txt) == 0:
                MyFile.SetTextMarkChannel(y_index, EventRate, 32)
            else:
                MyFile.SetTextMarkChannel(y_index, EventRate, max(len(s) for s in txt)+1)
            MyFile.SetChannelTitle(y_index, title)
            if eventfalldata[0] is not []:
                MyFile.WriteTextMarks(y_index, TMrkData)
                time.sleep(0.05)# might help?

            if verbose:
                print(f'{y_index}, {title}:')
                try:
                    print(MyFile.ReadTextMarks(int(y_index), nEvents, tFrom, tUpto))
                except:
                    print('error in print')

        def find_states(state_def_dict: dict):
            """
            state_def: dict, list, or None = None
            must be None (default)
            or dictionary of 
                'name' : str 
                    Channel name
                'onset' : str | list of str 
                    key for onset 
                'offset' : str | list of str 
                    key for offset
            or list of such dictionaries

            eg. dict(name='trial', onset='CS_Go', offset='refrac_period')
            eg. {'name':'trial', 'onset':'CS_Go', 'offset':'refrac_period'}
            eg. {'name':'trial', 'onset':'CS_Go', 'offset': ['refrac_period', 'break_after_abortion']}

            For each onset, find the first offset event before the next onset 
            You can use multiple definitions with OR operation, eg. 'offset' determined by 'abort' or 'success', whichever comes first            
            """
            if state_def_dict is None:
                return None

            if isinstance(state_def_dict['onset'], str):
                all_on_ms = self.times[state_def_dict['onset']]
            elif isinstance(state_def_dict['onset'], list):
                # OR operation
                all_on_ms = []
                for li in state_def_dict['onset']:
                    assert isinstance(li, str), 'onset must be str or list of str'
                    all_on_ms.extend(self.times[li])
                all_on_ms = sorted(all_on_ms)
                
            else:
                raise Exception("onset is in a wrong type") 

            if isinstance(state_def_dict['offset'], str):
                all_off_ms = self.times[state_def_dict['offset']]
            elif isinstance(state_def_dict['offset'], list):
                # OR operation
                all_off_ms = []
                for li in state_def_dict['offset']:
                    assert isinstance(li, str), 'offset must be str or list of str'                    
                    all_off_ms.extend(self.times[li])
                all_off_ms = sorted(all_off_ms)
            else:
                raise Exception("offset is in a wrong type") 

            onsets_ms = [np.NaN] * len(all_on_ms)
            offsets_ms = [np.NaN] * len(all_on_ms)

            for i, this_onset in enumerate(all_on_ms):  # slow
                good_offset_list_ms = []
                for j, _ in enumerate(all_off_ms):
                    if i < len(all_on_ms)-1:
                        if all_on_ms[i] < all_off_ms[j] and all_off_ms[j] < all_on_ms[i+1]:
                            good_offset_list_ms.append(all_off_ms[j])
                    else:
                        if all_on_ms[i] < all_off_ms[j]:
                            good_offset_list_ms.append(all_off_ms[j])

                if len(good_offset_list_ms) > 0:
                    onsets_ms[i] = this_onset
                    offsets_ms[i] = good_offset_list_ms[0]
                else:
                    ...  # keep them as nan

            onsets_ms = [x for x in onsets_ms if not np.isnan(x)]  # remove nan
            offsets_ms = [x for x in offsets_ms if not np.isnan(x)]

            state_ms = map(list, zip(onsets_ms, offsets_ms,
                           [np.NaN] * len(onsets_ms)))
            # [onset1, offset1, NaN, onset2, offset2, NaN, ....]
            state_ms = [item for sublist in state_ms for item in sublist]
            return state_ms

        y_index = 0
        for kind, k in enumerate(keys):
            y_index += 1
            line1 = go.Scatter(x=self.times[k]/1000, y=[k]
                        * len(self.times[k]), name=k, mode='markers', marker_symbol=symbols[y_index % 40])
            fig.add_trace(line1)

            if export_smrx:
                write_event(MyFile, self.times[k], k, y_index, EventRate, time_vec_ms)



        if print_expr is not None: #TODO
            if isinstance(print_expr, dict):
                print_expr = [print_expr]

            for dct in print_expr:
                y_index += 1
                expr = '^\d+(?= ' + dct['expr'] + ')'
                list_of_match = [re.match(expr, L) for L in self.print_lines if re.match(expr, L) is not None]
                ts_ms = [int(m.group(0)) for m in list_of_match]
                line2 = go.Scatter(
                    x=[TS_ms/1000 for TS_ms in ts_ms], y=[dct['name']] * len(ts_ms), 
                    name=dct['name'], mode='markers', marker_symbol=symbols[y_index % 40])
                fig.add_trace(line2)

                if export_smrx:
                    write_event(
                        MyFile, ts_ms, dct['name'], y_index, EventRate, time_vec_ms)

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

                if export_smrx:
                    write_event(
                        MyFile, dct['time_ms'], dct['name'], y_index, EventRate, time_vec_ms)

        if print_to_text:

            EXPR = '^(\d+)\s(.+)' #NOTE . doesn't capture \n and re.DOTALL is required below
            list_of_match = [re.match(EXPR, L, re.DOTALL) for L in self.print_lines if re.match(EXPR, L) is not None]
            ts_ms = [int(m.group(1)) for m in list_of_match]
            txt = [m.group(2) for m in list_of_match]
  
            # df_print = pd.DataFrame(list(zip(ts_ms, txt)), columns=['ms', 'text'])

            y_index += 1
            txtsc = go.Scatter(x=[TS_ms/1000 for TS_ms in ts_ms], y=['print_lines']*len(ts_ms), 
                text=txt, textposition="top center", 
                mode="markers", marker_symbol=symbols[y_index % 40])
            fig.add_trace(txtsc)

            if export_smrx:
                write_textmark( MyFile, ts_ms, 'print lines', y_index, txt, EventRate, time_vec_ms)

        if vchange_to_text:
            EXPR = '^([1-9]\d*)\s(.+)' #NOTE Need to ignore the defaults (V 0 ****)
            list_of_match = [re.match(EXPR, L) for L in self.v_lines if re.match(EXPR, L) is not None]
            ts_ms = [int(m.group(1)) for m in list_of_match]
            txt = [m.group(2) for m in list_of_match]
  
            # df_print = pd.DataFrame(list(zip(ts_ms, txt)), columns=['ms', 'text'])

            y_index += 1
            txtsc = go.Scatter(x=[TS_ms/1000 for TS_ms in ts_ms], y=['V changes']*len(ts_ms), 
                text=txt, textposition="top center", 
                mode="markers", marker_symbol=symbols[y_index % 40])
            fig.add_trace(txtsc)

            if export_smrx:
                write_textmark( MyFile, ts_ms, 'V changes', y_index, txt, EventRate, time_vec_ms)
        

        if state_def is not None:
            # Draw states as gapped lines
            # Assuming a list of lists of two names

            if isinstance(state_def, dict):# single entry
                state_def = [state_def]
                # state_ms = find_states(state_def)

                # line1 = go.Scatter(x=[x/1000 for x in state_ms], y=[state_def['name']] * len(state_ms), 
                #     name=state_def['name'], mode='lines', line=dict(width=5))
                # fig.add_trace(line1)

            if isinstance(state_def, list):# multiple entry
                state_ms = None
                for i in state_def:
                    assert isinstance(i, dict)
                    
                    y_index +=1
                    state_ms = find_states(i)

                    line1 = go.Scatter(x=[x/1000 for x in state_ms], y=[i['name']] * len(state_ms), 
                        name=i['name'], mode='lines', line=dict(width=5))
                    fig.add_trace(line1)

                    if export_smrx:
                        write_marker_for_state(MyFile, state_ms, i['name'], y_index, EventRate, time_vec_ms)
            else:
                state_ms = None
        else:
            state_ms = None
             

        fig.update_xaxes(title='Time (s)')
        fig.update_yaxes(fixedrange=True) # Fix the Y axis

        fig.update_layout(
            
            title =dict(
                text = f"{self.task_name}, {self.subject_ID} #{self.number}, on {self.datetime_string} via {self.setup_ID}"
            )
        )

        fig.show()

        if export_smrx:
            del MyFile
            #NOTE when failed to close the file, restart the kernel to delete the corrupted file(s)
            print(f'saved {smrx_filename}')

    # Implemented in Event_dataset(), in trial_dataset_classes but left here for convenience as well
    def plot_trials_events(self, events_to_plot:list = 'all',  sort:bool = False):

        # I dont get that K, review symbol selection? 
        raw_symbols  = SymbolValidator().values
        symbols = [raw_symbols[i+2] for i in range(0, len(raw_symbols), 12)]

        event_cols = [event_col for event_col in self.events_to_process]
        event_names = [event_col.split('_trial_time')[0] for event_col in event_cols]

        if events_to_plot == 'all':
            events_to_plot = self.events_to_process
        
        elif isinstance(events_to_plot, list):

            # check if events requested exist
            check = all(ev in event_names for ev in events_to_plot)

            if not check:
                raise Exception('Check your list of requested events, event not found')
            
            event_cols = [ev + '_trial_time' for ev in events_to_plot]
            event_names = events_to_plot

        elif isinstance(events_to_plot, str):
            if events_to_plot not in event_names:
                raise Exception('Check the name of your requested event, event not found')
            
            event_cols = [events_to_plot + '_trial_time']
            event_names = [events_to_plot]

        else:
            raise Exception('bad format for requesting plot_trials events')

        # Implement this as abstract method to check requested arguments (events) match the session obj.

        plot_names =  [trig + ' ' + event for event in event_cols for trig in self.triggers]

        # https://plotly.com/python/subplots/
        # https://plotly.com/python/line-charts/
        fig = make_subplots(
            rows= len(event_cols), 
            cols= len(self.triggers), 
            shared_xaxes= True,
            shared_yaxes= True,
            subplot_titles= plot_names
        )

        for trig_idx, trigger in enumerate(self.df_events.trigger.unique()):
            
            # sub-selection of df_events based on trigger, should be condition for event_dataset class
            df_subset = self.df_events[self.df_events.trigger == trigger]


            for ev_idx, event_col in enumerate(event_cols):
                # if sort:
                #     min_times = df_subset[event_cols[ev_idx]].apply(lambda x: find_min_time_list(x))
                #     min_times = np.sort(min_times)

                ev_times = df_subset[event_cols[ev_idx]].apply(lambda x: np.array(x)).values
                ev_trial_nb = [np.ones(len(array)) * df_subset.index[idx] for idx, array in enumerate(ev_times)]

                ev_trial_nb = np.concatenate(ev_trial_nb)
                ev_times =  np.concatenate(ev_times)

                fig.add_shape(type="line",
                    x0=0, y0=1, x1=0, y1= ev_trial_nb.max(),
                    line=dict(
                    color="Grey",
                    width=2,
                    dash="dot"
                    ),
                    row= ev_idx+1,
                    col = trig_idx+1)

                fig.add_trace(
                    go.Scatter(
                        x= ev_times/1000,
                        y= ev_trial_nb,
                        name= event_names[ev_idx],
                        mode='markers',
                        marker_symbol=symbols[ev_idx % 40]
                        ),
                        row= ev_idx+1,
                        col = trig_idx+1)

                    

                fig.update_xaxes(
                    title_text = 'time (s)',
                    ticks = 'outside',
                    ticklen = 6,
                    tickwidth = 2,
                    tickfont_size = 12,
                    showline = True,
                    linecolor = 'black',
                    # range=[self.trial_window[0]/1000, self.trial_window[1]/1000]
                    autorange = True,
                    row = ev_idx+1,
                    col = trig_idx+1
                    )
                
                fig.update_yaxes( 
                    title_text = 'trial nb', 
                    ticks = 'outside',
                    ticklen = 6,
                    tickwidth = 2,   
                    tickfont_size = 12,
                    showline = True,
                    linecolor = 'black',
                    range = [1, ev_trial_nb.max()],
                    showgrid=True,
                    row = ev_idx+1,
                    col = trig_idx+1
                    )

        fig.update_layout(
            title_text= f'Events Raster plot, ID:{self.subject_ID} / {self.task_name} / {self.datetime_string}',
            height=800,
            width=800
                        
        )

        fig.show()

#----------------------------------------------------------------------------------
# Experiment class
#----------------------------------------------------------------------------------

class Experiment():
    """
   
    Attributes
    ----------
    folder_name : str
        Folder name
    path : str
        Path of data folder or file including folder_name
    sessions : list
        List of Session objects
    by_trial : bool
    trial_window : list
        eg [-2000, 6000]

    Properties
    ----------
    subject_IDs : int
    n_subjects : scalar integer
    task_names
    sessions_per_subject

    Methods
    -------
    behav_events_to_dataset, 
    get_deeplabcut_groups, 
    get_photometry_groups, 
    get_sessions, 
    match_sessions_to_files, 
    match_to_photometry_files, 
    plot, 
    process_exp_by_trial,
    save
            
    """

    def __init__(
            self, 
            path: str, 
            int_subject_IDs: bool = True, 
            update: bool = False,
            folder_by_session: bool = False, 
            verbose: bool = False):
        """
        Import all sessions from specified folder to create experiment object.  Only sessions in the 
        specified folder (not in subfolders) will be imported.
        
        Arguments
        ---------
        path:           Path of data folder or file
        int_subject_IDs:    If True subject IDs are converted to integers, e.g. m012 is converted to 12.
        update:             If True, do not rely only on .pkl file but check for new files
        verbose:            If True, output verbose on the status of the pycontrol files    
        """
        # If data organization based on session's folders
        self.folder_by_session = folder_by_session

        if os.path.isfile(path):
            if path[-4:] == '.pkl':
                self.path = Path(path)
                with open(path,'rb') as sessions_file:
                    self._sessions = pickle.load(sessions_file)
            # for single session Experiment
            elif path[-4:] == '.txt':
                self._sessions = [(Session(path, int_subject_IDs))]
                self.path = Path(path)
            else:
                raise(Exception('path argument should be either:\
                    \n- a folder containing pycontrol .txt files\
                    \n- a sessions.pkl file\
                    \n- a single pycontrol .txt file'))

        # if path is a folder            
        elif os.path.isdir(path):
            self.path = Path(path)

            if folder_by_session:
                
                path = Path(path)
                session_folder_list = os.listdir(path)
                # check if all are folders
                is_folder = [os.path.isdir(path / s_path) for s_path in session_folder_list]
                # only integrate folders
                session_folder_list = list(compress(session_folder_list, is_folder))

                pycontrol_folders = [path / s_f / 'pycontrol' for s_f in session_folder_list]
                pycontrol_files = [f / os.listdir(f)[0] for f in pycontrol_folders if
                    (len(os.listdir(f)) == 1) and (os.listdir(f)[0][-4:] == '.txt')]


            # Import sessions.

            self._sessions = []

            
            try: # Load sessions from saved sessions.pkl file.
                with open(self.path / 'sessions.pkl','rb') as sessions_file:
                    self._sessions = pickle.load(sessions_file)
                print('Saved sessions loaded from: sessions.pkl')
                # TODO: precise and refine use of this by_trial attribute
                self.by_trial = True
            except IOError:
                self.by_trial = False
                pass
        

            if update:
                if not folder_by_session:
                    old_files = [session.file_name for session in self._sessions]
                    files = os.listdir(self.path)
                    new_files = [f for f in files if f[-4:] == '.txt' and f not in old_files]

                    for file_name in new_files:
                        try:
                            self._sessions.append(Session(Path(self.path) / file_name, int_subject_IDs))
                        except Exception as error_message:
                            if verbose:
                                print('Unable to import file: ' + file_name)
                                print(error_message)
                
                if folder_by_session:
                    new_files = pycontrol_files

                    for file_name in new_files:
                        try:
                            self._sessions.append(Session(file_name), int_subject_IDs)
                        except Exception as error_message:
                            if verbose:
                                print('Unable to import file: ' + file_name)
                                print(error_message)

                self.sessions = self._sessions # force to call the setter

        else:
            raise(NotImplementedError)

        # Assign session numbers.

        self.subject_IDs = list(set([s.subject_ID for s in self.sessions]))
        self.n_subjects = len(self.subject_IDs)

        self.task_names = list(set([s.task_name for s in self.sessions]))

        self.sessions.sort(key = lambda s:s.datetime_string + str(s.subject_ID))
        
        self.sessions_per_subject = {}
        for subject_ID in self.subject_IDs:
            subject_sessions = self.get_sessions(subject_IDs=subject_ID)

            for i, session in enumerate(subject_sessions):
                session.number = i+1
                if verbose:
                    print('session nb: ', session.number, session.subject_ID, session.datetime_string, session.task_name)
            self.sessions_per_subject[subject_ID] = subject_sessions[-1].number

    @property
    def sessions(self):
        return self._sessions

    @sessions.setter
    def sessions(self,value):
        self.subject_IDs = list(set([s.subject_ID for s in value]))
        self.n_subjects = len(self.subject_IDs)

        self.task_names = list(set([s.task_name for s in value]))

        value.sort(key = lambda s:s.datetime_string + str(s.subject_ID))
        
        # self.sessions_per_subject = {}
        # for subject_ID in self.subject_IDs:
        #     subject_sessions = self.get_sessions(subject_IDs=subject_ID)

        #     for i, session in enumerate(subject_sessions):
        #         session.number = i+1 #TODO this will update session.number when Experiment.sessions is edited. Ideally this should be only done in __init__
        #         # if verbose:
        #         #    print('session nb: ', session.number, session.subject_ID, session.datetime_string, session.task_name)
        #     self.sessions_per_subject[subject_ID] = subject_sessions[-1].number       
        self._sessions = value
    
    @sessions.deleter
    def sessions(self):
        self.subject_IDs = []
        self.n_subjects = None
        self.task_names = []
        self.sessions_per_subject = {}
        del self._sessions


    def save(self, name: str = None):
        '''Save all sessions as .pkl file. Speeds up subsequent instantiation of 
        experiment as sessions do not need to be reimported from data files.
        
        Arguments:
            name: str
                Do not include '.pkl' in name, it is append automatically
                If not None, save the pickle file in its original folder as
                <name>.pkl
        ''' 
        if self.path.is_file():
            folder_path = self.path.parent
        elif self.path.is_dir():
            folder_path = self.path
        
        if name is not None:

            with open(folder_path / (name + '.pkl'),'wb') as sessions_file:
                pickle.dump(self.sessions, sessions_file)
                print(f"saved {folder_path / (name + '.pkl')}") # I think it's a good practice that whener you make changes to files/folders print something

        else: 
            
            with open(folder_path / 'sessions.pkl','wb') as sessions_file:
                pickle.dump(self.sessions, sessions_file)
                print(f"saved {folder_path / 'sessions.pkl'}") # I think it's a good practice that whener you make changes to files/folders print something


    # def match_photometry_files(self, photometry_dir):



    def get_sessions(self, subject_IDs='all', when='all', task_names='all'):
        '''Return list of sessions which match specified subject ID and time.  
        Arguments:
        subject_ID: Set to 'all' to select sessions from all subjects or provide a list of subject IDs.
        when      : Determines session number or dates to select, see example usage below:
                    when = 'all'      # All sessions
                    when = 1          # Sessions numbered 1
                    when = [3,5,8]    # Session numbered 3,5 & 8
                    when = [...,10]   # Sessions numbered <= 10
                    when = [5,...]    # Sessions numbered >= 5
                    when = [5,...,10] # Sessions numbered 5 <= n <= 10
                    when = '2017-07-07' # Select sessions from date '2017-07-07'
                    when = ['2017-07-07','2017-07-08'] # Select specified list of dates
                    when = [...,'2017-07-07'] # Select session with date <= '2017-07-07'
                    when = ['2017-07-01',...,'2017-07-07'] # Select session with '2017-07-01' <= date <= '2017-07-07'.
        '''
        
        if subject_IDs == 'all':
            subject_IDs = self.subject_IDs
        if not isinstance(subject_IDs, list):
            subject_IDs = [subject_IDs]

        if when == 'all': # Select all sessions.
            when_func = lambda session: True

        else:
            if type(when) is not list:
                when = [when]

            if ... in when: # Select a range..

                if len(when) == 3:  # Start and end points defined.
                    assert type(when[0]) == type(when[2]), 'Start and end of time range must be same type.'
                    if type(when[0]) == int: # .. range of session numbers.
                        when_func = lambda session: when[0] <= session.number <= when[2]
                    else: # .. range of dates.
                        when_func = lambda session: _toDate(when[0]) <= session.datetime.date() <= _toDate(when[2])
                
                elif when.index(...) == 0: # End point only defined.
                    if type(when[1]) == int: # .. range of session numbers.
                        when_func = lambda session: session.number <= when[1]
                    else: # .. range of dates.
                        when_func = lambda session: session.datetime.date() <= _toDate(when[1])

                else: # Start point only defined.
                    if type(when[0]) == int: # .. range of session numbers.
                        when_func = lambda session: when[0] <= session.number
                    else: # .. range of dates.
                        when_func = lambda session: _toDate(when[0]) <= session.datetime.date()
                
            else: # Select specified..
                assert all([type(when[0]) == type(w) for w in when]), "All elements of 'when' must be same type."
                if type(when[0]) == int: # .. session numbers.
                    when_func = lambda session: session.number in when
                else: # .. dates.
                    dates = [_toDate(d) for d in when]
                    when_func = lambda session: session.datetime.date() in dates
        
        # can select session based on task_name string or list of task_name
        if task_names == 'all' or task_names is None:
            task_names = self.task_names
        if not isinstance(task_names, list):
            task_names = [task_names]
        
        # select valid sessions subject/task/time specific
        valid_sessions = [s for s in self.sessions if 
            s.subject_ID in subject_IDs
            and when_func(s) and s.task_name in task_names]
        
        return valid_sessions


    def process_exp_by_trial(
            self,
            trial_window: list,
            timelim: list = None, # consider deprecating in new methods, that was mostly for old tasks and way of doing things
            tasksfile: str = None, # normally the full path of params\tasks_params.csv in this repo, won't work with None
            blank_spurious_event: list = None, 
            blank_timelim: list = [0, 60], 
            verbose = False):
        """
        ARGUMENTS
        ---------
        self
        trial_window: list,                
            eg [-2000, 6000]
            Time window as to trigger for trial-based fragmentationof data.
        timelim: list,                     
            eg [0 2000]
            Time window for determining success
        tasksfile: str                     
            full filepath of tasks_params.csv in this repo
        blank_spurious_event: list = None, 
            Name(s) of event as to which spurious events will be discared within blank_timelim
            this was used as sometimes solenoid activation was triggering 50Hz style detection / false detection
            on the bar or spout sensor as an electical artifact of some sort
        blank_timelim: list = [0, 60],     reflecting v.spout_detect_timeout
            Set time window to discard spurious events around blank_spurious_event
        verbose = False

        create emtpy list to store idx of sessions without trials,
        can be extended to detect all kind of faulty sessions.

        self.sessions[i].df_conditions must not be empty

        """
        # Should be the only definition of trial window (Experiment level)
        self.trial_window = trial_window


        sessions_idx_to_remove = []
        
        for s_idx, s in enumerate(self.sessions):

            self.sessions[s_idx] = s.get_session_by_trial(self.trial_window, timelim,
                tasksfile, blank_spurious_event, blank_timelim, verbose = verbose)
            
            # for files too short
            if self.sessions[s_idx].analyzed == False:
                sessions_idx_to_remove.append(s_idx)
            # for files passing previous check but no trials
            if self.sessions[s_idx].analyzed == True and self.sessions[s_idx].df_conditions.empty:
                sessions_idx_to_remove.append(s_idx)

        if len(sessions_idx_to_remove) > 0:
            print('The following sessions will be removed for lack of trials:')
            # remove faulty sessions (in reverse order so it does not affect
            # the index number of previous elements in the sessions list)
            for r in sorted(sessions_idx_to_remove, reverse = True):
                print('Deleting: ', self.sessions[r].subject_ID, self.sessions[r].datetime, self.sessions[r].task_name)
                del self.sessions[r]

        # signal that the Experiment has been analyzed by trial
        self.by_trial = True

    def list_vids_to_run_in_dlc(
            self,
            list_filepath: str = '\\\\ettin\\Magill_Lab\\Julien\\Models\\to_DLC.csv',
            camera_keyword: str = 'Side', 
            vid_ext: str = 'mp4',
            dlc_ext: str = 'h5',
            scorer: str = None
            ) -> list:
        """
        After sessions have been linked to their respective videos and deeplabcut files
        using <experiment>.match_sessions_to_files(), this function write a .csv file
        with the videos full path which do not have a DeepLabCut match for the camera
        keyword or the specified DeepLabCut scorer 

        Arguments
        ---------
        self : Experiment object
        list_filepath : str = '\\\\ettin\\Magill_Lab\\Julien\\Models\\to_DLC.csv'
            full path where to write a table the videos to be scored by DeepLabCut
            in the form of a .csv file
        camera_keyword : str = 'Side'
            keyword present in the video filename which indicates which view of the
            setup it corresponds to, for multi-camera rigs
        vid_ext : str = 'mp4'
            Extension for the video files, usually mp4
        dlc_ext : str = 'h5'
            Extension for the DeepLabCut scored files, usually h5
        scorer : str = None
            Name of the scorer (trained Neural Network) requested/used for the videos
            e.g.: 'DLC_resnet50_side_2_hands_newobjAug26shuffle1_500000'

        Return
        ------
        unscored_vids : list
            list containing all the filenames of the videos to be scored by DeepLabCut
        """
        vidfiles = [[vidfile for vidfile in session.files[vid_ext] if camera_keyword in vidfile] 
            for session in self.sessions]    

        if scorer:
            dlcfiles = [[dlcfile for dlcfile in session.files[dlc_ext] if (camera_keyword in dlcfile) 
            and (scorer in dlcfile)] for session in self.sessions]
        else:
            dlcfiles = [[dlcfile for dlcfile in session.files[dlc_ext] if camera_keyword in dlcfile] 
                for session in self.sessions]


        unscored_vids = [vidfile for idx, vidfile in enumerate(vidfiles) if dlcfiles[idx] == []]
        unscored_vids = [item for sublist in unscored_vids for item in sublist]

        if list_filepath:
            unscored_df = pd.DataFrame(unscored_vids,columns=['video_path'])
            unscored_df.to_csv(list_filepath)
        
        return unscored_vids


    def check_groups(self, groups):

        # TODO: put all redundant args checks in a utility function
        # list: groups, conditions_list, cond_aliases, task_names, trig_on_ev

        if isinstance(groups, int):
            groups = [[groups]]
        elif groups == None:
            subject_IDs = list(set([session.subject_ID for session in self.sessions]))
            groups = [subject_IDs]
            group_ID = 0
        elif len(groups) > 0 and isinstance(groups[0], int):
            groups = [groups]
            group_ID = 0
        return groups



    # TODO: For this method in particular but for the
    # whole Experiment and Session classes, get rid of all
    # the redundancies between df_events and df_conditions
    # except the most "convenient" ones for sanity checks
    def behav_events_to_dataset(
            self,
            groups: list = None,
            conditions_list: list = None, 
            cond_aliases: list = None, 
            when = 'all', 
            task_names = 'all',
            trig_on_ev: str = None) -> Event_Dataset: 
        """
        Take all behavioural data from the relevant sessions
        and trials and assemble it as an Event_Dataset
        instance for further analyses.

        Arguments
        ---------
        self
        groups : list = None
        conditions_list : list = None
            List of dictionary. Used by Session.get_trials_times_from_conditions to create
            DataFrame Session.df_conditions, the rows of which represent trials. 
            The dictionary keys are used as part of DataFrame column names (columns) together with 'trigger', and 'valid'.
            The dictionary keys are used to determine if trials are valid and stored as the 'valid' column.
            The valid trials of df_conditions are then 
        cond_aliases : list = None
            The list of condition names.
        when = 'all'
        task_names = 'all'
        trig_on_ev: str = None
        """

        groups = self.check_groups(groups)

        if isinstance(conditions_list, dict):
            conditions_list = [conditions_list]
        # construct a list of dict of "all-inclusive" conditions
        elif not conditions_list:
            all_trigs = [session.triggers for session in self.sessions if hasattr(session, 'triggers')]
            all_trigs = set(itertools.chain.from_iterable(all_trigs))
            conditions_list=[]
            for trig_idx, trig in enumerate(all_trigs):
                conditions_list.append(dict())
                conditions_list[trig_idx]['trigger'] = trig

        df_events_exp = pd.DataFrame()
        df_conditions_exp = pd.DataFrame()

        for group_ID, group in enumerate(groups):
            subject_IDs = group
            
            for subject_ID in subject_IDs:    
                # recovering all the sessions for one subject
                sessions = self.get_sessions(subject_IDs = subject_ID, when = when, task_names = task_names)
                    
                # NOTE: events_to_process is an session attribute
                # but trig_on_ev should be checked at the experiment level?
                
                if not isinstance(trig_on_ev, (NoneType , str)):
                    raise TypeError('trig_on_ev argument must be a string')
                elif trig_on_ev != None and trig_on_ev not in sessions[0].events_to_process:
                    raise ValueError(
                        f'{trig_on_ev} is not in the processed events: {sessions[0].events_to_process}')
                
                                        
                for session in sessions:
                    try:
                        df_events = session.df_events.copy()
                        df_conditions = session.df_conditions.copy()
                    except:
                        # print(session.subject_ID, session.datetime)
                        continue
                    df_conditions['condition_ID'] = nan
                    # df_events['condition'] = nan
                    df_conditions['condition'] = nan
                    
                    col_to_modify = [ev + '_trial_time' for ev in session.events_to_process] 
                    # Why +2, investigate this : Probably because reset_index and shifting new resulting column
                    col_idxs = [df_events.columns.to_list().index(col)+2 for col in col_to_modify]
                    
                    # df_events['time_to_ev'] = nan
                    idx_all_cond = np.empty(0, dtype = int)
                    trials_times_all_cond = np.empty(0, dtype = int)
                    first_ev_times_all_cond = np.empty(0, dtype = int)
                    #
                    events_aggreg = pd.DataFrame()
                    for cond_ID, conditions_dict in enumerate(conditions_list):
                        # detect triggers present in the session and skip conditions if
                        # the trigger is not present in the session
                        trigs = set(session.df_conditions.trigger.values)
                        if conditions_dict['trigger'] not in trigs:
                            continue

                        if trig_on_ev and conditions_dict['trigger'] in session.triggers:
                            idx_joint, trials_times, first_ev_times = session.get_trials_times_from_conditions(
                                conditions_dict = conditions_dict,
                                trig_on_ev = trig_on_ev, output_first_ev = True)
                            # print(len(idx_joint), trials_times.shape, first_ev_times.shape)
                        
                            df_ev_cond = df_events.loc[idx_joint,:].copy()
                            df_ev_cond = df_ev_cond.reset_index()
                            # Loop that turns event times list to np.array and substract trial start
                            for ridx, row in enumerate(df_ev_cond.itertuples()): 
                                # print(ridx, row.Index)
                                for c, col_name in enumerate(col_to_modify):
                                    
                                    df_ev_cond.at[row.Index, col_name] = np.array(row[col_idxs[c]]) - first_ev_times[row.Index]

                            events_aggreg = pd.concat([events_aggreg,df_ev_cond])
                        elif conditions_dict['trigger'] in session.triggers: 
                            idx_joint, trials_times = session.get_trials_times_from_conditions(
                                conditions_dict = conditions_dict,
                                trig_on_ev = trig_on_ev,  output_first_ev = False)

                            df_ev_cond = df_events.loc[idx_joint,:].copy()
                            
                            # turn lists to arrays anyway
                            # (lists inherited from early implementation)
                            for col_name in col_to_modify:
                                df_ev_cond[col_name]=df_ev_cond[col_name].apply(lambda x : np.array(x))
                            
                            events_aggreg = pd.concat([events_aggreg,df_ev_cond])

                        # events_aggreg['datetime'] = pd.Series([session.datetime] * events_aggreg.shape[0], 
                        #     index = events_aggreg.index, dtype='datetime64[ns]')
                        # events_aggreg['datetime_string'] = pd.Series([session.datetime_string] * events_aggreg.shape[0],
                        #      index =  events_aggreg.index, dtype='datetime64[ns]')                     

                        idx_all_cond = np.concatenate([idx_all_cond, idx_joint])
                        trials_times_all_cond = np.concatenate([trials_times_all_cond, trials_times])

                        if isinstance(trig_on_ev, str):
                            first_ev_times_all_cond = np.concatenate([first_ev_times_all_cond, first_ev_times])
                        
                        df_conditions.loc[idx_joint, 'group_ID'] = group_ID
                        df_conditions.loc[idx_joint, 'condition_ID'] = cond_ID
                        if cond_aliases:
                            # df_events.loc[idx_joint, 'condition'] = cond_aliases[cond_ID]
                            df_conditions.loc[idx_joint, 'condition'] = cond_aliases[cond_ID]
                        
                        df_conditions['datetime'] = pd.Series([session.datetime] * df_conditions.shape[0], index = df_conditions.index)
                        #df_conditions['datetime_string'] = pd.Series([session.datetime_string] * df_conditions.shape[0], index = df_conditions.index)
                        # df_conditions['date'] = df_conditions['datetime'].dt.date

                    # df_events.dropna(subset=['condition_ID'], inplace=True)
                    # df_conditions.dropna(subset=['condition_ID'], inplace=True)
                    
                    # df_events = df_events.loc[idx_all_cond,:]
                    df_conditions = df_conditions.loc[idx_all_cond,:].reset_index()
                    
                    df_conditions['session_nb'] = session.number
                    df_conditions['group_ID'] = group_ID
                    df_conditions['subject_ID'] = subject_ID

                    # consider implementing session_nb metadata in Session methods

                    df_events_exp = pd.concat([df_events_exp, events_aggreg], ignore_index = True)

                    # Check and enfore boolean columns to be boolean. Workaround for FutureWarning
                    l1 = [all([type(row) is bool for row in df_conditions[col]]) for col in df_conditions.columns]
                    l2 = [df_conditions[col].dtype is not np.dtype('bool') for col in df_conditions.columns]
                    l3 = [a and b for a, b in zip(l1, l2)]
                    for c, d in zip(l3, df_conditions.columns):
                        if c:
                            df_conditions[d] = df_conditions[d].astype('bool')

                    df_conditions_exp = pd.concat([df_conditions_exp, df_conditions], ignore_index = True) #TODO
                    # FutureWarning: In a future version, object-dtype columns with all-bool values will not be 
                    # included in reductions with bool_only=True. Explicitly cast to bool dtype instead.
                    # https://stackoverflow.com/questions/73800841/add-series-as-a-new-row-into-dataframe-triggers-futurewarning
                    # https://stackoverflow.com/questions/71465386/why-pd-concat-of-two-dataframe-leads-to-futurewarning-behavior-when-concatena
        
        # df_conditions_exp['condition_ID'] = df_conditions_exp['condition_ID'].astype(int)         
        ev_dataset = Event_Dataset(df_events_exp, df_conditions_exp)

        ev_dataset.conditions = conditions_list
        ev_dataset.cond_aliases = cond_aliases
        # TODO: remove, should be a temporary check
        if hasattr(self, 'trial_window'):
            ev_dataset.set_trial_window(self.trial_window)
        return ev_dataset

    # Todo move out in utilities
    def match_sessions_to_files(self, files_dir, ext='mp4', verbose=False):
        '''
        Take an experiment instance and look for files within a directory
        taken the same day as the session and containing the subject_ID,
        store the filename(s) with the shortest timedelta compared to the
        start of the session in exp.sessions[x].files["ext"] as a list
        
                Parameters:
                        files_dir (str): path of the directory to look into
                        ext (str): extension used to filter files within a folder
                            do not include the dot. e.g.: "mp4"

                Returns:
                        None (store list in sessions[x].file["ext"])
        ''' 

        # subject_IDs = [session.subject_ID for session in self.sessions]
        # datetimes = [session.datetime for session in self.sessions]
        files_list = [f for f in os.listdir(files_dir) if os.path.isfile(
            os.path.join(files_dir, f)) and ext in f]

        if len(files_list) == 0:
            raise Exception(f'No files with the .{ext} extension where found in the following folder: {files_dir}')

        files_df = pd.DataFrame(columns=['filename','datetime'])

        files_df['filename'] = pd.DataFrame(files_list)
        files_df['datetime'] = files_df['filename'].apply(lambda x: get_datetime_from_datestr(get_datestr_from_filename(x)))
        # print(files_df['datetime'])
        for s_idx, session in enumerate(self.sessions):
            match_df = find_matching_files(session.subject_ID, session.datetime, files_df, ext)
            if verbose:
                print(session.subject_ID, session.datetime, match_df['filename'].values)
            
            if not hasattr(self.sessions[s_idx], 'files'):
                self.sessions[s_idx].files = dict()
            
            self.sessions[s_idx].files[ext] = [os.path.join(files_dir, filepath) for filepath in match_df['filename'].to_list()]
    
    # consider deleting and using match_sessions_to_files,
    # then only compute rsync aligner at extraction (or not
    # since it is good to know in advance which photometry
    # files are actually useful)
    def sync_photometry_files(self,  
            rsync_chan: int = 2,
            delete_unsynced: bool = True, 
            verbose: bool = False):
        """
        This function create a rsync aligment object into the corresponding
        session if the rsync pulses match betwwen pycontrol and pyphotometry files.

            Parameters:
                self (Experiment): An Experiment object instance
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
            
        for id_f, session in enumerate(self.sessions):

            if session.files['ppd'] != []:
                # try to align times with rsync
                try:
                    # Gives KeyError exception if no rsync pulses on pycontrol file
                    pycontrol_rsync_times = session.times['rsync']
                
                    photometry_dict = import_ppd(session.files['ppd'][0])
                    
                    photometry_rsync_times = photometry_dict['pulse_times_' + str(rsync_chan)]

                    pyphoto_aligner = Rsync_aligner(pulse_times_A= pycontrol_rsync_times, 
                        pulse_times_B= photometry_rsync_times, plot=False)
                    
                    if verbose:
                        print('pycontrol: ', session.subject_ID, session.datetime,
                        '/ pyphotometry: ', session.files['ppd'][0], ' : rsync does match')
                    
                    self.sessions[id_f].photometry_rsync = pyphoto_aligner

                # if rsync aligner fails    
                except (RsyncError, ValueError, KeyError):
                    self.sessions[id_f].photometry_rsync = None

                    if verbose:
                        print('pycontrol: ', session.subject_ID, session.datetime,
                        '/ pyphotometry: ', session.files['ppd'][0], ' : rsync does not match')

                    if delete_unsynced:
                        self.sessions[id_f].files['ppd'] = []

            # if there is no subject + date match in .ppd files
            else: 
                self.sessions[id_f].photometry_rsync = None

                if verbose:
                    print('pycontrol: ', session.subject_ID, session.datetime,
                    '/ pyphotometry: no file matching both subject and date')


    def get_deeplabcut_groups(
            self, 
            groups: list = None, # list of 
            conditions_list: list = None, # list of conditions dictionaries
            cond_aliases: list = None, # list of conditions str aliases
            when = 'all', # see get_sessions() for formatting
            task_names: str = 'all', # can be list of str
            trig_on_ev: str = None, 
            camera_fps: int = 100,
            camera_keyword: str ='side',
            dlc_scorer: str = None, 
            bodyparts_to_ave: list = None,
            names_of_ave_regions: list = None, 
            normalize_between: list = None, 
            bins_nb: int = 200, 
            p_thresh: float = 0.6, 
            bodyparts_to_store: list = None, 
            three_dims: bool = False, 
            verbose=False) -> Continuous_Dataset:

        '''
        get all deeplabcut trials for one or several group(s) of subject(s) in one or several conditions
        '''
        # TODO: Elaborate docstring. mention that return_full_session is not possible for an Experiment
        
        # if self.by_trial == False:
        #     raise Exception('Process experiment by trial first: experiment.process_exp_by_trial(trial_window, timelim, tasksfile)')

        if isinstance(conditions_list, dict):
            conditions_list = [conditions_list]
        
        if isinstance(cond_aliases, str):
            cond_aliases = [cond_aliases]

        if cond_aliases:
            if len(cond_aliases) != len(conditions_list):
                raise ValueError(
                    'conditions_list and cond_aliases must have the same length'
                )


        if isinstance(groups, int):
            groups = [[groups]]
        elif groups == None:
            subject_IDs = list(set([session.subject_ID for session in self.sessions]))
            groups = [subject_IDs]
            group_ID = 0
        elif len(groups) > 0 and isinstance(groups[0], int):
            groups = [groups]
            group_ID = 0
        
        for group_ID, group in enumerate(groups):
            subject_IDs = group

            for subject_ID in subject_IDs:    
                # recovering all the sessions for one subject
                sessions = self.get_sessions(subject_IDs = subject_ID, when = when, task_names = task_names)
                # Only take sessions which have a dlc file matching:
                sessions = [session for session in sessions if not session.files['h5'] == []]
                # if this subject has no dlc data
                if sessions == []:
                    continue

                for s_idx, session in enumerate(sessions):
                    # forward arguments to the session method:
                    if not hasattr(session, 'df_conditions'):
                        continue
                    if verbose:
                        print(f'Processing subject {session.subject_ID} at: {session.datetime_string}')

                    try:
                        df_meta_dlc, col_names_numpy, dlc_array = session.get_deeplabcut_trials(
                            conditions_list = conditions_list, 
                            cond_aliases = cond_aliases,
                            camera_fps = camera_fps, 
                            camera_keyword = camera_keyword,
                            bodyparts_to_ave = bodyparts_to_ave, 
                            names_of_ave_regions = names_of_ave_regions,
                            normalize_between = normalize_between, 
                            bins_nb = bins_nb, 
                            p_thresh = p_thresh,
                            bodyparts_to_store = bodyparts_to_store,
                            trig_on_ev = trig_on_ev, 
                            three_dims = three_dims, 
                            return_full_session = False, 
                            verbose = verbose)
                                        
                    except UnboundLocalError:
                        print(f'No trial in any condition for subject {session.subject_ID} at: {session.datetime_string}')
                        continue
                    except DeepLabCutFileError as DLCerr:
                        if verbose:
                            print(DLCerr)
                        continue
                    # consider implementing session_nb metadata in Session methods
                    df_meta_dlc['session_nb'] = session.number
                    df_meta_dlc['group_ID'] = group_ID

                    if 'df_meta_dlc_exp' in locals():
                        
                        df_meta_dlc_exp = pd.concat((df_meta_dlc_exp, df_meta_dlc), ignore_index=True)
                        dlc_array_exp = np.concatenate((dlc_array_exp, dlc_array), axis=0)

                    else:
                        df_meta_dlc_exp = df_meta_dlc
                        dlc_array_exp = dlc_array
            
            if 'df_meta_dlc_exp' not in locals():
                raise Exception(f'The following group: {subject_IDs} do not contain deeplabcut trials. \
                    \r\n consider looking for more sessions with when, or broadening conditions')
        
        cont_dataset = Continuous_Dataset(dlc_array_exp, df_meta_dlc_exp, col_names_numpy)
        cont_dataset.set_fs(camera_fps)
        cont_dataset.set_conditions(conditions_list)
        # TODO: remove, should be a temporary check
        if hasattr(self, 'trial_window'):
            cont_dataset.set_trial_window(self.trial_window)
        return cont_dataset
    
    # TODO: Implement params structure to save
    def get_photometry_groups(
            self, 
            groups = None,
            conditions_list = None,
            cond_aliases = None,
            trial_window: list = None,
            trig_on_ev: str = None,  # align to the first event of a kind e.g. None (meaning CS_Go onset), 'spout', 'bar_off'
            last_before: str = None,
            when = 'all',
            task_names = 'all',
            baseline_low_pass: int = None, # changed var name from high-pass to baseline_low_pass
            # due to https://github.com/juliencarponcy/trialexp/pull/9
            # fixed in https://github.com/juliencarponcy/trialexp/pull/9/commits/2bd4307af9ce2096ff1673b56cf6bacf0a2a8127#diff-90aedd18a2a5cd46987018614831622fb110ef9b08b1d3baad395bf36c0a6e1c
            low_pass: int = None, 
            median_filt: int = None,
            motion_corr: bool = False, 
            df_over_f: bool = False,
            z_score: bool = False, 
            downsampling_factor: int = None,
            export_vars: list = ['analog_1','analog_2'],
            # remove_artifacts: bool = False,
            verbose = False) -> Continuous_Dataset:
        '''
        get all photometry trials for one or several group(s) of subject(s) in one or several conditions
        '''
        # TODO: Elaborate docstring. mention that return_full_session is not possible for an Experiment
        
        if self.by_trial == False:
            raise Exception('Process experiment by trial first: experiment.process_exp_by_trial(trial_window, timelim, tasksfile)')
        # elif not isinstance(groups, list):
        #     raise Exception('groups variable must be a list [281, 282, 283], list of list [[281, 282, 283],[284, 285, 286]]')

        if isinstance(conditions_list, dict):
            conditions_list = [conditions_list]

        if isinstance(cond_aliases, str):
            cond_aliases = [cond_aliases]

        if cond_aliases:
            if len(cond_aliases) != len(conditions_list):
                raise ValueError(
                    'conditions_list and cond_aliases must have the same length'
                )


        if isinstance(groups, int):
            groups = [[groups]]
        elif groups == None:
            subject_IDs = list(set([session.subject_ID for session in self.sessions]))
            groups = [subject_IDs]
            group_ID = 0
        elif len(groups) > 0 and isinstance(groups[0], int):
            groups = [groups]
            group_ID = 0
        
        for group_ID, group in enumerate(groups):
            subject_IDs = group

            for subject_ID in subject_IDs:    
                # recovering all the sessions for one subject
                sessions = self.get_sessions(subject_IDs=subject_ID, when=when, task_names=task_names)
                # Only take sessions which have a photometry file matching:
                sessions = [session for session in sessions if session.photometry_rsync is not None]
                # if this subject has no photometry data
                if sessions == []:
                    continue

                for s_idx, session in enumerate(sessions):
                    # forward arguments to the session method:

                    if verbose:
                        print(f'Processing subject {session.subject_ID} at: {session.datetime_string}')

                    try:
                        df_meta_photo, col_names_numpy, photometry_array, fs = get_photometry_trials(
                            session,
                            conditions_list = conditions_list, 
                            cond_aliases = cond_aliases,
                            trial_window = self.trial_window,
                            trig_on_ev = trig_on_ev,
                            last_before = last_before,
                            baseline_low_pass = baseline_low_pass, # var name changed from former high-pass,
                            # was misleading on baseline computation
                            # see https://github.com/juliencarponcy/trialexp/issues/8
                            # first fix 
                            low_pass = low_pass, 
                            median_filt = median_filt,
                            motion_corr = motion_corr, 
                            df_over_f = df_over_f, 
                            z_score = z_score,
                            downsampling_factor = downsampling_factor, 
                            return_full_session = False,
                            export_vars = export_vars,
                            # remove_artifacts = remove_artifacts,
                            verbose = verbose)
                    
                    except UnboundLocalError:
                        print(f'No trial in any condition for subject {session.subject_ID} at: {session.datetime_string}')
                        continue

                    # consider implementing session_nb metadata in Session methods
                    df_meta_photo['session_nb'] = session.number
                    df_meta_photo['group_ID'] = group_ID

                    # if data already stored
                    if 'df_meta_photo_exp' in locals():
                        # check sampling rate consistency
                        if fs != fs_exp:
                            raise NotImplementedError(f'Current file has fs = {fs} \
                                whereas other files had fs = {fs_exp}')
                        
                        df_meta_photo_exp = pd.concat((df_meta_photo_exp, df_meta_photo), ignore_index=True)
                        photometry_array_exp = np.concatenate((photometry_array_exp, photometry_array), axis=0)

                    else:
                        fs_exp = fs # to check if all files have same sampling frequency

                        df_meta_photo_exp = df_meta_photo
                        photometry_array_exp = photometry_array
            
            if 'df_meta_photo_exp' not in locals():
                raise Exception(f'The following Experimental group: {subject_IDs} do not contain photometry trials. \
                    \r\n consider checking the task_names requested, include more sessions, or broadening conditions')
    
        cont_dataset = Continuous_Dataset(photometry_array_exp, df_meta_photo_exp, col_names_numpy)
        cont_dataset.set_fs(fs_exp)
        cont_dataset.set_conditions(conditions_list, cond_aliases)
        
        # TODO: remove, should be a temporary check
        if hasattr(self, 'trial_window'):
            cont_dataset.set_trial_window(self.trial_window)
        return cont_dataset
    
    
    def plot(self, what='behav.sucess_rate', events=None, groups=None, conditions_list=None,
                color=None, when='all', task_names='all', verbose=False):
        '''
        Old method, track behavioural metrics over sessions.
        TODO: DEPRECATE: turn into event dataset and adapt that method 
        '''
        if self.by_trial == False:
            raise Exception('Process experiment by trial first: experiment.process_exp_by_trial(trial_window, timelim, tasksfile)')
        # elif not isinstance(groups, list):
        #     raise Exception('groups variable must be a list [281, 282, 283], list of list [[281, 282, 283],[284, 285, 286]]')

        if isinstance(conditions_list, dict):
            conditions_list = [conditions_list]

        # time_delta computations need two events to compute only one metric
        if 'timedelta' in what.split('.')[1] and len(events) != 2:
            raise Exception('length of events arguments must be 2 when you request a timedelta')

        # creation of the results dataframe
        columns = ['datetime','session_ID','subject_ID','group_ID','condition_ID','task_name','num_trials', 'metric_name', 'metric']
        results = pd.DataFrame(columns=columns)

        if isinstance(groups, int):
            groups = [[groups]]
        elif groups == None:
            subject_IDs = list(set([session.subject_ID for session in self.sessions]))
            groups = [subject_IDs]
            group_ID = 0
        elif len(groups) > 0 and isinstance(groups[0], int):
            groups = [groups]
            group_ID = 0
        # elif len(groups) > 0 and isinstance(groups[0], list):
        #     ...


        print(groups)
        for condition_ID, conditions_dict in enumerate(conditions_list):

            for group_ID, group in enumerate(groups):
                subject_IDs = group

                for subject_ID in subject_IDs:    
                    # recovering all the sessions for one subject
                    sessions = self.get_sessions(subject_IDs=subject_ID, when=when, task_names=task_names)
                    
                    for s_idx, session in enumerate(sessions):
                        # Prevent analysis from crash if the file is too short to have been analyzed
                        if session.analyzed == False:
                            continue
                        if what.split('.')[0] == 'behav':
                            if verbose:
                                print('session nb: ', session.number, session.subject_ID, session.datetime_string, session.task_name)
                            sessions[s_idx] = session.compute_behav_metrics(conditions_dict=conditions_dict, events=events)
                            
                            # the following block differs mainly in how to construct metric_name
                            # and if iterations over different metrics must be performed (multiple events)
                            if 'success_rate' in what.split('.')[1] or 'timedelta' in what.split('.')[1]:
                                metric_name = what.split('.')[1]              
                                result_row = [session.datetime, s_idx, subject_ID, group_ID, condition_ID, session.task_name,
                                    session.behav['num_trials'], metric_name, session.behav[what.split('.')[1]]]
                                # append the result of the session to the global results dataframe
                                results.loc[len(results)] = result_row
                            elif 'timedelta' in what.split('.')[1]:
                                metric_name = what.split('.')[1] + '_' + events[0] + '_' + events[1]
                                result_row = [session.datetime, s_idx, subject_ID, group_ID, condition_ID, session.task_name,
                                    session.behav['num_trials'], metric_name, session.behav[what.split('.')[1]]]
                                # append the result of the session to the global results dataframe
                                results.loc[len(results)] = result_row
                            else:
                                for ev in events:
                                    metric_name = what.split('.')[1] + '_' + ev
                                    result_row = [session.datetime, s_idx, subject_ID, group_ID, condition_ID, session.task_name,
                                        session.behav['num_trials'], metric_name, sessions[s_idx].behav[metric_name]]
                                    # append the result of the session to the global results dataframe
                                    results.loc[len(results)] = result_row
                        

        plot_longitudinal(results)                   

        return results

   
#----------------------------------------------------------------------------------
# Helpers
#----------------------------------------------------------------------------------

# somehow broke when stored in pycontrol_utilities
def _toDate(d): # Convert input to datetime.date object.
    if type(d) is str:
        try:
            return datetime.strptime(d, '%Y-%m-%d').date()
        except ValueError:
            raise ValueError('Unable to convert string to date, format must be YYYY-MM-DD.')
    elif type(d) is datetime:
        return d.date()
    elif type(d) is date:
        return d
    else:
        raise ValueError('Unable to convert input to date.')
