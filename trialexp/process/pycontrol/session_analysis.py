
from cmath import isnan, nan

import os
from pathlib import Path
import pickle
import re
import datetime
import warnings
import datetime
import itertools

from collections import namedtuple
from operator import itemgetter

import numpy as np
import pandas as pd

from math import ceil
from scipy.signal import butter, filtfilt, decimate
from scipy.stats import linregress, zscore

#----------------------------------------------------------------------------------
# Session class
#----------------------------------------------------------------------------------


    
def add_time_rel_trigger(df_events, trigger_time, col_name, trial_window):
    #Add new time column to the event data, aligned to the trigger time
    # the new time column can also be negative, the search window in which the trigger will apply
    # is defined in time_window
    df = df_events.copy()
    df[col_name] = np.nan

    #TODO: can this be overalpping?

    trial_nb = 1
    for t in trigger_time:
        td = df.time - t
        idx = (trial_window[0]<td) & (td<trial_window[1])
        df.loc[idx, col_name] =  df[idx].time - t
        trial_nb += 1

    return df


def add_trial_nb(df_events, trigger_time, trial_window):
    # add trial number into the event dataframe
    df = df_events.copy()
    df['trial_nb'] = np.nan
    #TODO: can time window be overalpping?

    trial_nb = 1
    for t in trigger_time:
        td = df.time - t
        idx = (trial_window[0]<td) & (td<trial_window[1])
        df.loc[idx, ['trial_nb']] = trial_nb
        trial_nb += 1

    return df

def get_task_specs(tasks_trig_and_events, task_name):
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

        # match triggers (events/state used for t0)
        triggers = np.array2string(tasks_trig_and_events['triggers'][tasks_trig_and_events['task'] == task_name].values).strip("'[]").split('; ')

        conditions = np.array2string(tasks_trig_and_events['conditions'][tasks_trig_and_events['task'] == task_name].values).strip("'[]").split('; ')

        return conditions, triggers

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

        self.print_lines = [line[2:] for line in all_lines if line[0]=='P']
        
        self.state_IDs = state_IDs
        self.event_IDs = event_IDs
    
    @staticmethod
    def get_task_specs(tasks_trig_and_events, task_name):
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

        # match triggers (events/state used for t0)
        triggers = np.array2string(tasks_trig_and_events['triggers'][tasks_trig_and_events['task'] == task_name].values).strip("'[]").split('; ')
                
        # events to extract
        events_to_process = np.array2string(tasks_trig_and_events['events'][tasks_trig_and_events['task'] == task_name].values).strip("'[]").split('; ')
        # printed line in task file indicating
        # the type of optogenetic stimulation
        # used to group_by trials with same stim/sham
        conditions = np.array2string(tasks_trig_and_events['conditions'][tasks_trig_and_events['task'] == task_name].values).strip("'[]").split('; ')
        
        # REMOVED, now only at Experiment level to avoid inconsistencies
        # define trial_window parameter for extraction around triggers
        # self.trial_window = trial_window        
        return conditions, triggers

    # @staticmethod
    # def extract_data_from_session(df_events):
    #     """
    #     The two attributes
    #         self.df_events
    #         self.df_conditions
    #     are assigned by looking into a session data
    #     """

    #     df_events = pd.DataFrame(self.events, columns=['timestamp', 'event'])
        
    #     df_events['timestamp'] = df_events['timestamp'].astype('int')

    #     # parsing timestamps and events from print lines
    #     print_events = [line.split() for line in self.print_lines]
    #     print_ts = [int(line[0]) for line in print_events]
    #     print_text = [' '.join(line[1:]) for line in print_events]
        
    #     # put timestamp and text of print lines in a dataframe, and make the timestamp as index
    #     df_print_events = pd.DataFrame({'timestamp':print_ts,'event':print_text})
        
    #     # keep print_lines that are relevant to task analysis
    #     df_print_events = df_print_events.loc[df_print_events['event'].isin(
    #         self.triggers + self.events_to_process + self.conditions
    #     )]
 
    #     # keep events in df if any event is relevant for behaviour        
    #     df_events = df_events.loc[df_events['event'].isin(
    #         self.triggers + self.events_to_process + self.conditions
    #     )]

    #     # Merge print and events which are relevant to conditions of trials (time insensitive)
    #     df_conditions = pd.concat([df_print_events.loc[df_print_events['event'].isin(self.conditions)],
    #         df_events.loc[df_events['event'].isin(self.conditions)]]
    #     , ignore_index=False)

    #     # Merge print and events which are relevant to events of trials (time sensitive)
    #     df_events = pd.concat([df_print_events.loc[df_print_events['event'].isin(self.triggers + self.events_to_process)],
    #         df_events.loc[df_events['event'].isin(self.triggers + self.events_to_process)]]
    #     , ignore_index=True)

    #     # Turn into events/conditions string into categorical variables      
    #     df_events['event'] = df_events['event'].astype('category')
    #     df_conditions['event'] = df_conditions['event'].astype('category')

    #     # use timestamp as index
    #     df_events.set_index('timestamp',inplace=True, drop=True)
    #     df_conditions.set_index('timestamp',inplace=True, drop=True)

    #     self.df_events = df_events
    #     self.df_conditions = df_conditions
        
    #     # return session objet
    #     return self

    # VERY UGLY AND SLOW:
    # compute trial nb and triggering events types to aggegate and index on them
    # TODO: optimize with itertuples or apply 
    @staticmethod
    def compute_trial_nb(trial_window, df_conditions):

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

        # basically 
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

        new_df = pd.DataFrame(index= df_events.index.get_level_values('trial_nb').unique(),
            columns=columns)

        # Create unique identifiers for trials
        new_df['trial_nb'] = new_df.index.values
        new_df['uid'] = new_df['trial_nb'].apply(
            lambda x: f'{self.subject_ID}_{self.datetime.date()}_{self.datetime.time()}_{x}')
        new_df['trial_nb'].drop(columns='trial_nb', inplace=True)

        # fill new <event>_trial_time columns
        for ev in self.events_to_process:
            try:
                new_df[str(ev + '_trial_time')] = df_events.loc[
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
            df_conditions_summed = self.df_conditions[self.conditions + ['trial_nb']].groupby(['trial_nb'], as_index=False)
            # Aggregate different timestamp ()
            df_conditions_summed = df_conditions_summed.agg(lambda x: sum(x))   

            # reindexing to have conditions for all trials even if no condition has been detected for some trials
            df_conditions = df_conditions_summed.reindex(index=self.df_events.index, fill_value=0)   

            df_conditions = pd.concat([self.df_events[['trigger','valid']], df_conditions], axis='columns', join='outer')

            self.df_conditions = df_conditions
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
    def get_session_by_trial(self, trial_window: list, timelim: list,
            tasksfile, blank_spurious_event: list, blank_timelim: list, verbose=False):
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
        elif self.task_name in ['reaching_go_spout_bar_dual_all_reward_dec22', 
            'reaching_go_spout_bar_dual_dec22', 'reaching_go_spout_bar_nov22']:

            reach_time_before_reward = self.df_events.loc[:,['spout_trial_time','US_end_timer_trial_time']].apply(
                    lambda x: find_last_time_before_list(x['spout_trial_time'], x['US_end_timer_trial_time']), axis=1)    
            # select only trials with a spout event before a US_end_timer event
            reach_bool = reach_time_before_reward.notnull()
            # select trial where the hold time was present (not aborted)
            reach_success_bool = reach_bool & self.df_conditions.waiting_for_spout
            # set these trials as successful
            self.df_conditions.loc[(reach_success_bool), 'success'] = True


        # Reorder columns putting trigger, valid and success first for more clarity
        col_list = list(self.df_conditions.columns.values)
        col_to_put_first = ['trigger', 'success','valid']
        for c in col_to_put_first:
            col_list.remove(c)
        col_list = ['trigger', 'success','valid'] + col_list
        self.df_conditions = self.df_conditions[col_list]

        

        return self

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
            downsampling_factor: int = None,
            return_full_session: bool = False, 
            export_vars: list = ['analog_1','analog_2'],
            # remove_artifacts: bool = False,
            verbose: bool = False):
            
        # TODO write docstrings
        
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

            slope, intercept, r_value, p_value, std_err = linregress(x=photometry_dict['analog_2_filt'], y=photometry_dict['analog_1_filt'])
            photometry_dict['analog_1_est_motion'] = intercept + slope * photometry_dict['analog_2_filt']
            photometry_dict['analog_1_corrected'] = photometry_dict['analog_1_filt'] - photometry_dict['analog_1_est_motion']
            
            if df_over_f == False:
                export_vars.append('analog_1_corrected')
                # signal = photometry_dict['analog_1_corrected']
            elif df_over_f == True:

                b,a = butter(2, 0.001, btype='low', fs=photometry_dict['sampling_rate'])
                photometry_dict['analog_1_baseline_fluo'] = filtfilt(b,a, photometry_dict['analog_1_filt'], padtype='even')

                # Now calculate the dF/F by dividing the motion corrected signal by the time varying baseline fluorescence.
                photometry_dict['analog_1_df_over_f'] = photometry_dict['analog_1_corrected'] / photometry_dict['analog_1_baseline_fluo'] 
                export_vars.append('analog_1_df_over_f')
                # signal = photometry_dict['analog_1_df_over_f']

        elif high_pass or low_pass:
            # signal = photometry_dict['analog_1_filt']
            export_vars.append('analog_1_filt')

            # control = photometry_dict['analog_2_filt']
        else:
            export_vars.append('analog_1')
        # signal = photometry_dict['analog_1']

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

            timestamps_photometry = self.photometry_rsync.A_to_B(timestamps_pycontrol)
            photometry_idx = (timestamps_photometry / (1000/photometry_dict['sampling_rate'])).round().astype(int)
            

            # print(f'photometry {photometry_idx[0:10]} \r\n pycontrol {timestamps_pycontrol[0:10]}')
            
            # Compute mean time difference between A and B and replace NaN values at extremity of files
            # (rsync_aligner timestamps are only computed BETWEEN rsync pulses)
            # NOTE: Can be uncommented but likely to reintroduce timestamps outside of photometry timestamps span
            # nan_idx = isnan(timestamps_photometry)
            # mean_diff = nanmean(timestamps_photometry - timestamps_pycontrol.astype(float))
            # timestamps_photometry[nan_idx] = timestamps_pycontrol[nan_idx] + mean_diff
            # timestamps_photometry = timestamps_photometry.astype(int)

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

    