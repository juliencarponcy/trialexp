# Python classes for importing pyControl data files and representing pyControl 
# sessions and experiments.  Dependencies: Python 3.5+, Numpy.
from cmath import isnan, nan

import os
import pickle
import re
import datetime
import warnings
import datetime

from collections import namedtuple
from operator import itemgetter

import numpy as np
import pandas as pd

from math import ceil
from scipy.signal import butter, filtfilt, decimate
from scipy.stats import linregress

from trialexp.utils.pycontrol_utilities import *
from trialexp.utils.rsync import *
from trialexp.dataset_classes.trial_dataset_classes import *

Event = namedtuple('Event', ['time','name'])

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
    '''Import data from a pyControl file and represent it as an object with attributes:
      - file_name
      - experiment_name
      - task_name
      - setup_ID
          The COM port of the computer used (can be useful when multiple rigs on one computer)
      - subject_ID
          If argument int_subject_IDs is True, suject_ID is stored as an integer,
          otherwise subject_ID is stored as a string.
      - datetime
          The date and time that the session started stored as a datetime object.
      - datetime_string
          The date and time that the session started stored as a string of format 'YYYY-MM-DD HH:MM:SS'
      - events
          A list of all framework events and state entries in the order they occured. 
          Each entry is a namedtuple with fields 'time' & 'name', such that you can get the 
          name and time of event/state entry x with x.name and x.time respectively.
      - times
          A dictionary with keys that are the names of the framework events and states and 
          corresponding values which are Numpy arrays of all the times (in milliseconds since the
           start of the framework run) at which each event/state entry occured.
      - print_lines
          A list of all the lines output by print statements during the framework run, each line starts 
          with the time in milliseconds at which it was printed.
    '''

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
    
    def get_task_specs(self, tasksfile, trial_window, timelim):

        # all the df column named in this function, events and opto_categories must 
        # follow columns of the indicated tasksfile

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
        
        # define trial_window parameter for extraction around triggers
        self.trial_window = trial_window
        self.timelim = timelim
        
        return self

    def extract_data_from_session(self):

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
        df_events.set_index('timestamp',inplace=True, drop=False)
        df_conditions.set_index('timestamp',inplace=True, drop=False)

        self.df_events = df_events
        self.df_conditions = df_conditions

        # return session objet
        return self

    # VERY UGLY AND SLOW:
    # compute trial nb and triggering events types to aggegate and index on them
    # TODO: optimize with itertuples or apply 
    def compute_trial_nb(self):

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
                    (df_events.index >= trigtime + self.trial_window[0])
                , ntrial+1, inplace=True)

                df_conditions['trial_nb'].mask(
                    (df_conditions.index >= trigtime + self.trial_window[0])
                , ntrial+1, inplace=True)

                df_events['trial_time'].mask(
                    (df_events.index >= trigtime + self.trial_window[0])
                , df_events.index[
                    (df_events.index >= trigtime + self.trial_window[0])
                ] - trigtime, inplace=True)

                # determine triggering event
                df_conditions['trigger'].mask(
                    (df_conditions.index >= trigtime + self.trial_window[0])
                , all_trial_triggers_sorted[ntrial], inplace=True)

                # compute trial relative time
                df_conditions['trial_time'].mask(
                    (df_conditions.index >= trigtime + self.trial_window[0])
                , df_conditions.index[
                    (df_conditions.index >= trigtime + self.trial_window[0])
                ] - trigtime, inplace=True)

            # for every trial except last
            else: 
                #print('all but last trial: ', ntrial, 'over', len(all_trial_times_sorted),  all_trial_times_sorted[ntrial+1])

                df_events['trial_nb'].mask(
                    (df_events.index >= trigtime + self.trial_window[0]) &
                    (df_events.index <= all_trial_times_sorted[ntrial+1] + self.trial_window[0])
                , ntrial+1, inplace=True)

                df_conditions['trial_nb'].mask(
                    (df_conditions.index >= trigtime + self.trial_window[0]) &
                    (df_conditions.index <= all_trial_times_sorted[ntrial+1] + self.trial_window[0])
                , ntrial+1, inplace=True)

                df_events['trial_time'].mask(
                    (df_events.index >= trigtime + self.trial_window[0]) &
                    (df_events.index <= all_trial_times_sorted[ntrial+1] + self.trial_window[0])
                , df_events.index[
                    (df_events.index >= trigtime + self.trial_window[0]) &
                    (df_events.index <= all_trial_times_sorted[ntrial+1] + self.trial_window[0])
                ] - trigtime, inplace=True)

                # determine triggering event
                df_conditions['trigger'].mask(
                    (df_conditions.index >= trigtime + self.trial_window[0]) &
                    (df_conditions.index <= all_trial_times_sorted[ntrial+1] + self.trial_window[0])
                , all_trial_triggers_sorted[ntrial], inplace=True)

                # compute trial relative time
                df_conditions['trial_time'].mask(
                    (df_conditions.index >= trigtime + self.trial_window[0]) &
                    (df_conditions.index <= all_trial_times_sorted[ntrial+1] + self.trial_window[0])
                , df_conditions.index[
                    (df_conditions.index >= trigtime + self.trial_window[0]) &
                    (df_conditions.index <= all_trial_times_sorted[ntrial+1] + self.trial_window[0])
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

        columns=['timestamp', 'trigger', 'valid', 'success']
        columns = columns + ev_col_list

        new_df = pd.DataFrame(index= df_events.index.get_level_values('trial_nb').unique(),
            columns=columns)

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
        new_df['valid'] = new_df['timestamp'].diff() > self.trial_window[0]
        
        # validate first trial except if too early in the session
        if new_df['timestamp'].iloc[0] > abs(self.trial_window[0]):
           new_df['valid'] = True
        
        # assing the newly built dataframe into the session object
        self.df_events = new_df

        # replace NaN by an empty lists for trial without events
        for ev_times_col in ev_col_list:
            self.df_events.loc[(self.df_events[ev_times_col].isnull()), ev_times_col] = \
                self.df_events.loc[(self.df_events[ev_times_col].isnull()), ev_times_col].apply(lambda x: [])
            

        #self.df_events = new_df
        self.df_conditions = df_conditions

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
            df_conditions_summed = self.df_conditions[self.conditions + ['trial_nb']].groupby(['trial_nb'], as_index=True)


        # Aggregate different timestamp ()
        df_conditions_summed = df_conditions_summed.agg(lambda x: sum(x))   

        # reindexing to have conditions for all trials even if no condition has been detected for some trials
        df_conditions = df_conditions_summed.reindex(index=self.df_events.index, fill_value=0)   

        df_conditions = pd.concat([self.df_events[['trigger','valid']], df_conditions], axis='columns', join='outer')

        self.df_conditions = df_conditions
        self.df_conditions[self.conditions] = self.df_conditions[self.conditions].astype(bool)

        # Compute if trials are cued or uncued for this specific task
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
            else:
                self.df_conditions.loc[(self.df_events.timestamp > block_lim),'cued'] = True
                self.df_conditions.loc[(self.df_events.timestamp < block_lim),'cued'] = False

            # change triggers name for this task to cued and uncued
            self.triggers = ['cued', 'uncued']
            self.df_conditions.loc[(self.df_conditions.cued == True),['trigger']] = self.triggers[0]
            self.df_conditions.loc[(self.df_conditions.cued == False),['trigger']] = self.triggers[1]
            self.df_events.loc[(self.df_conditions.cued == True),['trigger']] = self.triggers[0]
            self.df_events.loc[(self.df_conditions.cued == False),['trigger']] = self.triggers[1]
        # print(self.df_events.shape, self.df_conditions.shape)
        return self

    # Perform all the pretreatments to analyze behavioural file by trials
    def get_session_by_trial(self, trial_window, timelim, tasksfile, verbose=False):
        
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
                self.trial_window = trial_window
                # get triggers and events to analyze, set trial_window to be extracted
                # and timelim for which trials are considered success/fail
                self = self.get_task_specs(tasksfile,trial_window, timelim)
                # get triggers and events to analyze
                if verbose:
                    print(f'processing by trial: {self.file_name} task: {self.task_name}')

                self = self.extract_data_from_session()
                self = self.compute_trial_nb() 
                self = self.compute_conditions_by_trial()
                self = self.compute_success()
                self.analyzed = True
                return self
                #pycontrol_utilities method

                #print(self.print_events,df_events)
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
        self.df_conditions['success'] = False
        self.df_events['success'] = False
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
            self.df_events.loc[(go_success_idx),'success'] = True
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
            self.df_events.loc[(nogo_success_idx),'success'] = True

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
            self.df_events.loc[(go_success_idx),'success'] = True

        # To perform for cued-uncued version of the go task
        elif self.task_name in ['reaching_go_spout_cued_uncued']:

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
            self.df_conditions.loc[(cued_success_idx + uncued_success_idx), 'success'] = True
            self.df_events.loc[(cued_success_idx + uncued_success_idx),'success'] = True
            print(self.task_name, self.subject_ID, self.datetime_string, len(cued_success_idx), len(uncued_success_idx))
        
        # Reorder columns putting trigger, valid and success first for more clarity
        col_list = list(self.df_conditions.columns.values)
        col_to_put_first = ['trigger', 'success','valid']
        for c in col_to_put_first:
            col_list.remove(c)
        col_list = ['trigger', 'success','valid'] + col_list
        self.df_conditions = self.df_conditions[col_list]

        

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
        
        if not hasattr(self, 'files'):
            raise Exception('The session has not been matched with a deeplabcut (.nwb / .csv) file, \
                build an experimental object, then run <Exp_name>.match_sessions_to_files(files_dir, ext=''h5'')')
        elif 'h5' not in self.files:
             raise Exception('The session has not been matched with a deeplabcut (.nwb / .csv) file, \
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

        dlc_file_to_score = []
        for dlc_file in self.files['h5']:
            if search(camera_keyword, dlc_file):
                dlc_file_to_score.append(dlc_file)
        
        # Check how many DLC file are matching the video
        if len(dlc_file_to_score) == 0:
            raise DeepLabCutFileError(
                self.subject_ID, self.datime, camera_keyword)

        # TODO: implement a DLC network keyword argument 
        # for when there is more than one DLC file per vid 
        elif len(dlc_file_to_score) > 1:
            dlc_file_to_score = os.path.realpath(dlc_file_to_score[0])
            print(f' Warning: multiple DLC files matching {self.subject_ID} at {self.datetime} on camera {camera_keyword}: \n\r \
                will use {dlc_file_to_score}')
        
        # when one single DLC file match the video (normal case)
        else:
            dlc_file_to_score = dlc_file_to_score[0]

        # normalize file path format
        dlc_file_to_score = os.path.realpath(dlc_file_to_score)

        # load DLC data
        df_dlc = pd.read_hdf(dlc_file_to_score)
        if verbose:
            print(f'Successfully loaded DLC file: {os.path.split(dlc_file_to_score)[1]}')
        scorer = df_dlc.columns.get_level_values(0).unique().values[0]
        bodyparts = df_dlc.columns.get_level_values(1).unique().values.tolist()
        len_dlc = df_dlc.shape[0]

        for b in bodyparts:
            df_dlc.loc[:, (scorer, b, 'x')].mask(df_dlc.loc[:, (scorer, b, 'likelihood')] < p_thresh, inplace=True)
            df_dlc.loc[:, (scorer, b, 'y')].mask(df_dlc.loc[:, (scorer, b, 'likelihood')] < p_thresh, inplace=True)
            if three_dims:
                df_dlc.loc[:, (scorer, b, 'z')].mask(df_dlc.loc[:, (scorer, b, 'likelihood')] < p_thresh, inplace=True)

        coord_dict = dict()

        # block to determine which regions to keep
        if names_of_ave_regions or bodyparts_to_store:
            if names_of_ave_regions and bodyparts_to_store:
                regions_to_store = names_of_ave_regions + bodyparts_to_store
            elif names_of_ave_regions and not bodyparts_to_store:
                regions_to_store = bodyparts_to_ave
            elif not names_of_ave_regions and bodyparts_to_store:
                regions_to_store = bodyparts_to_store
                
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
                 
    def get_photometry_trials(self, conditions_list = None, cond_aliases = None,
            trig_on_ev = None, high_pass = None, low_pass = None, median_filt = None, 
            motion_corr = False, df_over_f = False, downsampling_factor = None,
            return_full_session = False, export_vars = ['analog_1','analog_2'],
            verbose = False):
            
        # TODO write docstrings
        
        if not isinstance(conditions_list, list):
            conditions_list= [conditions_list] 
        
        if not isinstance(export_vars, list):
            export_vars= [export_vars] 

        if not hasattr(self, 'photometry_path'):
            raise Exception('The session has not been matched with a .ppd file, \
                please run experiment.match_to_photometry_files(kvargs)')
        elif self.photometry_path == None:
            raise Exception('The session has no matching .ppd file, or no alignment \
                could be performed between rsync pulses')
        
        if motion_corr == True and high_pass == None and low_pass == None and median_filt == None:
            raise Exception('You need to high_pass and/or low_pass and/or median_filt the signal for motion correction')
        if df_over_f == True and motion_corr == False:
            raise Exception('You need motion correction to compute dF/F')

        try: 
            photometry_dict = import_ppd(self.photometry_path, high_pass=high_pass, low_pass=low_pass, median_filt=median_filt)
        except:
            raise Exception('could not load photometry file, check path')

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
        

        for condition_ID, conditions_dict in enumerate(conditions_list):
            # TEST the option of triggering on the first event of a trial

            trials_idx, timestamps_pycontrol = self.get_trials_times_from_conditions(conditions_dict, trig_on_ev=trig_on_ev)

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
            complete_mask = (photometry_idx + self.trial_window[0]/(1000/photometry_dict['sampling_rate']) >= 0) & (
                photometry_idx + self.trial_window[1] < len(photometry_dict[export_vars[0]])) 

            # complete_idx = np.where(complete_mask)
            trials_idx = np.array(trials_idx)
            photometry_idx = np.array(photometry_idx)

            trials_idx = trials_idx[complete_mask]           
            photometry_idx = photometry_idx[complete_mask]
            
            if verbose:
                print(f'condition {condition_ID} trials: {len(trials_idx)}')

            # TODO: Test when no trials in first or second condition
            if len(photometry_idx) == 0 :
                continue

            photometry_idx = [range(idx + int(self.trial_window[0]/(1000/photometry_dict['sampling_rate'])) ,
                idx + int(self.trial_window[1]/(1000/photometry_dict['sampling_rate']))) for idx in photometry_idx]


            if condition_ID == 0:
                # initialization of 3D numpy arrays
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
                # initialization of temp 3D numpy arrays
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

                photo_array = np.concatenate((photo_array, photo_array_temp), axis=0)

        if 'photo_array' in locals():
            photo_array = photo_array.swapaxes(2,1)
        else:
            # would occur anyway without the previous check, 
            # avoid it happening spontaneously on return.
            # useless but could be use to convey extra information to calling method
            raise UnboundLocalError()


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
            output_first_ev: bool = False):

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

        if trig_on_ev == None:
            trials_times = self.df_events.loc[(idx_joint),'timestamp'].values
        elif trig_on_ev not in self.events_to_process:
            raise Exception('trig_on_ev not in events_to_process')
        else:    
            trials_times = self.df_events.loc[(idx_joint), 'timestamp'].copy()
            
            

            # implemented due to error at following line
            # TypeError: 'numpy.int32' object is not iterable?
            # worked well before? unknown reason
            
            def find_min_time_list(x):
                if len(x) >= 1:
                    min_time = min([i for i in x if i>0], default=np.NaN)
                elif isinstance(x, int) and x > 0:
                    min_time = x
                elif isinstance(x, int) and x > 0:
                    min_time = np.NaN
                elif len(x) == 0:
                    min_time = np.NaN
                else:
                    print(x,type(x))
                return min_time


            df_ev_copy = self.df_events.copy()
            
            first_ev_times = df_ev_copy.loc[(idx_joint), trig_on_ev + '_trial_time'].apply(
                lambda x: find_min_time_list(x))
            

            #keep ony trial_times with a first event
            first_ev_times_nona = first_ev_times.dropna(inplace=False)
            
            # Add time of the first event to trial onset
            trials_times = trials_times.loc[first_ev_times_nona.index] + first_ev_times_nona
            # trials_times.dropna(inplace=True)
            # Keep only the index where events (trig_on_ev) were found
            idx_joint = trials_times.index.values.astype(int)
            # retransmorm trial_times as an array
            trials_times = trials_times.values.astype(int)
            first_ev_times = first_ev_times_nona.astype(int).values
        
            #print(idx_joint.shape,first_ev_times_nona.shape, first_ev_times.shape)
        if output_first_ev:
            return idx_joint, trials_times, first_ev_times
        else:    
            return idx_joint, trials_times

    def compute_behav_metrics(self, conditions_dict, events=None):
        ''' if two events are passed, time_delta between the first occurence of events[0]
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

#----------------------------------------------------------------------------------
# Experiment class
#----------------------------------------------------------------------------------

class Experiment():
    def __init__(self, folder_path, int_subject_IDs=True, verbose=False):
        '''
        Import all sessions from specified folder to create experiment object.  Only sessions in the 
        specified folder (not in subfolders) will be imported.
        Arguments:
        folder_path: Path of data folder.
        int_subject_IDs:  If True subject IDs are converted to integers, e.g. m012 is converted to 12.
        '''

        self.folder_name = os.path.split(folder_path)[1]
        self.path = folder_path

        # Import sessions.

        self.sessions = []
        try: # Load sessions from saved sessions.pkl file.
            with open(os.path.join(self.path, 'sessions.pkl'),'rb') as sessions_file:
                self.sessions = pickle.load(sessions_file)
            print('Saved sessions loaded from: sessions.pkl')
            # TODO: precise and refine use of this by_trial attribute
            self.by_trial = True
        except IOError:
            self.by_trial = False
            pass

        old_files = [session.file_name for session in self.sessions]
        files = os.listdir(self.path)
        new_files = [f for f in files if f[-4:] == '.txt' and f not in old_files]

        if len(new_files) > 0:
            if verbose:
                print('Loading new data files..')
            self.by_trial = False
            for file_name in new_files:
                try:
                    self.sessions.append(Session(os.path.join(self.path, file_name), int_subject_IDs))
                except Exception as error_message:
                    if verbose:
                        print('Unable to import file: ' + file_name)
                        print(error_message)

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


    def save(self):
        '''Save all sessions as .pkl file. Speeds up subsequent instantiation of 
        experiment as sessions do not need to be reimported from data files.''' 
        
        with open(os.path.join(self.path, 'sessions.pkl'),'wb') as sessions_file:
            pickle.dump(self.sessions, sessions_file)

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
        if task_names == 'all':
            task_names = self.task_names
        if not isinstance(task_names, list):
            task_names = [task_names]
        
        # select valid sessions subject/task/time specific
        valid_sessions = [s for s in self.sessions if 
            s.subject_ID in subject_IDs
            and when_func(s) and s.task_name in task_names]
        
        return valid_sessions


    def process_exp_by_trial(self, trial_window, timelim, tasksfile):
        # create emtpy list to store idx of sessions without trials,
        # can be extended to detect all kind of faulty sessions.
        sessions_idx_to_remove = []
        
        for s_idx, s in enumerate(self.sessions):

            self.sessions[s_idx] = s.get_session_by_trial(trial_window, timelim, tasksfile)
            
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
        self.trial_window = trial_window
        self.by_trial = True


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
        '''
        Take all behavioural data from the relevant sessions
        and trials and assemble it as an Event_Dataset
        instance for further analyses.
        '''
        print('caca')
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

        if isinstance(conditions_list, dict):
            conditions_list = [conditions_list]


        # Checks to put in dedicated function until here

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
                
                print('pipi')                         
                for session in sessions:
                    try:
                        df_events = session.df_events.copy()
                        df_conditions = session.df_conditions.copy()
                    except:
                        print(session.subject_ID, session.datetime)
                        continue
                    print(conditions_list)
                    df_conditions['condition_ID'] = nan
                    
                    # df_events['condition'] = nan
                    df_conditions['condition'] = nan
                    
                    col_to_modify = [ev + '_trial_time' for ev in session.events_to_process]    
                    col_idxs = [df_events.columns.to_list().index(col)+2 for col in col_to_modify]
                    
                    # df_events['time_to_ev'] = nan
                    idx_all_cond = []
                    trials_times_all_cond = []
                    first_ev_times_all_cond = []
                    events_aggreg = pd.DataFrame()
                    print(conditions_list)
                    for cond_idx, conditions_dict in enumerate(conditions_list):
                        
                        if trig_on_ev:
                            idx_joint, trials_times, first_ev_times = session.get_trials_times_from_conditions(
                                conditions_dict = conditions_dict,
                                trig_on_ev = trig_on_ev, output_first_ev = True)
                            print(conditions_dict, len(idx_joint), trials_times.shape, first_ev_times.shape)
                        
                            df_ev_cond = df_events.loc[idx_joint,:].copy()
                            df_ev_cond = df_ev_cond.reset_index()
                            # print(type(first_ev_times),df_ev_cond.index.values)
                            for ridx, row in enumerate(df_ev_cond.itertuples()): 
                                # print(ridx, row.Index)
                                for c, col_name in enumerate(col_to_modify):
                                    
                                    df_ev_cond.at[row.Index, col_name] = np.array(row[col_idxs[c]]) - first_ev_times[row.Index]

                            events_aggreg = pd.concat([events_aggreg,df_ev_cond])
                        else: 
                            idx_joint, trials_times = session.get_trials_times_from_conditions(
                                conditions_dict = conditions_dict,
                                trig_on_ev = trig_on_ev,  output_first_ev = False)

                            df_ev_cond = df_events.loc[idx_joint,:].copy()
                            
                            # turn lists to arrays anyway
                            # (lists inherited from early implementation)
                            for col_name in col_to_modify:
                                df_ev_cond[col_name]=df_ev_cond[col_name].apply(lambda x : np.array(x))
                            
                            events_aggreg = pd.concat([events_aggreg,df_ev_cond])

                        idx_all_cond = np.concatenate([idx_all_cond, idx_joint])
                        trials_times_all_cond = np.concatenate([trials_times_all_cond, trials_times])
                        print('prout')
                        if isinstance(trig_on_ev, str):
                            first_ev_times_all_cond = np.concatenate([first_ev_times_all_cond, first_ev_times])
                        
                        df_conditions.loc[idx_joint, 'group_ID'] = group_ID
                        df_conditions.loc[idx_joint, 'condition_ID'] = cond_idx
                        if cond_aliases:
                            # df_events.loc[idx_joint, 'condition'] = cond_aliases[cond_idx]
                            df_conditions.loc[idx_joint, 'condition'] = cond_aliases[cond_idx]
                    
                    idx_all_cond = idx_all_cond.astype(int)
                    # df_events.dropna(subset=['condition_ID'], inplace=True)
                    # df_conditions.dropna(subset=['condition_ID'], inplace=True)
                    
                    # df_events = df_events.loc[idx_all_cond,:]
                    df_conditions = df_conditions.loc[idx_all_cond,:].reset_index()
                    
                    df_conditions['session_nb'] = session.number
                    df_conditions['group_ID'] = group_ID
                    df_conditions['subject_ID'] = subject_ID

                    # consider implementing session_nb metadata in Session methods

                    df_events_exp = pd.concat([df_events_exp, events_aggreg], ignore_index = True)
                    df_conditions_exp = pd.concat([df_conditions_exp, df_conditions], ignore_index = True)
        
        df_conditions_exp['condition_ID'] = df_conditions_exp['condition_ID'].astype(int)         
        ev_dataset = Event_Dataset(df_events_exp, df_conditions_exp)

        ev_dataset.conditions = conditions_list
        ev_dataset.cond_aliases = cond_aliases
        # TODO: remove, should be a temporary check
        if hasattr(self, 'trial_window'):
            ev_dataset.set_trial_window(self.trial_window)
        return ev_dataset

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
        
        if self.by_trial == False:
            raise Exception('Process experiment by trial first: experiment.process_exp_by_trial(trial_window, timelim, tasksfile)')

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
                            conditions_list = conditions_list, cond_aliases = cond_aliases,
                            camera_fps = camera_fps, camera_keyword = camera_keyword,
                            bodyparts_to_ave = bodyparts_to_ave, names_of_ave_regions = names_of_ave_regions,
                            normalize_between = normalize_between, bins_nb = bins_nb, p_thresh = p_thresh,
                            bodyparts_to_store = bodyparts_to_store,
                            trig_on_ev = trig_on_ev, three_dims = three_dims, 
                            return_full_session = False, verbose = verbose)
                                        
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
            when = 'all',
            task_names = 'all',
            trig_on_ev = None,
            high_pass = None, 
            low_pass = None, 
            median_filt = None,
            motion_corr = False, 
            df_over_f = False, 
            downsampling_factor = None,
            export_vars = ['analog_1','analog_2'],
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
                sessions = [session for session in sessions if session.photometry_path is not None]
                # if this subject has no photometry data
                if sessions == []:
                    continue

                for s_idx, session in enumerate(sessions):
                    # forward arguments to the session method:

                    if verbose:
                        print(f'Processing subject {session.subject_ID} at: {session.datetime_string}')

                    try:
                        df_meta_photo, col_names_numpy, photometry_array, fs = session.get_photometry_trials(
                            conditions_list = conditions_list, cond_aliases = cond_aliases, trig_on_ev=trig_on_ev,
                            high_pass = high_pass, low_pass = low_pass, median_filt = median_filt,
                            motion_corr = motion_corr, df_over_f = df_over_f, downsampling_factor = downsampling_factor, 
                            return_full_session = False, export_vars = export_vars, verbose = verbose)
                    
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
                raise Exception(f'The following group: {subject_IDs} do not contain photometry trials. \
                    \r\n consider looking for more sessions with when, or broadening conditions')
    
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

    # consider deleting and using match_sessions_to_files,
    # then only compute rsync aligner at extraction (or not
    # since it is good to know in advance which photometry
    # files are actually useful)
    def match_to_photometry_files(self, 
            photometry_dir, 
            rsync_chan: int = 2, 
            verbose: bool = True):
        '''
        This function is a class method for Experiment objects. 
        For each session, it checks into an horizontal photometry file repository,
        trying to find a .ppd file matching subject and date and takes the closest
        file. It then try to create a rsync aligment object into the corresponding
        session if the pulses match.

                Parameters:
                        self (Experiment): An Experiment object instance
                        photometry_dir (str): Path of the photometry repository
                            Note: On windows, double antislash must be entered
                            between each level. 
                            e.g.: 'C:\\Users\\Documents\\GitHub\\photometry_repo'
                        rsync_chan (int): Channel on which pulses have been
                            recorded on the py_photometry device.
                        verbose (bool): display match/no match messages for each file

                Returns:
                        None
                    
        '''
            
        pycontrol_subjects = [session.subject_ID for session in self.sessions]
        pycontrol_datetime = [session.datetime for session in self.sessions]

        all_photo_files = [f for f in os.listdir(photometry_dir) if os.path.isfile(
        os.path.join(photometry_dir, f)) and ".ppd" in f]

        # parsing filenames to extract integer of subject_name (aka subject_ID)
        photometry_subjects = [int(re.split('(\d+)', f.split('-')[0])[1]) for f in all_photo_files]
        # only extract datetime string from the filename (leave appart subject ID and .ppd extension)
        photometry_datestr = [f.split('-',1)[1][:-4] for f in all_photo_files]
        # convert date-time string to datetime format
        photometry_datetime = [datetime.strptime(datestr, "%Y-%m-%d-%H%M%S") for datestr in photometry_datestr]

        for id_f, pycontrol_subject in enumerate(pycontrol_subjects):
            # find photometry files which match subject and date of the pycontrol session
            subject_date_match_idx = [photo_idx for (photo_idx, photo_subject) in enumerate(photometry_subjects) 
                if photo_subject == pycontrol_subject and
                photometry_datetime[photo_idx].date() == pycontrol_datetime[id_f].date()]

            # if a match is detected
            if len(subject_date_match_idx) > 0:
                # compute absolute time difference of all possible files matching
                time_diff = [abs(pycontrol_datetime[id_f] - photometry_datetime[match_idx]).seconds
                    for match_idx in subject_date_match_idx]
                # extract the idx of the photometry file with the shortest time difference  
                # compared to the pycontrol files
                best_match_idx = subject_date_match_idx[time_diff.index(min(time_diff))]
                
                # try to align times with rsync
                try:
                    # Gives KeyError exception if no rsync pulses on pycontrol file
                    pycontrol_rsync_times = self.sessions[id_f].times['rsync']
                
                    photometry_dict = import_ppd(os.path.join(photometry_dir,all_photo_files[best_match_idx]))
                    
                    photometry_rsync_times = photometry_dict['pulse_times_' + str(rsync_chan)]

                    pyphoto_aligner = Rsync_aligner(pulse_times_A= pycontrol_rsync_times, 
                        pulse_times_B= photometry_rsync_times, plot=False)
                    
                    if verbose:
                        print('pycontrol: ', pycontrol_subjects[id_f], pycontrol_datetime[id_f],
                        '/ pyphotometry: ', photometry_datetime[best_match_idx], ' : rsync does match')
                    
                    self.sessions[id_f].photometry_rsync = pyphoto_aligner
                    self.sessions[id_f].photometry_path = os.path.join(photometry_dir,all_photo_files[best_match_idx])

                    # # TODO: take out that mean and std computation and possibly bring to get_photometry_trials(Session_method)
                    # # If no filter limit specified, take raw values
                    # if all([low_pass == None, high_pass == None]):
                    #     self.sessions[id_f].photometry_mean = [np.mean(photometry_dict['analog_1']), 
                    #         np.mean(photometry_dict['analog_2'])]

                    #     print(photometry_dict['analog_1'])

                    #     self.sessions[id_f].photometry_std = [np.std(photometry_dict['analog_1']), 
                    #         np.std(photometry_dict['analog_2'])]
                    # # Otherwise, take filtered values
                    # else:
                    #     self.sessions[id_f].photometry_mean = [np.mean(photometry_dict['analog_1_filt']), 
                    #         np.mean(photometry_dict['analog_2_filt'])]

                    #     self.sessions[id_f].photometry_std = [np.std(photometry_dict['analog_1_filt']), 
                    #         np.std(photometry_dict['analog_2_filt'])]
                
                # if rsync aligner fails    
                except (RsyncError, ValueError):
                    self.sessions[id_f].photometry_rsync = None
                    self.sessions[id_f].photometry_path = None
                    if verbose:
                        print('pycontrol: ', pycontrol_subjects[id_f], pycontrol_datetime[id_f],
                        '/ pyphotometry: ', photometry_datetime[best_match_idx], ' : rsync does not match')

                except KeyError:
                    self.sessions[id_f].photometry_rsync = None
                    self.sessions[id_f].photometry_path = None
                    if verbose:
                        print('pycontrol: ', pycontrol_subjects[id_f], pycontrol_datetime[id_f],
                        '/ pyphotometry: ', photometry_datetime[best_match_idx], ' : rsync does not match')
            
            # if there is no subject + date match in .ppd files
            else: 
                self.sessions[id_f].photometry_rsync = None
                self.sessions[id_f].photometry_path = None
                if verbose:
                    print('pycontrol: ', pycontrol_subjects[id_f], pycontrol_datetime[id_f],
                    '/ pyphotometry: no file matching both subject and date')

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
