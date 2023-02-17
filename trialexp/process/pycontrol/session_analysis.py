
from cmath import isnan, nan

import os
from pathlib import Path
import pickle
import re
from datetime import datetime
import warnings
import itertools

from collections import namedtuple
from operator import itemgetter

import numpy as np
import pandas as pd

from math import ceil
from scipy.signal import butter, filtfilt, decimate
from scipy.stats import linregress, zscore
from trialexp.process.data_import import Event
from trialexp.process.pycontrol.utils import find_if_event_within_timelim, find_last_time_before_list

    
def add_time_rel_trigger(df_events, trigger_time, trigger_name, col_name, trial_window):
    #Add new time column to the event data, aligned to the trigger time
    # the new time column can also be negative, the search window in which the trigger will apply
    # is defined in time_window
    df = df_events.copy()
    df[col_name] = np.nan
    df['trigger'] = trigger_name

    #TODO: can this be overalpping?

    trial_nb = 1
    for t in trigger_time:
        td = df.time - t
        idx = (trial_window[0]<td) & (td<trial_window[1])
        df.loc[idx, col_name] =  df[idx].time - t
        trial_nb += 1

    return df

def assign_val_rel_trigger(df_events, trigger_time, col_name, value, trial_window):
    # Assign a value to the col around a trigger in the trial_window
    df = df_events.copy()
    df[col_name] = np.nan

    #TODO: can this be overalpping?

    trial_nb = 1
    for t in trigger_time:
        td = df.time - t
        idx = (trial_window[0]<td) & (td<trial_window[1])
        df.loc[idx, col_name] =  value
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
    
    
def extract_trial_by_trigger(df_events, trigger, event2analysis, trial_window, subject_ID, datetime_obj):
    
    # add trial number and calculate the time from trigger
    trigger_time = df_events[(df_events.name==trigger) & (df_events.type == 'state')].time
    df_events = add_trial_nb(df_events, trigger_time,trial_window) #add trial number according to the trigger
    df_events = add_time_rel_trigger(df_events, trigger_time, trigger, 'trial_time', trial_window) #calculate time relative to trigger
    df_events.dropna(inplace=True)
    
    
    # sometimes state variable is also included in the event2analysis, try to account for them
    # state2check = set(df_events.state).intersection(event2analysis)
    # if len(state2check)>0:
    #     state2check = state2check+trigger
    
    # Filter out events we don't want
    # df_events = df_events[df_events.name.isin(event2analysis)]

    # # group events according to trial number and event name
    # df_events_trials = df_events.groupby(['trial_nb', 'name']).agg(list)
    # df_events_trials = df_events_trials.loc[:, ['trial_time']]
    # df_events_trials = df_events_trials.unstack('event_name') #convert the event names to columns
    # df_events_trials.columns = df_events_trials.columns.droplevel() # dropping the multiindex of the columns

    # if 'state_change' in df_events_trials.columns: 
    #     df_events_trials = df_events_trials.drop(columns=['state_change'])

    # # rename the column for compatibility
    # df_events_trials.columns = [col+'_trial_time' for col in df_events_trials.columns]

    # # add uuid
    # df_events_trials['trial_nb'] = df_events_trials.index.values
    # df_events_trials['uid'] = df_events_trials['trial_nb'].apply(lambda x: f'{subject_ID}_{datetime_obj.date()}_{datetime_obj.time()}_{x}')

    # # fill the new_df with timestamps of trigger and trigger types
    # df_events_trials['timestamp'] = trigger_time.values
    # df_events_trials['trigger'] = trigger

    # # validate trials in function of the time difference between trials (must be > at length of trial_window)
    # df_events_trials['valid'] = df_events_trials['timestamp'].diff() > trial_window[0]

    # # validate first trial except if too early in the session
    # if df_events_trials['timestamp'].iloc[0] > abs(trial_window[0]):
    #    df_events_trials.loc[1, 'valid'] = True
    
    # return df_events_trials, df_events
    return df_events

def compute_conditions_by_trial(df_events_trials, conditions):

    df_conditions = df_events_trials[['uid','trigger','valid']].copy()
    for con in conditions:
        # Find the corresponding trial time
        colname = con+'_trial_time'
        if colname in df_events_trials.columns:
            df_conditions[con] = df_events_trials[colname].notna()
        else:
            df_conditions[con] = False
        
    return df_conditions

def compute_success(df_events_trials, df_cond, task_name, triggers=None, timelim=None):
    """computes success trial numbers

    This methods includes task_name-specific definitions of successful trials.
    The results are stored in the 'success' columns of self.df_events and self.df_conditions as bool (True or False).
    """
    df_conditions = df_cond.copy()
    df_conditions['success'] = False
    # self.df_events['success'] = False
    #print(self.task_name)a
    df_events = df_events_trials.copy()
    
    # To perform for all Go-NoGo variants of the task (list below)
    if task_name in ['reaching_go_nogo', 'reaching_go_nogo_jc', 'reaching_go_nogo_opto_continuous',
        'reaching_go_nogo_opto_sinusoid' , 'reaching_go_nogo_opto_sinusoid_spout', 
        'reaching_go_nogo_reversal', 'reaching_go_nogo_reversal_incentive',
        'reaching_go_nogo_touch_spout']:
        # self.triggers[0] refers to CS_Go triggering event most of the time whereas self.triggers[1] refers to CS_NoGo
        # find if spout event within timelim for go trials
        go_success = df_events.loc[
            (df_events[df_events.trigger == triggers[0]].index),'spout_trial_time'].apply(
                lambda x: find_if_event_within_timelim(x, timelim))
        go_success_idx = go_success[go_success == True].index
        #print(go_success_idx)
        # categorize successful go trials which have a spout event within timelim
        df_conditions.loc[(go_success_idx),'success'] = True
        # df_events.loc[(go_success_idx),'success'] = True
        # find if no bar_off event within timelim for nogo trials
        nogo_success = ~df_events.loc[
            (df_events[df_events.trigger == triggers[1]].index),'bar_off_trial_time'].apply(
                lambda x: find_if_event_within_timelim(x, timelim))
        nogo_success_idx = nogo_success[nogo_success == True].index
        #print(go_success_idx, nogo_success_idx)
        # categorize as successful trials which contains no bar_off but are not Go trials
        # nogo_success_idx = nogo_success_idx.get_level_values('trial_nb').difference(
        #     df_conditions[df_conditions['trigger'] == self.triggers[0]].index.get_level_values('trial_nb'))
        df_conditions.loc[(nogo_success_idx),'success'] = True
        # df_events.loc[(nogo_success_idx),'success'] = True

    # To perform for simple pavlovian Go task, 
    elif task_name in ['train_Go_CS-US_pavlovian','reaching_yp', 'reaching_test','reaching_test_CS',
        'train_CSgo_US_coterminated','train_Go_CS-US_pavlovian', 'train_Go_CS-US_pavlovian_with_bar', 
        'pavlovian_nobar_nodelay']:

        # self.triggers[0] refers to CS_Go triggering event most of the time whereas self.triggers[1] refers to CS_NoGo
        # find if spout event within timelim for go trials
        go_success = df_events.loc[
            (df_events[df_events.trigger == triggers[0]].index),'spout_trial_time'].apply(
            lambda x: find_if_event_within_timelim(x, timelim))
        go_success_idx = go_success[go_success == True].index
        # categorize successful go trials which have a spout event within timelim
        df_conditions.loc[(go_success_idx),'success'] = True
        # df_events.loc[(go_success_idx),'success'] = True

    # To perform for cued-uncued version of the go task
    elif task_name in ['reaching_go_spout_cued_uncued', 'cued_uncued_oct22']:
        # reformatting trigger name for that one task, with lower case
        if task_name in ['cued_uncued_oct22']:
            df_conditions.trigger = df_conditions.trigger.str.lower()
            df_events.trigger = df_events.trigger.str.lower()

        # for cued trials, find if spout event within timelim           
        cued_success = df_events.loc[
            (df_events[df_conditions.trigger == 'cued'].index),'spout_trial_time'].apply(
            lambda x: find_if_event_within_timelim(x, timelim))
        cued_success_idx = cued_success[cued_success == True].index

        # for uncued trials, just check if there is a spout event after trial start
        uncued_success = df_events.loc[
            (df_events[df_conditions.trigger == 'uncued'].index),'spout_trial_time'].apply(
            lambda x: x[-1] > 0 if len(x) > 0 else False)
        uncued_success_idx = uncued_success[uncued_success == True].index
        
        # categorize successful go trials
        df_conditions.loc[np.hstack((cued_success_idx.values, uncued_success_idx.values)), 'success'] = True
        # df_events.loc[np.hstack((cued_success_idx.values, uncued_success_idx.values)),'success'] = True
        # print(task_name, self.subject_ID, self.datetime_string, len(cued_success_idx), len(uncued_success_idx))


    elif task_name in ['reaching_go_spout_nov22']:
        reach_time_before_reward = df_events.loc[:,['spout_trial_time','US_end_timer_trial_time']].apply(
            lambda x: find_last_time_before_list(x['spout_trial_time'], x['US_end_timer_trial_time']), axis=1)  
        # select only trials with a spout event before a US_end_timer event
        reach_bool = reach_time_before_reward.notnull()
        # select trial where the hold time was present (not aborted)
        reach_success_bool = reach_bool & (df_conditions.trigger =='busy_win')
        # set these trials as successful
        df_conditions.loc[(reach_success_bool), 'success'] = True

    # To perform for delayed tasks (check whether a US_end_timer was preceded by a spout)
    elif task_name in ['reaching_go_spout_bar_dual_all_reward_dec22', 
        'reaching_go_spout_bar_dual_dec22', 'reaching_go_spout_bar_nov22']:

        reach_time_before_reward = df_events.loc[:,['spout_trial_time','US_end_timer_trial_time']].apply(
                lambda x: find_last_time_before_list(x['spout_trial_time'], x['US_end_timer_trial_time']), axis=1)    
        # select only trials with a spout event before a US_end_timer event
        reach_bool = reach_time_before_reward.notnull()
        # select trial where the hold time was present (not aborted)
        reach_success_bool = reach_bool & df_conditions.waiting_for_spout
        # set these trials as successful
        df_conditions.loc[(reach_success_bool), 'success'] = True


    # Reorder columns putting trigger, valid and success first for more clarity
    # col_list = list(df_conditions.columns.values)
    # col_to_put_first = ['trigger', 'success','valid']
    # for c in col_to_put_first:
    #     col_list.remove(c)
    # col_list = ['trigger', 'success','valid'] + col_list
    # df_conditions = df_conditions[col_list]

    

    return df_conditions

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
        return conditions, triggers, events_to_process


    def create_metadata_dict(self, trial_window, timelim):
        metadata_dict = {
            'subject_ID' : self.subject_ID,
            'datetime' : self.datetime,
            'timelim': timelim,
            'task' : self.task_name,
            'trial_window' : trial_window,
            'com_port' : self.setup_ID
        }
        return metadata_dict


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

    