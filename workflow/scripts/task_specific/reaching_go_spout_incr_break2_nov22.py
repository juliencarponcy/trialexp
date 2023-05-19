'''
Additional analysis for a particular task
'''
#%%
from snakehelper.SnakeIOHelper import getSnake
from trialexp.process.pyphotometry.utils import *
from glob import glob
import xarray as xr
from trialexp.utils.rsync import *
import pandas as pd 
import numpy as np
from trialexp.process.pycontrol import event_filters
from trialexp.process.pycontrol.event_filters import extract_event_time
from workflow.scripts import settings
from pathlib import Path
import pickle 

#%% Load inputs
(sinput, soutput) = getSnake(locals(), 'workflow/pycontrol.smk',
   [settings.debug_folder + '/processed/log/task_specific_analysis.done'],
  'task_specifc_analysis')

# %%
df_event = pd.read_pickle(sinput.event_dataframe)
xr_session =  xr.open_dataset(sinput.xr_session)
xr_photometry = xr.open_dataset(sinput.xr_photometry)
with open(sinput.pyphoto_aligner, 'rb') as f:
    pyphoto_aligner = pickle.load(f)
#%%
trial_window = xr_session.attrs['trial_window']

#%%
add_event_data(df_event, event_filters.get_first_bar_off, trial_window,
               pyphoto_aligner, xr_photometry, xr_photometry.event_time, 
               'analog_1_df_over_f', 'first_bar_off', xr_photometry.attrs['sampling_rate'])

# %%
def extract_clean_trigger_event(df_trial, target_event, clean_window, ignore_events=None):
    # This function will extract an event with nothing happening (except those in ignore_events) before and after the clean window
    
    # extract all event within the clean_window
    target_time = df_trial[df_trial['name'] == target_event].iloc[0].time
    idx = (df_trial.time > (target_time+clean_window[0])) & (df_trial.time <(target_time+clean_window[1]))
    if ignore_events is not None:
      idx = idx & ~df_trial['name'].isin(ignore_events)
    
    if sum(idx) ==1 and df_trial.loc[idx].iloc[0]['name'] == target_event:
      return df_trial.loc[idx]
    
    
df_event_filtered = df_event.groupby('trial_nb',group_keys=True).apply(extract_clean_trigger_event, 
                                                                       target_event='busy_win', 
                                                                       clean_window=[-1000,1000],
                                                                       ignore_events=['button_press'])


