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
from trialexp.process.pycontrol.event_filters import extract_clean_trigger_event
from workflow.scripts import settings
from pathlib import Path
import pickle 
import seaborn as sns

#%% Load inputs
(sinput, soutput) = getSnake(locals(), 'workflow/pycontrol.smk',
    [settings.debug_folder + '/processed/log/task_specific_analysis.done'],
    'task_specifc_analysis')

# %% load data
df_event = pd.read_pickle(sinput.event_dataframe)
xr_session =  xr.load_dataset(sinput.xr_session) # load for modification
xr_photometry = xr.load_dataset(sinput.xr_photometry, engine='h5netcdf')
with open(sinput.pyphoto_aligner, 'rb') as f:
    pyphoto_aligner = pickle.load(f)
#%%
trial_window = xr_session.attrs['trial_window']

#%%

add_event_data(df_event, extract_clean_trigger_event, trial_window,
               pyphoto_aligner, xr_photometry, xr_photometry.event_time, 
               'zscored_df_over_f', 'clean_busy_win', xr_photometry.attrs['sampling_rate'], 
              filter_func_kwargs = dict(clean_window = [-1000,1000], target_event_name='busy_win'))

#%%
add_event_data(df_event, extract_clean_trigger_event, trial_window,
               pyphoto_aligner, xr_photometry, xr_photometry.event_time, 
               'zscored_df_over_f', 'clean_spout', xr_photometry.attrs['sampling_rate'], 
              filter_func_kwargs = dict(clean_window = [-1000,1000], target_event_name='spout', ignore_events=['spout_off','bar_off']))

# %% Re-save the files

dataset_binned = xr_photometry.coarsen(time=10, event_time=10, boundary='trim').mean()

# add the data back to xr_session
xr_session['clean_spout_zscored_df_over_f'] = dataset_binned['clean_spout_zscored_df_over_f']
xr_session['clean_busy_win_zscored_df_over_f']  = dataset_binned['clean_busy_win_zscored_df_over_f'] 

xr_photometry.to_netcdf(sinput.xr_photometry, engine='h5netcdf')
xr_session.to_netcdf(sinput.xr_session, engine='h5netcdf')
