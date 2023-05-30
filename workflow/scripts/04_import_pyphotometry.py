'''
This script import pyphotometry and perform the necessary processing
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
   [settings.debug_folder + '/processed/xr_photometry.nc'],
  'import_pyphotometry')


#%% Load pyphotometry file
fn = list(Path(sinput.photometry_folder).glob('*.ppd'))[0]
data_photometry = import_ppd(fn)
data_photometry = denoise_filter(data_photometry)
data_photometry = motion_correction_win(data_photometry)
data_photometry = compute_df_over_f(data_photometry, low_pass_cutoff=0.001)
data_photometry = compute_zscore(data_photometry)

#%% Convert to xarray
skip_var = ['analog_1_est_motion','analog_1_corrected', 'analog_1_baseline_fluo', 'analog_2_baseline_fluo']
dataset = photometry2xarray(data_photometry, skip_var = skip_var)
 
#%% Load pycontrol file
df_pycontrol = pd.read_pickle(sinput.pycontrol_dataframe)
df_event = pd.read_pickle(sinput.event_dataframe)
df_condition = pd.read_pickle(sinput.condition_dataframe)
trial_window = df_event.attrs['trial_window']

# %% synchornize pyphotometry with pycontrol
rsync_time = df_pycontrol[df_pycontrol.name=='rsync'].time
photo_rsync = dataset.attrs['pulse_times_2']

#align pyphotometry time to pycontrol
pycontrol_aligner = Rsync_aligner(pulse_times_A=  photo_rsync,
                pulse_times_B= rsync_time, plot=False) #align pycontrol time to pyphotometry time


dataset = align_photometry_to_pycontrol(dataset, df_event, pycontrol_aligner)


#%% Add in the relative time to different events
event_period = (trial_window[1] - trial_window[0])/1000
sampling_freq = 1000
event_time_coord= np.linspace(trial_window[0], trial_window[1], int(event_period*sampling_freq)) #TODO

#%% Add trigger
trigger = df_event.attrs['triggers'][0]
add_event_data(df_event, event_filters.get_first_event_from_name,
               trial_window, dataset, event_time_coord, 
               'zscored_df_over_f', trigger, dataset.attrs['sampling_rate'],
               filter_func_kwargs={'evt_name':trigger})

#%% Add first bar off
add_event_data(df_event, event_filters.get_first_bar_off, trial_window, dataset,event_time_coord, 
               'zscored_df_over_f', 'first_bar_off', dataset.attrs['sampling_rate'])

#%% Add first spout
add_event_data(df_event, event_filters.get_first_spout, trial_window, dataset, event_time_coord, 
               'zscored_df_over_f', 'first_spout', dataset.attrs['sampling_rate'])

#%% Add last bar_off before first spout

add_event_data(df_event, event_filters.get_last_bar_off_before_first_spout, trial_window,
               dataset,event_time_coord, 
               'zscored_df_over_f', 'last_bar_off', dataset.attrs['sampling_rate'])


#%%
dataset = dataset.sel(time = dataset.trial>=0) #remove data outside of task

# add in all metadata
dataset.attrs.update(df_pycontrol.attrs)
dataset.attrs.update(df_event.attrs)
dataset.to_netcdf(soutput.xr_photometry, engine='h5netcdf')

# %%
# Bin the data such that we only have 1 data point per time bin
# bin according to 50ms time bin, original sampling frequency is at 1000Hz
dataset_binned = dataset.coarsen(time=10, event_time=10, boundary='trim').mean()
dataset_binned.attrs.update(dataset.attrs)

#%% Merge conditions
df_condition = df_condition[df_condition.index>0]
ds_condition = xr.Dataset.from_dataframe(df_condition)
xr_session = xr.merge([ds_condition, dataset_binned])

#add in session_id so that we can combine multiple sessions easily later
xr_session = xr_session.expand_dims({'session_id':[dataset.attrs['session_id']]})

xr_session.attrs.update(dataset_binned.attrs)

#Save the final dataset
xr_session.to_netcdf(soutput.xr_session, engine='h5netcdf')

# %%
#Also save the pyphoto_aligner
with open(soutput.pycontrol_aligner, 'wb') as f:
  pickle.dump(pycontrol_aligner, f)
# %%
