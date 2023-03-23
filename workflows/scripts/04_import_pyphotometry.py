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
import seaborn as sns 
import numpy as np
import logging

#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflows/spout_bar_nov22.smk',
#  ['Z:/Julien/Data/head-fixed/_Other/test_folder/by_session_folder/JC317L-2022-12-16-173145\processed/xr_photometry.nc'],
  ['Z:/Teris/ASAP/expt_sessions/kms064-2023-02-08-100449/processed/xr_photometry.nc'],
  'import_pyphotometry')


#%% Load pyphotometry file
fn = glob(sinput.photometry_folder+'\*.ppd')[0]
data_photometry = import_ppd(fn)
data_photometry = denoise_filter(data_photometry)
data_photometry = motion_correction(data_photometry)
data_photometry = compute_df_over_f(data_photometry, low_pass_cutoff=0.001)

#%% Conver to xarray
skip_var = ['analog_1_est_motion','analog_1_corrected', 'analog_1_baseline_fluo']
dataset = photometry2xarray(data_photometry, skip_var = skip_var)

#%% Load pycontrol file
df_pycontrol = pd.read_pickle(sinput.pycontrol_dataframe)
df_event = pd.read_pickle(sinput.event_dataframe)
df_condition = pd.read_pickle(sinput.condition_dataframe)

# %% synchornize pyphotometry with pycontrol
rsync_time = df_pycontrol[df_pycontrol.name=='rsync'].time
photo_rsync = dataset.attrs['pulse_times_2']

#align pycontrol time to pyphotometry
pyphoto_aligner = Rsync_aligner(pulse_times_A= rsync_time, 
                pulse_times_B= photo_rsync, plot=False) #align pycontrol time to pyphotometry time


#%% Add in the relative time to different events
df_trigger = df_pycontrol[df_pycontrol.name=='hold_for_water']

time_rel = get_rel_time(df_trigger.time, [-2000,3000], pyphoto_aligner, dataset.time)

rel_time_hold_for_water = xr.DataArray(
    time_rel, coords={'time':dataset.time}, dims=('time')
)

dataset['rel_time_hold_for_water'] = rel_time_hold_for_water

#%% Add in trial number
trial_nb = resample_event(pyphoto_aligner, dataset.time, df_event.time, df_event.trial_nb)
trial_nb_xr = xr.DataArray(
    trial_nb.astype(np.int16), coords={'time':dataset.time}, dims=('time')
)

dataset['trial_nb'] = trial_nb_xr

dataset = dataset.sel(time = dataset.trial_nb>0) #remove data outside of task

dataset.to_netcdf(soutput.xr_photometry, engine='h5netcdf')

# %%
# Bin the data such that we only have 1 data point per time bin
dataset_binned = bin_dataset(dataset, 50) 

#%% Merge conditions

xr_condition = make_condition_xarray(df_condition, dataset_binned)

# %%
xr_session = xr.merge([xr_condition, dataset_binned], compat='override')

xr_session.to_netcdf(soutput.xr_session, engine='h5netcdf')

# %%

sns.lineplot(x='rel_time_hold_for_water',hue='spout',
             y='analog_1_df_over_f', data=xr_session)
# %%
