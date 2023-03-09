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
from scipy.interpolate import interp1d
import seaborn as sns 
import numpy as np
#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflows/spout_bar_nov22.smk',
#  ['Z:/Julien/Data/head-fixed/_Other/test_folder/by_session_folder/JC317L-2022-12-16-174417\processed/df_photometry.pkl'],
  ['Z:/Teris/ASAP/expt_sessions/kms064-2023-02-08-100449/processed/df_photometry.nc'],
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
dataset.to_netcdf(soutput.df_photometry, engine='h5netcdf')

#%% Load pycontrol file
df_pycontrol = pd.read_pickle(sinput.pycontrol_dataframe)
df_event = pd.read_pickle(sinput.event_dataframe)
df_condition = pd.read_pickle(sinput.condition_dataframe)

# %% synchornize pyphotometry with pycontrol
rsync_time = df_pycontrol[df_pycontrol.name=='rsync'].time
photo_rsync = dataset.attrs['pulse_times_2']
pyphoto_aligner = Rsync_aligner(pulse_times_A= rsync_time, 
                pulse_times_B= photo_rsync, plot=False) #align pycontrol time to pyphotometry time


#%% Calulate the relative time
def get_rel_time(trigger_timestamp, window, aligner, ref_time):
    # Calculate the time relative to a trigger timestamp)
    ts = aligner.A_to_B(trigger_timestamp)
    time_relative = np.ones_like(ref_time)*np.NaN
    
    for t in ts: 
        d = ref_time-t
        idx = (d>window[0]) & (d<window[1])
        time_relative[idx] = d[idx]
        
    return time_relative

df_trigger = df_pycontrol[df_pycontrol.name=='hold_for_water']

time_rel = get_rel_time(df_trigger.time, [-2000,3000], pyphoto_aligner, dataset.time)


#%%
rel_time_hold_for_water = xr.DataArray(
    time_rel, coords={'time':dataset.time}, dims=('time')
)

dataset['rel_time_hold_for_water'] = rel_time_hold_for_water

#%%
df2plot = dataset[['rel_time_hold_for_water','analog_1_df_over_f']].to_dataframe()
df2plot.rel_time_hold_for_water = df2plot.rel_time_hold_for_water//10*10 # time windows: 100ms
sns.lineplot(x='rel_time_hold_for_water', y='analog_1_df_over_f', data=df2plot)


# %%
