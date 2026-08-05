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
from trialexp import config
from pathlib import Path
import pickle 
import seaborn as sns

#%% Load inputs
(sinput, soutput) = getSnake(locals(), 'workflow/pycontrol.smk',
    [config.debug_folder + '/processed/log/task_specific_analysis.done'],
    'task_specifc_analysis')

# %% load data
df_event = pd.read_pickle(sinput.event_dataframe)
xr_session =  xr.load_dataset(sinput.xr_session) # load for modification
xr_photometry = xr.load_dataset(sinput.xr_photometry, engine='h5netcdf')
trial_window = xr_session.attrs['trial_window']

#%%

        
var2add = ['zscored_df_over_f']
if 'zscored_df_over_f_analog_2' in  xr_photometry:
    var2add.append('zscored_df_over_f_analog_2')
    
if 'zscored_df_over_f_analog_3' in xr_photometry:
    var2add.append('zscored_df_over_f_analog_3')
    
for var in var2add:

    add_event_data(df_event, extract_clean_trigger_event, trial_window, xr_photometry, xr_photometry.event_time, 
                var, 'clean_busy_win', xr_photometry.attrs['sampling_rate'], 
                filter_func_kwargs = dict(clean_window = [-500,500], target_event_name='busy_win'))

    add_event_data(df_event, extract_clean_trigger_event, trial_window, xr_photometry, xr_photometry.event_time, 
                var, 'clean_spout', xr_photometry.attrs['sampling_rate'], 
                filter_func_kwargs = dict(clean_window = [-500,500], target_event_name='spout', ignore_events=['spout_off','bar_off']))

    # down_sample_ratio = int(xr_photometry.attrs['sampling_rate']/100)
    # if down_sample_ratio>0:
    #     dataset_binned = xr_photometry.coarsen(time=down_sample_ratio, event_time=down_sample_ratio, boundary='trim').mean()
    # else:
    #     dataset_binned = xr_photometry
    
    xr_photometry = xr_photometry.interp(event_time = xr_session.event_time)

    # # add the data back to xr_session
    xr_session[f'clean_spout_{var}'] = xr_photometry[f'clean_spout_{var}']
    xr_session[f'clean_busy_win_{var}']  = xr_photometry[f'clean_busy_win_{var}'] 
    
#%%
xr_session.to_netcdf(Path(sinput.xr_session).parent/'xr_session_task_specific.nc', engine='h5netcdf')


# %%
