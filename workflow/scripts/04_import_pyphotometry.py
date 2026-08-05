'''
This script import pyphotometry and perform the necessary processing
'''
#%%
from snakehelper.SnakeIOHelper import getSnake
from trialexp.process.pyphotometry.utils import *
from glob import glob
import xarray as xr
from trialexp.utils.pyphotometry_utilities import create_photo_sync
from trialexp.utils.rsync import *
import pandas as pd 
import numpy as np
from trialexp.process.pycontrol import event_filters
from trialexp.process.pycontrol.event_filters import extract_event_time
from trialexp.process.pycontrol.utils import calculate_lick_rate
from trialexp import config
from pathlib import Path
import pickle
from datetime import datetime
#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflow/pycontrol.smk',
   [config.debug_folder + '/processed/xr_photometry.nc'],
  'import_pyphotometry')

 
#%% Load pycontrol file
df_pycontrol = pd.read_pickle(sinput.pycontrol_dataframe)
df_event = pd.read_pickle(sinput.event_dataframe)
df_condition = pd.read_pickle(sinput.condition_dataframe)
trial_window = df_event.attrs['trial_window']

df_dataformat = pd.read_csv('params/data_format.csv')


#%% Load pyphotometry file
try:
    pyphotometry_file = list(Path(sinput.photometry_folder).glob('*.ppd'))[0]
    has_photometry = True
    data_photometry = import_ppd_auto(pyphotometry_file)
    data_photometry = preprocess_photometry(data_photometry, df_pycontrol)
    
    # Convert to xarray
    skip_var = ['analog_1_est_motion','time',
                'analog_1_corrected', 'analog_1_baseline_fluo', 
                'analog_2_baseline_fluo',
                'isos_bleach_baseline', 'analog_1_bleach_baseline',
                'analog_1_detrend', 'isos_detrended']
    
    dataset = photometry2xarray(data_photometry, skip_var = skip_var)
except IndexError:
    has_photometry = False

#%% Align data
if has_photometry:    
    photo_rsync = dataset.attrs['pulse_times_2']

    #align pyphotometry time to pycontrol
    pycontrol_aligner = create_photo_sync(df_pycontrol, data_photometry)

    dataset = align_photometry_to_pycontrol(dataset, df_event, pycontrol_aligner)


    # Also add lick rate as another variables
    if any(df_event.content == 'lick'):
        xa_lick_rate = calculate_lick_rate(df_event, dataset)
        dataset['lick_rate'] = xa_lick_rate

#%% Add in the relative time to different events
event_period = (trial_window[1] - trial_window[0])/1000


if has_photometry:
    sampling_freq = dataset.attrs['sampling_rate']
    event_time_coord= np.linspace(trial_window[0], trial_window[1], int(event_period*sampling_freq)) #TODO
       
    # var2add = ['zscored_df_over_f']
    # if 'zscored_df_over_f_analog_2' in  dataset:
    #     var2add.append('zscored_df_over_f_analog_2')
        
    # if 'zscored_df_over_f_analog_3' in dataset:
    #     var2add.append('zscored_df_over_f_analog_3')

    var2add = [v for v in dataset.data_vars if v.startswith('zscored')]
    
    if 'lick_rate' in dataset:
        var2add.append('lick_rate')

    # if 'zscored_df_over_f_bleedthrough_ch2' in dataset:
    #     var2add.append('zscored_df_over_f_bleedthrough_ch2')


    for var in var2add:
         # Add trigger
        trigger = df_event.attrs['triggers'][0]
        add_event_data(df_event, event_filters.get_first_event_from_name,
                    trial_window, dataset, event_time_coord, 
                    var, trigger, dataset.attrs['sampling_rate'],
                    filter_func_kwargs={'evt_name':trigger})
        
        # Add first bar off
        add_event_data(df_event, event_filters.get_first_bar_off, trial_window, dataset,event_time_coord, 
                    var, 'first_bar_off', dataset.attrs['sampling_rate'])

        # Add first spout
        add_event_data(df_event, event_filters.get_first_spout, trial_window, dataset, event_time_coord, 
                    var, 'first_spout', dataset.attrs['sampling_rate'])

        # Add last bar_off before first spout
        add_event_data(df_event, event_filters.get_last_bar_off_before_first_spout, trial_window,
                    dataset,event_time_coord, 
                    var, 'last_bar_off', dataset.attrs['sampling_rate'])
        
        #get event to process
        if 'opto' in df_event.attrs['task_name']:
            # hard code for now, TODO: change the task param to yaml
            for evt in df_event.attrs['events_to_process']:
                add_event_data(df_event, event_filters.get_first_event_from_name,
                        trial_window, dataset, event_time_coord, 
                        var, f'first_{evt}', dataset.attrs['sampling_rate'],
                        filter_func_kwargs={'evt_name':evt})
                
    
        # Also add data from any additional event triggers
        if 'extra_event_triggers' in df_event.attrs:
            for evt_triggers in df_event.attrs['extra_event_triggers']:
                #Note: df_event only contains event extract around the trigger
                add_event_data(df_event, event_filters.get_events_from_name,
                    trial_window, dataset, event_time_coord, 
                    var, evt_triggers, dataset.attrs['sampling_rate'],
                    groupby_col=None,
                    filter_func_kwargs={'evt_name':evt_triggers})
            


    dataset = dataset.sel(time = dataset.trial>=0) #remove data outside of task

    # add in all metadata
    dataset.attrs.update(df_pycontrol.attrs)
    dataset.attrs.update(df_event.attrs)

    dataset.to_netcdf(soutput.xr_photometry, engine='h5netcdf')

else:
    # create the dataset for pycontrol data only
    dataset = xr.Dataset()
    t = df_pycontrol.time.values
    time_coords = np.arange(t[0], t[-1])
    sampling_rate = 1000 # by default, pycontrol is samplied at 1000Hz
    event_time_coord= np.linspace(trial_window[0], trial_window[1], int(event_period*sampling_rate)) 

    dataset = dataset.assign_coords(time=('time',time_coords))
    dataset = dataset.assign_coords(event_time=('event_time', event_time_coord))
    
    # add in all metadata
    dataset.attrs.update(df_pycontrol.attrs)
    dataset.attrs.update(df_event.attrs)
    
    dataset.attrs['sampling_rate'] = sampling_rate

    # save an empty photometry dataset (no photometry data available)
    dataset.to_netcdf(soutput.xr_photometry, engine='h5netcdf')
        
#%% Bin the data such that we only have 1 data point per time bin
down_sample_ratio = int(dataset.attrs['sampling_rate']/100)
if down_sample_ratio>0:
    # bin according to 10ms time bin (aka 100Hz), original sampling frequency is at 1000Hz for continuous data
    dataset_binned = dataset.coarsen(time=down_sample_ratio, event_time=down_sample_ratio, boundary='trim').mean()
else:
    # do not resample when sampling rate is lower than 100
    dataset_binned = dataset
dataset_binned['event_time'] = dataset_binned.event_time.astype(int) #cast to int to avoid floating point error later
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
if has_photometry:
    with open(soutput.pycontrol_aligner, 'wb') as f:
        pickle.dump(pycontrol_aligner, f)
else:
    Path(soutput.pycontrol_aligner).touch()

# %%
