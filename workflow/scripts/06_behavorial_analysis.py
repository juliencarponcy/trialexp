#%%
from snakehelper.SnakeIOHelper import getSnake
from trialexp.process.pyphotometry.plotting_utils import annotate_trial_number, plot_and_handler_error, plot_pyphoto_heatmap
from trialexp.process.pyphotometry.utils import *
from glob import glob
import xarray as xr
from trialexp.utils.rsync import *
import pandas as pd 
from scipy.interpolate import interp1d
import seaborn as sns 
import numpy as np
import os
from trialexp import config
from trialexp.process.pycontrol import event_filters

#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflow/pycontrol.smk',
  [config.debug_folder + '/processed/xr_behaviour.nc'],
  'behavorial_analysis')


# %%
df_event = pd.read_pickle(sinput.event_dataframe)
df_conditions = pd.read_pickle(sinput.condition_dataframe)
# %% Time between the bar off and first spout touch

# travel time
first_reach_travel_time = df_event.groupby('trial_nb').apply(event_filters.get_reach_travel_time)
xr_first_reach_travel_time = xr.DataArray(first_reach_travel_time)

#reach time
last_bar_off_time = df_event.groupby('trial_nb').apply(event_filters.get_last_bar_off_time)
xr_last_bar_off_time = xr.DataArray(last_bar_off_time)


#%% trial time of the first siginificant bar off

first_sig_bar_off_time = df_event.groupby('trial_nb').apply(event_filters.get_first_sig_bar_off_time)
xr_first_sig_bar_off_time = xr.DataArray(first_sig_bar_off_time)

# find the first bar off
first_bar_off_time = df_event.groupby('trial_nb').apply(event_filters.get_first_bar_off_time)
xr_first_bar_off_time = xr.DataArray(first_bar_off_time)

# %%

xr_behaviour = xr.Dataset({'first_reach_travel_time':xr_first_reach_travel_time,
                           'last_bar_off_time': xr_last_bar_off_time,
                           'first_sig_bar_off_trial_time': xr_first_sig_bar_off_time,
                           'first_bar_off_trial_time': xr_first_bar_off_time})

# %%
xr_behaviour = xr_behaviour.expand_dims({'session_id':[df_event.attrs['session_id']]})

# %%
xr_behaviour.to_netcdf(soutput.xr_behaviour, engine='h5netcdf')

