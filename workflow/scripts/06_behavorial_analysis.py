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
from workflow.scripts import settings
from trialexp.process.pycontrol import event_filters

#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflow/pycontrol.smk',
  [settings.debug_folder + '/processed/xr_behaviour.nc'],
  'behavorial_analysis')


# %%
df_event = pd.read_pickle(sinput.event_dataframe)
df_conditions = pd.read_pickle(sinput.condition_dataframe)

# %% Time between the bar off and first spout touch

first_reach_travel_time = df_event.groupby('trial_nb').apply(event_filters.get_reach_travel_time)
xr_first_reach_time = xr.DataArray(first_reach_travel_time)
#%% trial time of the first siginificant bar off

first_sig_bar_off_time = df_event.groupby('trial_nb').apply(event_filters.get_first_sig_bar_off_time)
xr_first_sig_bar_off_time = xr.DataArray(first_sig_bar_off_time)
# %%

xr_behaviour = xr.Dataset({'first_reach_travel_time':xr_first_reach_time,
                           'first_sig_bar_off_trial_time': xr_first_sig_bar_off_time})

# Merge conditions
# ds_condition = xr.Dataset.from_dataframe(df_conditions)
# xr_behaviour = xr.merge([ds_condition, xr_behaviour])
# %%
xr_behaviour = xr_behaviour.expand_dims({'session_id':[df_event.attrs['session_id']]})

# %%
xr_behaviour.to_netcdf(soutput.xr_behaviour, engine='h5netcdf')
