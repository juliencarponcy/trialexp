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
  [settings.debug_folder + '/processed/df_behavoiral.pkl'],
  'behavorial_analysis')
# %%
df_event = pd.read_pickle(sinput.event_dataframe)
# %%
add_event_data(df_event, event_filters.get_first_bar_off, trial_window, dataset,event_time_coord, 
               'zscored_df_over_f', 'first_bar_off', dataset.attrs['sampling_rate'])
# %%
