'''
Example script to perform correlations between spikes and photometry
for the whole session
'''
#%%
import os
from pathlib import Path


import numpy as np
import pandas as pd
import xarray as xr

import seaborn as sns
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns

from snakehelper.SnakeIOHelper import getSnake

from workflow.scripts import settings
from trialexp.process.pyphotometry.utils import *


#%% Load inputs


(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [settings.debug_folder + r'/processed/xr_session_correlations.nc'],
  'session_correlations')

# %% Path definitions

verbose = True

root_path = Path(os.environ['SESSION_ROOT_DIR'])
# Where to store globally computed figures
clusters_figure_path = Path(os.environ['CLUSTERS_FIGURES_PATH'])
# where to store global processed data
clusters_data_path = Path(os.environ['PROCCESSED_CLUSTERS_PATH'])

# Get probe names from folder path
probe_names = [folder.stem for folder in list(Path(sinput.sorting_path).glob('*'))]


# %% bin all the clusters from all probes continuously, return nb_of_spikes per bin * (1000[ms]/bin_duratiion[ms])
# so if 1 spike in 20ms bin -> inst. FR = 1 * (1000/20) = 50Hz

# if bin duration == 1ms, we will have a BOOL arary (@1000Hz)


xr_spikes_session = xr.open_dataset(sinput.xr_spikes_session)

df_events_cond_path = Path(sinput.xr_session).parent / 'df_events_cond.pkl'
df_events_cond = pd.read_pickle(df_events_cond_path)

df_pycontrol_path = Path(sinput.xr_session).parent / 'df_pycontrol.pkl'
df_pycontrol = pd.read_pickle(df_pycontrol_path)



