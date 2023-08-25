'''
Example script to perform correlations between spikes and photometry
for the whole session

Work in progress, should not have been commited
'''
#%%
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import xarray as xr
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from snakehelper.SnakeIOHelper import getSnake

from trialexp.process.ephys.utils import crosscorr_lag_range, plot_correlated_neurons
from trialexp.process.group_analysis.plot_utils import style_plot
from trialexp.process.pyphotometry.utils import *
from workflow.scripts import settings

#%% Load inputs


(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [settings.debug_folder + r'/processed/df_cross_corr.pkl'],
  'session_correlations')

# %% Path definitions

verbose = True
root_path = Path(os.environ['SESSION_ROOT_DIR'])

# %% Loading files

xr_spike_fr = xr.load_dataset(sinput.xr_spike_fr)
session_root_path = Path(sinput.xr_spike_fr).parent

xr_session = xr.load_dataset(session_root_path/'xr_session.nc')

xr_spike_fr_interp = xr_spike_fr.interp(time=xr_session.time)
xr_spike_session = xr.merge([xr_session, xr_spike_fr_interp]) # make sure their time coord is the same

df_events_cond = pd.read_pickle(session_root_path/'df_events_cond.pkl')
df_pycontrol = pd.read_pickle(session_root_path/'df_pycontrol.pkl')

# %%
#Cross correlation between the firing rate and photometry signal
lags=np.arange(-50,50,5) #each bin is 10s, 25 bins = 250ms

df_over_f = xr_spike_session.analog_1_df_over_f.to_series()
UIDs = xr_spike_session.cluID.to_series()
cross_corr = np.ndarray(shape=(len(UIDs), len(lags)))

def calculate_crosscorr(uid):
    z_scored_firing = xr_spike_session.spikes_zFR_session[:, xr_spike_session.cluID == uid].to_series()
    return crosscorr_lag_range(df_over_f, z_scored_firing, lags=lags)

cross_corr = Parallel(n_jobs=20)(delayed(calculate_crosscorr)(uid) for uid in UIDs)
cross_corr = np.stack(cross_corr)
    
#%%
df_cross_corr = pd.DataFrame({
    'UID': UIDs.values,
    'cross_corr': cross_corr.tolist()
})
df_cross_corr.to_pickle(soutput.df_cross_corr)
#%% Plot the first 5 cell with maximum cross correlation with photometry

fig = plot_correlated_neurons(cross_corr, xr_spike_session)
# %%
#TODO: event in specific type of trial