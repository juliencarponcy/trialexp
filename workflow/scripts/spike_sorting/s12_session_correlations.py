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
# probe_names = [folder.stem for folder in list(Path(sinput.sorting_path).glob('*'))]


# %% bin all the clusters from all probes continuously, return nb_of_spikes per bin * (1000[ms]/bin_duratiion[ms])
# so if 1 spike in 20ms bin -> inst. FR = 1 * (1000/20) = 50Hz

# if bin duration == 1ms, we will have a BOOL arary (@1000Hz)


xr_spikes_session = xr.open_dataset(sinput.xr_spikes_session)

df_events_cond_path = Path(sinput.xr_spikes_session).parent / 'df_events_cond.pkl'
df_events_cond = pd.read_pickle(df_events_cond_path)

df_pycontrol_path = Path(sinput.xr_spikes_session).parent / 'df_pycontrol.pkl'
df_pycontrol = pd.read_pickle(df_pycontrol_path)




# %%

# Cross-corr with lags from: https://towardsdatascience.com/computing-cross-correlation-between-geophysical-time-series-488642be7bf0
def crosscorr(datax: pd.Series, datay: pd.Series, lag:int =0):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag))

def crosscorr_lag_range(datax: pd.Series, datay: pd.Series, lag_limits: list = [-25, 25]):
    lag_range = range(lag_limits[0], lag_limits[1]+1)
    cross_corr = np.ndarray(shape=(len(list(lag_range))))
    for lag_idx, lag in enumerate(lag_range):
        cross_corr[lag_idx] = crosscorr(datax,datay,lag)

    return cross_corr

# %%
lag_limits=[-25, 25]

df_over_f = xr_spikes_session.analog_1_df_over_f.to_series()
UIDs = xr_spikes_session.UID.to_series().values[20:30]
cross_corr = np.ndarray(shape=(len(UIDs), len(list(range(lag_limits[0], lag_limits[1]+1)))))

for uid_idx, uid in enumerate(UIDs):
    print(uid)
    z_scored_firing = xr_spikes_session.spikes_Zscore[
        :,xr_spikes_session.spikes_Zscore.UID == uid].to_series()
    cross_corr[uid_idx,:] = crosscorr_lag_range(df_over_f, z_scored_firing, lag_limits=lag_limits)
    
# %%
    figure, axes = plt.subplots(1, 2, sharex=True,
                                figsize=(15, 5))
    figure.suptitle('Cluster responses to trial sorted by max / min z-scored peak response time')

    sns.heatmap(spike_zscored_xr[np.flip(cluster_ID_sorted_max),:], 
                vmin=-3, vmax=3, ax=axes[0])

    sns.heatmap(spike_fr_xr[np.flip(cluster_ID_sorted_max),:],
                vmin=0, vmax=30, ax=axes[1])
    # need to specify 
    axes[0].set_title('Firing rate (z-score) around trial')
    axes[1].set_title('Firing rate (spikes/s) around trial')# %% Continuous All session long binning
    axes[1].set_xlabel('Time ')
    figure.savefig(session_figure_path / f'cluster_responses_to_{ev_name}.png')