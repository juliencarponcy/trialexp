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
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from snakehelper.SnakeIOHelper import getSnake

from trialexp.process.pyphotometry.utils import *
from workflow.scripts import settings

#%% Load inputs


(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [settings.debug_folder + r'/processed/xr_session_correlations.nc'],
  'session_correlations')

# %% Path definitions

verbose = True

root_path = Path(os.environ['SESSION_ROOT_DIR'])
# Where to store globally computed figures
# clusters_figure_path = Path(os.environ['CLUSTERS_FIGURES_PATH'])
# where to store global processed data
# clusters_data_path = Path(os.environ['PROCCESSED_CLUSTERS_PATH'])

# Get probe names from folder path
# probe_names = [folder.stem for folder in list(Path(sinput.sorting_path).glob('*'))]


# %% bin all the clusters from all probes continuously, return nb_of_spikes per bin * (1000[ms]/bin_duratiion[ms])
# so if 1 spike in 20ms bin -> inst. FR = 1 * (1000/20) = 50Hz

# if bin duration == 1ms, we will have a BOOL arary (@1000Hz)


xr_spike_fr = xr.open_dataset(sinput.xr_spike_fr)
session_root_path = Path(sinput.xr_spike_fr).parent

xr_session = xr.open_dataset(session_root_path/'xr_session.nc')


xr_spike_session = xr.merge([xr_session, xr_spike_fr]) # make sure their time coord is the same

df_events_cond = pd.read_pickle(session_root_path/'df_events_cond.pkl')
df_pycontrol = pd.read_pickle(session_root_path/'df_pycontrol.pkl')


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
#Cross correlation between the firing rate and photometry signal
lag_limits=[-25, 25]

df_over_f = xr_spike_session.analog_1_df_over_f.to_series()
UIDs = xr_spike_session.cluID.to_series().values[20:30]
cross_corr = np.ndarray(shape=(len(UIDs), len(list(range(lag_limits[0], lag_limits[1]+1)))))

for uid_idx, uid in enumerate(UIDs):
    print(uid)
    z_scored_firing = xr_spike_session.spikes_zFR_session[:, xr_spike_session.cluID == uid].to_series()
    cross_corr[uid_idx,:] = crosscorr_lag_range(df_over_f, z_scored_firing, lag_limits=lag_limits)
    
# # %%
#     figure, axes = plt.subplots(1, 2, sharex=True,
#                                 figsize=(15, 5))
#     figure.suptitle('Cluster responses to trial sorted by max / min z-scored peak response time')

#     sns.heatmap(spike_zscored_xr[np.flip(cluster_ID_sorted_max),:], 
#                 vmin=-3, vmax=3, ax=axes[0])

#     sns.heatmap(spike_fr_xr[np.flip(cluster_ID_sorted_max),:],
#                 vmin=0, vmax=30, ax=axes[1])
#     # need to specify 
#     axes[0].set_title('Firing rate (z-score) around trial')
#     axes[1].set_title('Firing rate (spikes/s) around trial')# %% Continuous All session long binning
#     axes[1].set_xlabel('Time ')
#     figure.savefig(session_figure_path / f'cluster_responses_to_{ev_name}.png')