'''
Script to create the session folder structure
'''
#%%
import os
from pathlib import Path
from timeit import timeit

import numpy as np
import pandas as pd
import xarray as xr

from sklearn.preprocessing import StandardScaler

import seaborn as sns
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns

from snakehelper.SnakeIOHelper import getSnake

from workflow.scripts import settings
from trialexp.process.pyphotometry.utils import *
from trialexp.process.pycontrol import event_filters
from trialexp.process.ephys.spikes_preprocessing import bin_spikes_from_all_probes,bin_spikes_from_all_probes_averaged, bin_spikes_from_all_probes_by_trials, halfgaussian_filter1d 
#%% Load inputs


(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [settings.debug_folder + r'/processed/xr_cells.nc'],
  'cells_to_xarray')

# %% Path definitions

verbose = True

root_path = Path(os.environ['SESSION_ROOT_DIR'])
# Where to store globally computed figures
clusters_figure_path = Path(os.environ['CLUSTERS_FIGURES_PATH'])
# where to store global processed data
clusters_data_path = Path(os.environ['PROCCESSED_CLUSTERS_PATH'])

# Get probe names from folder path
probe_names = [folder.stem for folder in list(Path(sinput.sorting_path).glob('*'))]

# Fetch file paths from all probes
synced_timestamp_files = list(Path(sinput.sorting_path).glob('*/sorter_output/rsync_corrected_spike_times.npy'))
spike_clusters_files = list(Path(sinput.sorting_path).glob('*/sorter_output/spike_clusters.npy'))
cell_metrics_files = list(Path(sinput.sorting_path).glob('*/sorter_output/cell_metrics_df_full.pkl'))


# %% bin all the clusters from all probes continuously, return nb_of_spikes per bin * (1000[ms]/bin_duratiion[ms])
# so if 1 spike in 20ms bin -> inst. FR = 1 * (1000/20) = 50Hz

# if bin duration == 1ms, we will have a BOOL arary (@1000Hz)

bin_duration = 20 # in ms
xr_session = xr.open_dataset(sinput.xr_session)
xr_photometry = xr.open_dataset(Path(sinput.xr_session).parent / 'xr_photometry.nc')

df_events_cond_path = Path(sinput.xr_session).parent / 'df_events_cond.pkl'
df_events_cond = pd.read_pickle(df_events_cond_path)

df_pycontrol_path = Path(sinput.xr_session).parent / 'df_pycontrol.pkl'
df_pycontrol = pd.read_pickle(df_pycontrol_path)

trigger = df_events_cond.attrs['triggers'][0]
trial_window = xr_session.attrs['trial_window']

# At the moment, this is the time we used to trigger extraction around trials
# This may be adapted to different triggering like first bar_off or first spout
trial_times = df_pycontrol[df_pycontrol.name == trigger].time.values.astype(int)

# %% Bin spikes from all probes and aggegate over trials around trial_times
all_probes_binned_array_averaged, spike_time_bins, all_clusters_UIDs = bin_spikes_from_all_probes_averaged(
    synced_timestamp_files = synced_timestamp_files, 
    spike_clusters_files = spike_clusters_files,
    trial_times = trial_times, 
    trial_window = trial_window,
    bin_duration = bin_duration)


# %% Applying Kernel to binned firing rate
# if we want the gaussian to have sigma = 0.5s
time_for_1SD = 0.2 # sec
sigma = time_for_1SD * (1000/bin_duration)

# May not be applied until bins are aggregated by trials
# Return the binned spike array convoluted by half-gaussian window (so FR do not increases before spike [but binning])
convoluted_binned_array = halfgaussian_filter1d(all_probes_binned_array_averaged, sigma = sigma, axis=-1, output=None,
                      mode="constant", cval=0.0, truncate=4.0) # truncate the window at 4SD

scaler = StandardScaler()
# z-scored firing rate
# .T).T to z-score along time dimension
z_conv_FR = scaler.fit_transform(convoluted_binned_array.T).T

# %% Combining spikes rates and scored

spike_fr_xr = xr.DataArray(
    convoluted_binned_array,
    name = 'spikes_FR',
    coords={'UID': all_clusters_UIDs, 'time':spike_time_bins[1:]},
    dims=('UID', 'time')
)

spike_zscored_xr = xr.DataArray(
    z_conv_FR,
    name = 'spikes_Zscore',
    coords={'UID': all_clusters_UIDs, 'time':spike_time_bins[1:]},
    dims=('UID', 'time')
)

# %% Combine cell metrics and spike bins

session_cell_metrics = pd.DataFrame(data={'UID':all_clusters_UIDs})
session_cell_metrics.set_index('UID', inplace=True)
uids = list()
for f_idx, cell_metrics_file in enumerate(cell_metrics_files):
    cell_metrics_df = pd.read_pickle(cell_metrics_file)
    session_cell_metrics = pd.concat([session_cell_metrics,cell_metrics_df])
    uids = uids + cell_metrics_df.index.tolist()


# add clusters_UIDs from spike_clusters.npy + those of cell metrics and merge
uids = list(set(uids+all_clusters_UIDs))

cluster_cell_IDs = pd.DataFrame(data={'UID':all_clusters_UIDs})
# Add sorted UIDs without cell metrics :  To investigate maybe some units not only present before / after 1st rsync?
session_cell_metrics = cell_metrics_df.merge(cluster_cell_IDs, on='UID', how='outer',)

session_cell_metrics.set_index('UID', inplace=True)
# A bit of tidy up is needed after merging so str columns can be str and not objects due to merge
types_dict = dict(zip(session_cell_metrics.columns,session_cell_metrics.dtypes))
for (col, dtype) in types_dict.items():
    if dtype == "dtype('O')":
        session_cell_metrics.loc[:,col] = session_cell_metrics.loc[:,col].astype('|S')

xr_cell_metrics = session_cell_metrics.to_xarray()

# Create the xr dataset
xr_spikes_averaged = xr.merge([spike_fr_xr, spike_zscored_xr, xr_cell_metrics])

# %% Save
# Do not work
# xr_spikes.to_netcdf(Path(sinput.xr_session).parent / 'xr_spikes.nc', engine='h5netcdf')

# %% Preview of cluster responses to trigger
# Sort clusters by peak time in the trial
idx_max_FR = np.argmax(convoluted_binned_array,1)
cluster_ID_sorted = np.argsort(idx_max_FR)

sns.heatmap(spike_zscored_xr[np.flip(cluster_ID_sorted),:], vmin=-3, vmax=3)

# %% Continuous All session long binning

# bin all the clusters from all probes continuously, return nb_of_spikes per bin * (1000[ms]/bin_duratiion[ms])
# so if 1 spike in 20ms bin -> inst. FR = 1 * (1000/20) = 50Hz

# if bin duration == 1ms, we will have a BOOL arary (@1000Hz)
bin_duration = 20 #bin_duration in ms

# bin_spikes_from_all_probes : All session. Dimensions cluster x time [long]
all_probes_binned_array, spike_time_bins, all_clusters_UIDs = bin_spikes_from_all_probes(
    synced_timestamp_files = synced_timestamp_files, 
    spike_clusters_files = spike_clusters_files, 
    bin_duration = bin_duration)

# %% Applying Kernel to binned firing rate (Continuous long session)
# if we want the gaussian to have sigma = 0.5s
time_for_1SD = 0.2 # sec
sigma = time_for_1SD * (1000/bin_duration)

# May not be applied until bins are aggregated by trials
# Return the binned spike array convoluted by half-gaussian window (so FR do not increases before spike [but binning])
convoluted_binned_array = halfgaussian_filter1d(all_probes_binned_array, sigma = sigma, axis=-1, output=None,
                      mode="constant", cval=0.0, truncate=4.0) # truncate the window at 4SD

# %%
scaler = StandardScaler()
# z-scored firing rate
# .T).T to z-score along time dimension
z_conv_FR = scaler.fit_transform(convoluted_binned_array.T).T


spike_fr_xr = xr.DataArray(
    all_probes_binned_array,
    name = 'spikes_FR',
    coords={'time':spike_time_bins[1:] - (bin_duration/2), 'UID': all_clusters_UIDs},
    dims=('UID', 'time')
)

spike_zscored_xr = xr.DataArray(
    z_conv_FR,
    name = 'spikes_Zscore',
    coords={'UID': all_clusters_UIDs, 'time':spike_time_bins[1:] - (bin_duration/2)},
    dims=('UID', 'time')
)

# Create the xr dataset
xr_spikes_session = xr.merge([spike_fr_xr, spike_zscored_xr, xr_cell_metrics, xr_session],
    fill_value={
        'subject_ID': 'ND', 
        'synapticEffect': 'ND', 
        'sessionName': 'ND', 
        'putativeCellType': 'ND',
        'labels': 'ND',
        'UID' : 'ND'
        })


# Create the xr dataset
# %% Binning by trials, 3D output (trial, time, cluster)
all_probes_binned_array_by_trials, spike_time_bins, all_clusters_UIDs = bin_spikes_from_all_probes_by_trials(
    synced_timestamp_files = synced_timestamp_files, 
    spike_clusters_files = spike_clusters_files,
    trial_times = trial_times,
    trial_window = trial_window,
    bin_duration = bin_duration,
    normalize = False)

# %% Combining spikes rates and scored

spike_fr_xr = xr.DataArray(
    all_probes_binned_array_by_trials,
    name = 'spikes_FR',
    coords={'trial': list(range(0,len(trial_times))), 'event_time':spike_time_bins[1:] - (bin_duration/2), 'UID': all_clusters_UIDs},
    dims=('trial', 'event_time', 'UID')
)

# Create the xr dataset
xr_spikes_trials = xr.merge([spike_fr_xr, xr_cell_metrics])
# %%
