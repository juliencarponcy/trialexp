'''
Script to create the session folder structure
'''
#%%
import os
from pathlib import Path

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

from trialexp.process.ephys.spikes_preprocessing import bin_spikes_from_all_probes, halfgaussian_filter1d
#%% Load inputs


(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [settings.debug_folder + r'/processed/xr_cells.nc'],
  'cells_to_xarray')

# %% Path definitions

sorter_name = 'kilosort3'
verbose = True

root_path = Path(os.environ['SESSION_ROOT_DIR'])
# Where to store globally computed figures
clusters_figure_path = Path(os.environ['CLUSTERS_FIGURES_PATH'])
# where to store global processed data
clusters_data_path = Path(os.environ['PROCCESSED_CLUSTERS_PATH'])

synced_timestamp_files = list(Path(sinput.sorting_path).glob('*/sorter_output/rsync_corrected_spike_times.npy'))
spike_clusters_files = list(Path(sinput.sorting_path).glob('*/sorter_output/spike_clusters.npy'))

# Get probe names from folder path
probe_names = [folder.stem for folder in list(Path(sinput.sorting_path).glob('*'))]

# %% bin all the clusters from all probes continuously, return nb_of_spikes per bin * (1000[ms]/bin_duratiion[ms])
# so if 1 spike in 20ms bin -> inst. FR = 1 * (1000/20) = 50Hz
bin_duration = 50 #bin_duration in ms

all_probes_binned_array, spike_time_bins, all_clusters_UIDs = bin_spikes_from_all_probes(
                                                                synced_timestamp_files = synced_timestamp_files, 
                                                                spike_clusters_files = spike_clusters_files, 
                                                                bin_duration = bin_duration)
# %% Applying Kernel to binned firing rate
# if we want the gaussian to have sigma = 0.5s
time_for_1SD = 0.5 # sec
sigma = time_for_1SD * (1000/bin_duration)

# Return the binned spike array convoluted by half-gaussian window (so FR do not increases before spike!)
convoluted_binned_array = halfgaussian_filter1d(all_probes_binned_array.astype(float), sigma = sigma, axis=-1, output=None,
                      mode="constant", cval=0.0, truncate=4.0) # truncate the window at 4SD

scaler = StandardScaler()
# z-scored firing rate
z_conv_FR = scaler.fit_transform(convoluted_binned_array.T).T

# %% Combining spikes rates and scored

spike_fr_xr = xr.DataArray(
    convoluted_binned_array,
    name = 'spikes_FR',
    coords={'cluster_UID':all_clusters_UIDs, 'time':spike_time_bins[1:]},
    dims=('cluster_UID', 'time')
)

spike_zscored_xr = xr.DataArray(
    z_conv_FR,
    name = 'spikes_Zscore',
    coords={'cluster_UID':all_clusters_UIDs, 'time':spike_time_bins[1:]},
    dims=('cluster_UID', 'time')
)

xr_dict = {'spikes_FR': spike_fr_xr, 'spikes_Zscore': spike_zscored_xr}
xr_spikes = xr.Dataset(xr_dict)

# %% Preview
sns.heatmap(spike_fr_xr[:,-10000:], vmin=0, vmax=10)
# %%
xr_session = xr.open_dataset(sinput.xr_session)

# %% Opening session xarray
# Loading dataframe containing whole dataset dimensionality reductions for cell metrics
dim_reduc_aggregate = pd.read_pickle(clusters_data_path / 'aggregate_cell_metrics_dim_reduc.pkl')
# Loading dataframe containing whole dataset for cell metrics
aggregate_cell_metrics_df = pd.read_pickle(clusters_data_path/ 'aggregate_cell_metrics_df_full.pkl')
# Loading dataframe containing clustering data for cell metrics
aggregate_cell_metrics_df_clustering = pd.read_pickle(clusters_data_path/ 'aggregate_cell_metrics_df_clustering.pkl')

# Loading numpy array of raw waveforms across all channels
aggregate_raw_waveforms = np.load(clusters_data_path / 'all_raw_waveforms.npy')

# Check if aggregate_cell_metrics_df and dim_reduc_aggregate have the same length 
# TODO potentially identify culprit file(s)/sessions by disimilarity of the indices
assert len(dim_reduc_aggregate) == len(aggregate_cell_metrics_df) == aggregate_raw_waveforms.shape[2], \
  f" aggregate_cell_metrics_df, dim_reduc_aggregate and raw_waveforms don't have the same length: {len(dim_reduc_aggregate)}, {len(aggregate_cell_metrics_df)}, {aggregate_raw_waveforms.shape[2]}"


# %%
