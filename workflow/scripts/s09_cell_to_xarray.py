'''
Script to create the session folder structure
'''
#%%
import os
from pathlib import Path
from itertools import cycle, islice

import numpy as np
import pandas as pd
import xarray as xr
from sklearn import cluster, mixture

from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns

from snakehelper.SnakeIOHelper import getSnake
from workflow.scripts import settings

#%% Load inputs


(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [settings.debug_folder + r'/processed/xr_cells.nc'],
  'cells_to_xarray')


#%% Half gaussian kernel convolution functions
# from https://stackoverflow.com/questions/71003634/applying-a-half-gaussian-filter-to-binned-time-series-data-in-python

import scipy.ndimage

def halfgaussian_kernel1d(sigma, radius):
    """
    Computes a 1-D Half-Gaussian convolution kernel.
    """
    sigma2 = sigma * sigma
    x = np.arange(0, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    return phi_x

def halfgaussian_filter1d(input, sigma, axis=-1, output=None,
                      mode="constant", cval=0.0, truncate=4.0):
    """
    Convolves a 1-D Half-Gaussian convolution kernel.
    """
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    weights = halfgaussian_kernel1d(sigma, lw)
    origin = -lw // 2
    return scipy.ndimage.convolve1d(input, weights, axis, output, mode, cval, origin)

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

idx_probe = 0
# %% # Loading synced spike timestamps

synced_ts = np.load(synced_timestamp_files[idx_probe]).squeeze()

spike_clusters = np.load(spike_clusters_files[idx_probe]).squeeze()

unique_clusters = np.unique(spike_clusters)

# Build a list where each item is a np.array containing spike times for a single cluster
ts_list = [synced_ts[np.where(spike_clusters==cluster_nb)] for cluster_nb in unique_clusters]
          
assert len(unique_clusters) == len(ts_list)

# %%

def get_max_bin_edge_all_probes(timestamp_files: list):
  """
  return up rounded timestamp of the latest spike over all probes
  """
  max_ts = []
  for ts_file in timestamp_files:
    synced_ts = np.load(ts_file)
    max_ts.append(int(np.ceil(np.nanmax(synced_ts))))

  # return up rounded timestamp of the latest spike over all probes
  return max(max_ts)

def get_cluster_UIDs_from_path(cluster_file: Path):
  # take Path or str
  cluster_file = Path(cluster_file)
  # extract session and probe name from folder structure
  session_id = cluster_file.parts[-6]
  probe_name = cluster_file.parts[-3]

  # unique cluster nb
  cluster_nbs = np.unique(np.load(cluster_file))
  
  # return list of unique cluster IDs strings format <session_ID>_<probe_name>_<cluster_nb>
  cluster_UIDs = [session_id + '_' + probe_name + '_' + str(cluster_nb) for cluster_nb in cluster_nbs]

  return cluster_UIDs

# define 1ms bins for discretizing in bool at 1000Hz 
def bin_millisecond_spikes_from_probes(synced_timestamp_files: list, spike_clusters_files: list, verbose: bool = True):
  
  # maximum spike bin end at upper rounding of latest spike : int(np.ceil(np.nanmax(synced_ts)))
  # bins = np.array([t for t in range(int(np.ceil(np.nanmax(synced_ts))))][:])
  bin_max = get_max_bin_edge_all_probes(synced_timestamp_files)

  # n clusters x m timebin
  # all_probes_binned_array = np.ndarray()
  # initialize UIDs list
  all_clusters_UIDs = list()
  

  for idx_probe, synced_file in enumerate(synced_timestamp_files):
    if idx_probe == 0:
      all_clusters_UIDs = get_cluster_UIDs_from_path(spike_clusters_files[idx_probe])
    else:
      cluster_UIDs = get_cluster_UIDs_from_path(spike_clusters_files[idx_probe])
      all_clusters_UIDs = all_clusters_UIDs + cluster_UIDs
      # extend list of cluster UIDs 
      
    synced_ts = np.load(synced_file).squeeze()
    spike_clusters = np.load(spike_clusters_files[idx_probe]).squeeze()
    

    unique_clusters = np.unique(spike_clusters)

    # Build a list where each item is a np.array containing spike times for a single cluster
    ts_list = [synced_ts[np.where(spike_clusters==cluster_nb)] for cluster_nb in unique_clusters]
    # change to a dict with clu_ID
    # iterate over the list of array to discretize (histogram / binning)
    cluster_idx_all = 0
    for cluster_idx, cluster_ts in enumerate(ts_list): # change to a dict?
      # establisn index is the ndarray for all the probes
      cluster_idx_all = cluster_idx_all + cluster_idx
      if verbose:
        print(f'probe nb {idx_probe+1}/{len(synced_timestamp_files)} cluster nb {cluster_idx+1}/{len(ts_list)}')
      
      # perform binning for the cluster
      binned_spike, bins = np.histogram(cluster_ts, bins = bin_max, range = (0, bin_max))
      binned_spike = binned_spike.astype(bool)

      # first creation of the array
      if cluster_idx == 0 and idx_probe == 0:
        all_probes_binned_array = np.ndarray(shape = (len(unique_clusters), binned_spike.shape[0]))
        all_probes_binned_array = all_probes_binned_array.astype(bool)
        all_probes_binned_array[cluster_idx,:] = binned_spike
      
      # concacatenate empty bool array for next probe
      elif cluster_idx == 0 and idx_probe != 0:
        probe_empty_binned_array = np.ndarray(shape = (len(unique_clusters), binned_spike.shape[0]), dtype=bool)
        all_probes_binned_array = np.concatenate([all_probes_binned_array, probe_empty_binned_array], axis=0)
        # all_probes_binned_array = all_probes_binned_array.astype(bool)

      else:
        # concat the next neurons bool binning
        all_probes_binned_array[cluster_idx,:] = binned_spike

    cluster_idx_all = cluster_idx_all + cluster_idx
    # convert bins to int
    bins = bins.astype(int)

  return all_probes_binned_array, bins, all_clusters_UIDs

# %%
# get_cluster_UIDs_from_path(spike_clusters_files[0])
all_probes_binned_array, spike_time_bins, all_clusters_UIDs = bin_millisecond_spikes_from_probes(synced_timestamp_files= synced_timestamp_files, spike_clusters_files = spike_clusters_files)# %% Opening session xarray
xr_session = xr.open_dataset(sinput.xr_session)
# %%
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
