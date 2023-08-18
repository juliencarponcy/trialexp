from pathlib import Path

import numpy as np
import pandas as pd

from neo.core import SpikeTrain # %% Extract and bin spikes by cluster_ID 

from trialexp.process.ephys.utils import dataframe_cleanup
import xarray as xr

## %
def get_max_timestamps_from_probes(timestamp_files: list):
    max_ts = np.ndarray(shape=(len(timestamp_files)))
    for f_idx, ts_file in enumerate(timestamp_files):
        synced_ts = np.load(ts_file)
        max_ts[f_idx] = np.nanmax(synced_ts)
    return max(max_ts)

def get_spike_trains(
        synced_timestamp_files: list, 
        spike_clusters_files: list):
    
    # Note: UID is the id used internally in cellexplorer
    # clusID the is the label from kilosort
    # the cluster label from cluster_KSLabel.tsv and spike_clusters.npy are the same
    # by default, cell explorer will only load good unit from kilosort as defined in the cluster_KSLabel.tsv
    # defination of 'good' is  ContamPct < 10, ContamPct is based on a refactory period of 2msec
    # so the all_clusters_UIDs here is the super-set of the cluID from Cell Explorer
    
    max_ts = get_max_timestamps_from_probes(synced_timestamp_files)

    for idx_probe, synced_file in enumerate(synced_timestamp_files):
        if idx_probe == 0:
            all_clusters_UIDs = get_cluster_UIDs_from_path(spike_clusters_files[idx_probe])
        else:
            cluster_UIDs = get_cluster_UIDs_from_path(spike_clusters_files[idx_probe])
            all_clusters_UIDs = all_clusters_UIDs + cluster_UIDs

    spike_trains = list()

    for idx_probe, synced_file in enumerate(synced_timestamp_files):
        
        synced_ts = np.load(synced_file).squeeze()
        spike_clusters = np.load(spike_clusters_files[idx_probe]).squeeze()

        unique_clusters = np.unique(spike_clusters)

        # Build a list where each item is a np.array containing spike times for a single cluster
        ts_list = [synced_ts[np.where(spike_clusters==cluster_nb)] for cluster_nb in unique_clusters]

        
        for cluster_idx, cluster_ts in enumerate(ts_list): # change to a dict?
            spike_trains.append(SpikeTrain(times=cluster_ts, 
                                                   units='ms', 
                                                   t_stop=max_ts, 
                                                   name=all_clusters_UIDs[cluster_idx], 
                                                   file_origin=synced_file))
            
    return spike_trains, all_clusters_UIDs


def extract_trial_data(xr_inst_rates, evt_timestamps, trial_window, bin_duration):
    # Extract instantaneous rate triggered by some event timestamps
    num_trials = len(evt_timestamps)
    num_clusters = len(xr_inst_rates.cluID)
    time_vector = xr_inst_rates.time

    num_time_points = int(trial_window[0] + trial_window[1]) // bin_duration +1
    trial_time_vec = np.linspace(-trial_window[0], trial_window[1], num_time_points)
    trial_data = np.empty((num_trials, num_time_points, num_clusters))

    for i, timestamp in enumerate(evt_timestamps):
        if np.isnan(timestamp):  # Skip NaN timestamps
            continue
        
        start_time = timestamp - trial_window[0]

        # Find the indices of the time points within the trial window
        start_idx = np.searchsorted(time_vector, start_time, side='left')
        # Extract the data for the trial and assign it to the trial_data array
        try:
            trial_data[i, :, :] = xr_inst_rates.data[start_idx:start_idx + num_time_points, :]
        except ValueError:
            # cannot find the data from the specifed timestamp, fill with NaN
            trial_data[i, :, :] = np.empty((num_time_points, num_clusters))*np.nan

    return trial_data, trial_time_vec


def build_evt_fr_xarray(fr_xr, timestamps, trial_index, name, trial_window, bin_duration):
    # Construct an xr.DataArray with firing rate triggered by the specified timestamps
    
    trial_rates, trial_time_vec = extract_trial_data(fr_xr, timestamps, trial_window, bin_duration)
    
    da = xr.DataArray(
        trial_rates,
        # name = f'spikes_FR.{ev_name}',
        name = name,
        coords={'trial_nb': trial_index, 'event_time': trial_time_vec, 'cluID': fr_xr.cluID},
        dims=('trial_nb', 'event_time', 'cluID')
        )
    
    return da

def get_cluster_UIDs_from_path(cluster_file: Path):
    # take Path or str
    cluster_file = Path(cluster_file)
    # extract session and probe name from folder structure
    session_id = cluster_file.parts[-5]
    probe_name = cluster_file.parts[-2]

    # unique cluster nb
    cluster_nbs = np.unique(np.load(cluster_file))

    # return list of unique cluster IDs strings format <session_ID>_<probe_name>_<cluster_nb>
    cluster_UIDs = [session_id + '_' + probe_name + '_' + str(cluster_nb) for cluster_nb in cluster_nbs]

    return cluster_UIDs

def merge_cell_metrics_and_spikes(
        cell_metrics_files: list,
        cluster_UIDs: list) -> pd.DataFrame:
    '''
    Merge spikes from spike_clusters.npy
    and cell_metrics_df (DataFrame with CellExplorer metrics)

    cell_metrics_files is a list of cell_metrics_df_full.pkl files path
    return a DataFrame with grouped CellExplorer cell metrics and spike
    clusters extracted from spike_clusters.npy files from both probes.
    '''
    session_cell_metrics = pd.DataFrame(data={'UID': cluster_UIDs})
    session_cell_metrics.set_index('UID', inplace=True)
    uids = list()
    for f_idx, cell_metrics_file in enumerate(cell_metrics_files):
        cell_metrics_df = pd.read_pickle(cell_metrics_file)
        session_cell_metrics = pd.concat([session_cell_metrics,cell_metrics_df])
        uids = uids + cell_metrics_df.index.tolist()

    # add clusters_UIDs from spike_clusters.npy + those of cell metrics and merge
    uids = list(set(uids + cluster_UIDs))

    cluster_cell_IDs = pd.DataFrame(data={'UID': cluster_UIDs})
    # Add sorted UIDs without cell metrics :  To investigate maybe some units not only present before / after 1st rsync?
    session_cell_metrics = cell_metrics_df.merge(cluster_cell_IDs, on='UID', how='outer',)

    session_cell_metrics.set_index('UID', inplace=True)

    # A bit of tidy up is needed after merging so str columns can be str and not objects due to merge
    session_cell_metrics = dataframe_cleanup(session_cell_metrics)

    return session_cell_metrics
