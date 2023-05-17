from pathlib import Path
from pathlib import Path

import numpy as np


# %% Extract and bin spikes by cluster_ID 

def get_max_bin_edge_all_probes(timestamp_files: list, bin_duration: int):
    """
    return up rounded timestamp of the latest spike over all probes
    both timestamps and bin duration must be in ms
    """

    max_ts = np.ndarray(shape=(len(timestamp_files)))
    for f_idx, ts_file in enumerate(timestamp_files):
        synced_ts = np.load(ts_file)
        max_ts[f_idx] = np.nanmax(synced_ts)

    bins_nb = np.int(np.ceil((max(max_ts) / bin_duration)))
    bin_max = bins_nb * bin_duration

    # return up rounded timestamp of the latest spike over all probes
    return bins_nb, bin_max

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

# define 1ms bins for discretizing at 1000Hz 
def bin_spikes_from_all_probes_by_trials(
        synced_timestamp_files: list, 
        spike_clusters_files: list, 
        trial_times: list, 
        trial_window: list, 
        bin_duration: int = 10, 
        verbose: bool = True):
    '''
    Bin all clusters from all probes for the session
    
    return bin spikes array with dimensions (cluster, time)
    # Important:
    if bin_duration is 1 ms, the resulting array will be a boolean ndarray
    '''
    # maximum spike bin end at upper rounding of latest spike : int(np.ceil(np.nanmax(synced_ts)))
    # bins = np.array([t for t in range(int(np.ceil(np.nanmax(synced_ts))))][:])
    bins_nb, bin_max = get_max_bin_edge_all_probes(synced_timestamp_files, bin_duration)

    # n clusters x m timebin
    # all_probes_binned_array = np.ndarray()
    # initialize UIDs list
    all_clusters_UIDs = list()
    cluster_idx_all = 0

    for idx_probe, synced_file in enumerate(synced_timestamp_files):
        if idx_probe == 0:
            all_clusters_UIDs = get_cluster_UIDs_from_path(spike_clusters_files[idx_probe])
        else:
            cluster_UIDs = get_cluster_UIDs_from_path(spike_clusters_files[idx_probe])
            all_clusters_UIDs = all_clusters_UIDs + cluster_UIDs
            # extend list of cluster UIDs 

    for idx_probe, synced_file in enumerate(synced_timestamp_files):
        
        synced_ts = np.load(synced_file).squeeze()
        spike_clusters = np.load(spike_clusters_files[idx_probe]).squeeze()

        unique_clusters = np.unique(spike_clusters)

        # Build a list where each item is a np.array containing spike times for a single cluster
        ts_list = [synced_ts[np.where(spike_clusters==cluster_nb)] for cluster_nb in unique_clusters]
        # change to a dict with clu_ID
        # iterate over the list of array to discretize (histogram / binning)
        
        for cluster_idx, cluster_ts in enumerate(ts_list): # change to a dict?
            
            cluster_trial_times = np.empty((0))
            for t in trial_times:
                spikes_filter = np.logical_and(cluster_ts >= t+trial_window[0], cluster_ts <= t+trial_window[1])
                cluster_trial_times = np.hstack([cluster_trial_times, cluster_ts[np.where(spikes_filter)] - t])

            if verbose:
                print(f'probe nb {idx_probe+1}/{len(synced_timestamp_files)} cluster nb {cluster_idx+1}/{len(ts_list)}')
            
            # perform binning for the cluster
            binned_spike, bins = np.histogram(cluster_trial_times, bins = list(range(trial_window[0],trial_window[1]+bin_duration, bin_duration)))
            # Convert in Hz (spikes/s) - > normalize by bin_duration and number of trials
            binned_spike = binned_spike * (1000 / bin_duration) / len(trial_times)

            # first creation of the array
            if cluster_idx == 0 and idx_probe == 0:
                all_probes_binned_array = np.ndarray(shape=(len(all_clusters_UIDs), binned_spike.shape[0]))

            all_probes_binned_array[cluster_idx_all,:] = binned_spike

            cluster_idx_all = cluster_idx_all +1
    
    # convert time bins to int (ms)
    bins = bins.astype(int)

    return all_probes_binned_array, bins, all_clusters_UIDs


# define 1ms bins for discretizing at 1000Hz 
def bin_spikes_from_all_probes(synced_timestamp_files: list, spike_clusters_files: list, bin_duration: int = 10, verbose: bool = True):
    '''
    Bin all clusters from all probes for the session
    
    return bin spikes array with dimensions (cluster, time)
    # Important:
    if bin_duration is 1 ms, the resulting array will be a boolean ndarray
    '''
    # maximum spike bin end at upper rounding of latest spike : int(np.ceil(np.nanmax(synced_ts)))
    # bins = np.array([t for t in range(int(np.ceil(np.nanmax(synced_ts))))][:])
    bins_nb, bin_max = get_max_bin_edge_all_probes(synced_timestamp_files, bin_duration)

    # n clusters x m timebin
    # all_probes_binned_array = np.ndarray()
    # initialize UIDs list
    all_clusters_UIDs = list()
    cluster_idx_all = 0

    for idx_probe, synced_file in enumerate(synced_timestamp_files):
        if idx_probe == 0:
            all_clusters_UIDs = get_cluster_UIDs_from_path(spike_clusters_files[idx_probe])
        else:
            cluster_UIDs = get_cluster_UIDs_from_path(spike_clusters_files[idx_probe])
            all_clusters_UIDs = all_clusters_UIDs + cluster_UIDs
            # extend list of cluster UIDs 

    for idx_probe, synced_file in enumerate(synced_timestamp_files):
        
        synced_ts = np.load(synced_file).squeeze()
        spike_clusters = np.load(spike_clusters_files[idx_probe]).squeeze()

        unique_clusters = np.unique(spike_clusters)

        # Build a list where each item is a np.array containing spike times for a single cluster
        ts_list = [synced_ts[np.where(spike_clusters==cluster_nb)] for cluster_nb in unique_clusters]
        # change to a dict with clu_ID
        # iterate over the list of array to discretize (histogram / binning)
        
        for cluster_idx, cluster_ts in enumerate(ts_list): # change to a dict?

            if verbose:
                print(f'probe nb {idx_probe+1}/{len(synced_timestamp_files)} cluster nb {cluster_idx+1}/{len(ts_list)}')
            
            # perform binning for the cluster
            binned_spike, bins = np.histogram(cluster_ts, bins = bins_nb, range = (0, bin_max))
            # Convert in Hz (spikes/s)
            binned_spike = binned_spike * (1000 / bin_duration)
            
            # if binned at 1ms turn into bool, otherwise int
            if bin_duration == 1:
                binned_spike = binned_spike.astype(bool)
            else:
                binned_spike = binned_spike.astype(int)

            # first creation of the array
            if cluster_idx == 0 and idx_probe == 0:
                all_probes_binned_array = np.ndarray(shape=(len(all_clusters_UIDs), binned_spike.shape[0]))
                if bin_duration == 1:
                    all_probes_binned_array = all_probes_binned_array.astype(bool)                
                else:    
                    all_probes_binned_array = all_probes_binned_array.astype(int)
                all_probes_binned_array[cluster_idx_all,:] = binned_spike
            else:
                # concat the next neurons bool binning
                all_probes_binned_array[cluster_idx_all,:] = binned_spike

            cluster_idx_all = cluster_idx_all +1
    
    # convert time bins to int (ms)
    bins = bins.astype(int)

    return all_probes_binned_array, bins, all_clusters_UIDs



#%% Half gaussian kernel convolution functions to compute spike density estimations, and standardization

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