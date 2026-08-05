from pathlib import Path

import numpy as np
import pandas as pd

from neo.core import SpikeTrain # %% Extract and bin spikes by cluster_ID 
import xarray as xr
from trialexp.process.pycontrol import event_filters
from pandas.api.types import infer_dtype
from functools import partial
from scipy.optimize import curve_fit
import matplotlib.pylab as plt 
        
def dataframe_cleanup(dataframe: pd.DataFrame):
    '''
    Turn object columns into str columns and fill empty gaps with ''
    '''
    types_dict = dict(zip(dataframe.columns,dataframe.dtypes))
    for (col, dtype) in types_dict.items():
        if dtype == np.dtype(object):
            dtype_inferred = infer_dtype(dataframe[col])
            dataframe[col] = dataframe[col].fillna('', downcast={np.dtype(object):str}).astype(str)
            dataframe[col] = dataframe[col].astype(dtype_inferred)
            # session_cell_metrics[col] = session_cell_metrics[col].astype(str)
    
    return dataframe

## %
def get_max_timestamps_from_probes(timestamp_files: list):
    max_ts = np.ndarray(shape=(len(timestamp_files)))
    for f_idx, ts_file in enumerate(timestamp_files):
        synced_ts = np.load(ts_file)
        max_ts[f_idx] = np.nanmax(synced_ts)
    return max(max_ts)

def load_kilosort(ks_result_folder, skip_PC_feature=False):
    #load the results from kilosort
    ks_results = {}
    for f in Path(ks_result_folder).glob('*.npy'):
        if skip_PC_feature and 'pc_features' in f.name:
            continue

        ks_results[f.stem]=np.load(f, allow_pickle=True)
        
    for f in Path(ks_result_folder).glob('*.tsv'):
        ks_results[f.stem]=pd.read_csv(f, sep='\t')
        
    return ks_results


def add_ks_metadata(ks_results, df_metrics, good_only=False):
    """
    Adds metadata from ks_results to df_metrics DataFrame.
    
    Args:
        ks_results (dict): Dictionary containing results from Kilosort spike sorting.
        df_metrics (pandas.DataFrame): DataFrame containing spike metrics.
    
    Returns:
        None
    """
    ks_labels = ks_results['cluster_KSLabel']['KSLabel'].values
    good_idx = (ks_labels=='good')
    
    sel_idx = np.arange(len(ks_labels))
    if good_only:
        sel_idx = good_idx
            
    templates = ks_results['templates']
    max_chans = np.array([np.argmax(np.max(np.abs(te),axis=0)) for te in templates])
    chan_pos = np.stack([ks_results['channel_positions'][ch] for ch in max_chans])
    
    df_metrics['maxWaveformCh'] = max_chans[sel_idx]
    df_metrics['ks_chan_pos_x'] = chan_pos[sel_idx,0]
    df_metrics['ks_chan_pos_y'] = chan_pos[sel_idx,1]
    df_metrics['ks_labels'] = ks_labels[sel_idx]
    
    # make sure the order is correct
    assert all(df_metrics['unit_id'].values == ks_results['cluster_KSLabel']['cluster_id'][sel_idx].values), 'unit_id mismatch'

def get_spike_trains(
        synced_timestamp_files: list, 
        spike_clusters_files: list, 
        kslabels:list):
    
    
    
    # Note: UID is the id used internally in cellexplorer
    # clusID the is the label from kilosort
    # the cluster label from cluster_KSLabel.tsv and spike_clusters.npy are the same
    # by default, cell explorer will only load good unit from kilosort as defined in the cluster_KSLabel.tsv
    # defination of 'good' is  ContamPct < 10, ContamPct is based on a refactory period of 2msec
    # so the all_clusters_UIDs here is the super-set of the cluID from Cell Explorer
    
    assert len(synced_timestamp_files) == len(spike_clusters_files) == len(kslabels), 'Cluster files do not match'
    
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
         
    df_kslabels = []
    for idx_probe, kslabel in enumerate(kslabels):
        df_ks = pd.read_csv(kslabel, sep='\t')
        df_ks['cluster_id'] = df_ks['cluster_id'].apply(lambda x: all_clusters_UIDs[x])
        df_kslabels.append(df_ks)
        
    df_kslabels = pd.concat(df_kslabels)
        
    return spike_trains, all_clusters_UIDs, df_kslabels


def extract_trial_data(xr_inst_rates, evt_timestamps, trial_window, bin_duration):
    # Extract instantaneous rate triggered by some event timestamps
    num_trials = len(evt_timestamps)
    num_clusters = len(xr_inst_rates.cluID)
    time_vector = xr_inst_rates.time

    num_time_points = int(trial_window[1]-trial_window[0]) // bin_duration +1
    trial_time_vec = np.linspace(trial_window[0], trial_window[1], num_time_points)
    trial_data = np.empty((num_trials, num_time_points, num_clusters))

    for i, timestamp in enumerate(evt_timestamps):
        if np.isnan(timestamp):  # Skip NaN timestamps
            continue
        
        start_time = timestamp + trial_window[0]

        # Find the indices of the time points within the trial window
        start_idx = np.searchsorted(time_vector, start_time, side='left')
        # Extract the data for the trial and assign it to the trial_data array
        try:
            trial_data[i, :, :] = xr_inst_rates.data[start_idx:start_idx + num_time_points, :]
        except ValueError:
            # cannot find the data from the specifed timestamp, fill with NaN
            trial_data[i, :, :] = np.empty((num_time_points, num_clusters))*np.nan

    return trial_data, trial_time_vec


def build_evt_fr_xarray(fr_xr, timestamps, trial_index, name, trial_window, bin_duration, trial_based=True):
    # Construct an xr.DataArray with firing rate triggered by the specified timestamps
    
    
    if trial_based:
        trial_rates, trial_time_vec = extract_trial_data(fr_xr, timestamps, trial_window, bin_duration)
        da = xr.DataArray(
            trial_rates,
            name = name,
            coords={'trial_nb': trial_index, 'spk_event_time': trial_time_vec, 'cluID': fr_xr.cluID},
            dims=('trial_nb', 'spk_event_time', 'cluID')
            )
    else:
        # Concatenate the timestamps
        timestamps = timestamps.dropna()
        if len(timestamps)> 0:
            timestamps = timestamps.sum() #concatenate list of lists
        trial_rates, trial_time_vec = extract_trial_data(fr_xr, timestamps, trial_window, bin_duration)

        
        da = xr.DataArray(
            trial_rates,
            name = name,
            coords={f'{name}_idx': np.arange(len(timestamps)), 'spk_event_time': trial_time_vec, 'cluID': fr_xr.cluID},
            dims=(f'{name}_idx', 'spk_event_time', 'cluID')
            )
        
    
    return da

def get_cluster_UIDs_from_path(cluster_file: Path):
    # take Path or str
    cluster_file = Path(cluster_file)
    # extract session and probe name from folder structure
    session_id = cluster_file.parts[-5]
    probe_name = cluster_file.parts[-2]

    # unique cluster nb
    cluster_nbs = sorted(np.unique(np.load(cluster_file)))

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


def make_evt_dataframe(df_trials, df_conditions, df_events_cond):
    # trial onset is the trigger time
    trial_onsets = df_trials[df_trials.valid == True].timestamp

    # Defining filters for different triggering time point for behavioral phases
    behav_phases_filters = {
        'first_bar_off' : event_filters.get_first_bar_off,
        'last_bar_off' : event_filters.get_last_bar_off_before_first_spout,
        'first_spout' : event_filters.get_first_spout
    }
    trial_outcomes = df_conditions.trial_outcome


    # get the time for each important events
    df_aggregated = pd.concat([trial_outcomes, trial_onsets], axis=1)

    for ev_name, filter in behav_phases_filters.items():
        # add timestamp of particuliar behavioral phases
        df_aggregated = pd.concat([df_aggregated, event_filters.extract_event_time(df_events_cond, filter, dict())], axis=1)

    #add any extra event triggers
    extra_event_triggers = df_events_cond.attrs['extra_event_triggers']
    for ev_name in extra_event_triggers:
        df = event_filters.get_events_from_name(df_events_cond, ev_name)
        evt_col = df.groupby('trial_nb')['time'].agg(list)
        df_aggregated = pd.concat([df_aggregated, evt_col], axis=1)

    
    # rename the columns
    trigger = df_events_cond.attrs['triggers'][0]
    df_aggregated.columns = ['trial_outcome', trigger,  *behav_phases_filters.keys(), *extra_event_triggers]
    df_aggregated['reward'] = df_aggregated.first_spout + 500 # Hard coded, 500ms delay, perhaps adapt to a parameter?

    return df_aggregated


def exp_decay(x, b0, b1):
    return b0 * np.exp(-b1 * x)

def fit_spatial_decay(tmp, chan_positions):
    CHANNEL_TOLERANCE = 33
    NUM_CHANNELS_FOR_FIT = 10

    cb = np.argmax(np.ptp(tmp,axis=0)) # max. channel, similar to bombcell implementation
    max_amp = np.max(np.abs(tmp),axis=0)

    # only select channels there are close
    candidates = (chan_positions[cb,0] - chan_positions[:,0])<CHANNEL_TOLERANCE

    #calculate the relative distance from other channel to the max channel
    distance = np.sqrt(np.sum((chan_positions[cb,:] - chan_positions)**2,axis=1))
    sort_idx = np.argsort(distance)
    
    x_data = distance[sort_idx][:NUM_CHANNELS_FOR_FIT]
    y_data = max_amp[sort_idx][:NUM_CHANNELS_FOR_FIT]

    # Fit the model
    popt, pcov = curve_fit(exp_decay, x_data, y_data, p0=(0.1, 0.1))  # initial guess for b1 and b2
    
    return popt, x_data, y_data

def plot_spatial_decay_fit(popt, x_data, y_data):

    y_fit = exp_decay(x_data, popt[0], popt[1])
    
    # Plot
    plt.scatter(x_data, y_data, label='Data')
    plt.plot(x_data, y_fit, color='red', label=f'Fit: {popt[0]:.2f} * exp(-{popt[1]:.2f} * x)')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Exponential Fit')


def compute_spatial_decay(sorting_result_folder, cluID):
    cluster_labels = np.array([int(c.split('_')[-1]) for c in cluID]) #get the cluster label
    sorting_result = load_kilosort(sorting_result_folder, skip_PC_feature=True)
    if len(sorting_result.keys()) == 0:
        raise FileNotFoundError(f'Error reaching the result folder, please check the path {sorting_result_folder}')
    templates = sorting_result['templates']
    chan_positions = sorting_result['channel_positions']

    good_templates = templates[cluster_labels,:,:]

    decay_coeff = np.zeros((good_templates.shape[0],))
    for i in range(good_templates.shape[0]):
        popt, _, _= fit_spatial_decay(good_templates[i,:,:], chan_positions)
        decay_coeff[i] = popt[1]
    return decay_coeff
