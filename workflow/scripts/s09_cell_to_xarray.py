'''
Script to convert raw timestamps to xarray datasets based on sorting and 
other previous step for behaviour and photometry
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
from trialexp.process.pyphotometry.utils import *
from trialexp.process.pycontrol import event_filters
from trialexp.process.ephys.spikes_preprocessing import \
    merge_cell_metrics_and_spikes, bin_spikes_from_all_probes, \
    bin_spikes_from_all_probes_averaged, bin_spikes_from_all_probes_by_trials, \
    halfgaussian_filter1d 


#%% Load inputs


(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [settings.debug_folder + r'/processed/xr_spikes_trials.nc'],
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
ce_cell_metrics_files = list(Path(sinput.sorting_path).glob('*/sorter_output/cell_metrics_df_full.pkl'))

# session outputs
session_figure_path = Path(sinput.xr_session).parent / 'figures'
session_waveform_path = Path(sinput.xr_session).parent / 'waveforms'

#%% Variables definition

bin_duration = 20 # ms for binning spike timestamps
sigma_ms = 200 # ms for half-gaussian kernel size (1SD duration)

#%% File loading
ce_cell_metrics_df_full = pd.DataFrame()
for cell_metrics_df_file in ce_cell_metrics_files:
    ce_cell_metrics_df_full = pd.concat([ce_cell_metrics_df_full, pd.read_pickle(cell_metrics_df_file)])

si_cell_metrics_df = pd.read_pickle(session_waveform_path / 'si_metrics_df.pkl')
si_cell_metrics_df = si_cell_metrics_df.set_index('UID')

xr_session = xr.open_dataset(sinput.xr_session)
xr_photometry = xr.open_dataset(Path(sinput.xr_session).parent / 'xr_photometry.nc')

df_events_cond_path = Path(sinput.xr_session).parent / 'df_events_cond.pkl'
df_events_cond = pd.read_pickle(df_events_cond_path)

df_conditions_path = Path(sinput.xr_session).parent / 'df_conditions.pkl'
df_conditions = pd.read_pickle(df_conditions_path)

df_trials_path = Path(sinput.xr_session).parent / 'df_trials.pkl'
df_trials = pd.read_pickle(df_trials_path)

df_pycontrol_path = Path(sinput.xr_session).parent / 'df_pycontrol.pkl'
df_pycontrol = pd.read_pickle(df_pycontrol_path)

trigger = df_events_cond.attrs['triggers'][0]
# trial_window = xr_session.attrs['trial_window']
trial_window = [-500, 500]

#%% Defining filters for different triggering time point for behavioral phases

trial_onsets = df_trials[df_trials.valid == True].timestamp

behav_phases_filters = {
    # 'trial_onset' : None,
    'first_bar_off' : event_filters.get_first_bar_off,
    'last_bar_off' : event_filters.get_last_bar_off_before_first_spout,
    'spout' : event_filters.get_first_spout
}
trial_outcomes = df_conditions.trial_outcome

df_aggregated = pd.concat([trial_onsets, trial_outcomes], axis=1)

for ev_name, filter in behav_phases_filters.items():
    # add timestamp of particuliar behavioral phases
    # print(ev_name)
    df_aggregated = pd.concat([df_aggregated, event_filters.extract_event_time(df_events_cond, filter, dict())], axis=1)

# rename the columns
df_aggregated.columns = ['timestamp', 'trial_outcome', *behav_phases_filters.keys()]

# At the moment, this is the time we used to trigger extraction around trials
# This may be adapted to different triggering like first bar_off or first spout
trial_times = df_pycontrol[df_pycontrol.name == trigger].time.values.astype(int)

# %% Bin spikes from all probes and aggegate over trials around trial_times
# Loop over all behavioural phases 
behav_phases = behav_phases_filters.keys()
outcomes = df_aggregated.trial_outcome.unique()

for ev_idx, ev_name in enumerate(behav_phases):
    
    # full plots with nrows = len(behav_phases) and ncols = len(outcomes) were too crowded
    figure, axes = plt.subplots(1, len(outcomes), sharex=True,
                            figsize=(4*len(outcomes), 5))
    
    figure.suptitle(f'Cluster responses to {ev_name} sorted by max z-scored peak response time')
    for outcome_idx, trial_outcome in enumerate(outcomes):

        # Selecting only trial times for a particular outcome
        ev_times = df_aggregated[df_aggregated.trial_outcome == trial_outcome].loc[:,ev_name].values

        # bin and average spike trains around these trial_times
        all_probes_binned_array_averaged, spike_time_bins, all_clusters_UIDs = bin_spikes_from_all_probes_averaged(
            synced_timestamp_files = synced_timestamp_files, 
            spike_clusters_files = spike_clusters_files,
            trial_times = ev_times, 
            trial_window = trial_window,
            bin_duration = bin_duration)

        # Applying Kernel to binned firing rate

        # May not be applied until bins are aggregated by trials
        # Return the binned spike array convoluted by half-gaussian window (so FR do not increases before spike [but binning])
        convoluted_binned_array = halfgaussian_filter1d(
            all_probes_binned_array_averaged, 
            bin_duration=bin_duration, 
            sigma_ms = sigma_ms, 
            axis=-1, 
            output=None,
            mode="constant", 
            cval=0.0, 
            truncate=4.0) # truncate the window at 4SD

        scaler = StandardScaler()
        # z-scored firing rate
        # .T).T to z-score along time dimension
        z_conv_FR = scaler.fit_transform(convoluted_binned_array.T).T

        # Combining spikes rates and scored

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

        # Combine all cell metrics (SpikeInterface + CellExplorer) and spike bins


        session_ce_cell_metrics = merge_cell_metrics_and_spikes(ce_cell_metrics_files, all_clusters_UIDs)
        

        all_cell_metrics = pd.merge(si_cell_metrics_df,session_ce_cell_metrics, left_index=True, right_index=True, how='outer')

        xr_cell_metrics = all_cell_metrics.to_xarray()

        # Create the xr dataset
        xr_spikes_averaged = xr.merge([spike_fr_xr, spike_zscored_xr, xr_cell_metrics])
        xr_spikes_averaged.attrs['bin_duration'] = bin_duration

        # Save
        xr_spikes_averaged.to_netcdf(Path(sinput.xr_session).parent / f'xr_spikes_averaged_{ev_name}.nc', engine='h5netcdf')


        # Preview of cluster responses to trigger
        # Sort clusters by peak time in the trial
        idx_max_FR = np.argmax(convoluted_binned_array,1)

        cluster_ID_sorted_max = np.argsort(idx_max_FR)



        sns.heatmap(spike_zscored_xr[np.flip(cluster_ID_sorted_max),:], 
                    vmin=-3, vmax=3, ax=axes[outcome_idx])

        # sns.heatmap(spike_fr_xr[np.flip(cluster_ID_sorted_max),:],
        #             vmin=0, vmax=30, ax=axes[ev_idx,outcome_idx])
        # need to specify 
        axes[outcome_idx].set_title(f'{trial_outcome} trials (n={str(len(ev_times))})')
        # axes[outcome_idx].set_title('Firing rate (spikes/s) around trial')# %% Continuous All session long binning
        axes[outcome_idx].set_xlabel(f'Time around {ev_name}')
        axes[outcome_idx].set_ylabel(f'Cluster #')
        # TODO: proper xticks will take some work
        # axes[outcome_idx].set_xticks([trial_window[0],0,trial_window[1]])
    # Saving figure to each behav event
    figure.savefig(session_figure_path / f'cluster_responses_to_{ev_name}.png')

#%% Continuous session-long binning

# bin all the clusters from all probes continuously, return nb_of_spikes per bin * (1000[ms]/bin_duratiion[ms])
# so if 1 spike in 20ms bin -> inst. FR = 1 * (1000/20) = 50Hz

# if bin duration == 1ms, we will have a BOOL arary (@1000Hz)

# bin_spikes_from_all_probes : All session. Dimensions  time x cluster [long]
all_probes_binned_array, spike_time_bins, all_clusters_UIDs = bin_spikes_from_all_probes(
    synced_timestamp_files = synced_timestamp_files, 
    spike_clusters_files = spike_clusters_files, 
    bin_duration = bin_duration)

# %% Applying Kernel to binned firing rate (Continuous long session)
# if we want the gaussian to have sigma = 0.2s


# May not be applied until bins are aggregated by trials
# Return the binned spike array convoluted by half-gaussian window (so FR do not increases before spike [but binning])
convoluted_binned_array = halfgaussian_filter1d(
    all_probes_binned_array, 
    bin_duration=bin_duration, 
    sigma_ms = sigma_ms, 
    axis=-1, 
    output=None,
    mode="constant", 
    cval=0.0, 
    truncate=4.0) # truncate the window at 4SD
# %%
scaler = StandardScaler()
# z-scored firing rate
# .T).T to z-score along time dimension
z_conv_FR = scaler.fit_transform(convoluted_binned_array.T).T

spike_fr_xr = xr.DataArray(
    all_probes_binned_array,
    name = 'spikes_FR',
    coords={'time':spike_time_bins[1:] - (bin_duration/2), 'UID': all_clusters_UIDs},
    dims=('time', 'UID')
)

spike_zscored_xr = xr.DataArray(
    z_conv_FR,
    name = 'spikes_Zscore',
    coords={'time':spike_time_bins[1:] - (bin_duration/2), 'UID': all_clusters_UIDs},
    dims=('time', 'UID')
)
# Create the xr dataset
xr_spikes_session = xr.merge([spike_fr_xr, spike_zscored_xr, xr_cell_metrics, xr_session])
xr_spikes_session.attrs['bin_duration'] = bin_duration

# Save it
xr_spikes_session.to_netcdf(Path(sinput.xr_session).parent / 'xr_spikes_session.nc', engine='h5netcdf')


# %% Binning by trials, 3D output (trial, time, cluster)
all_probes_binned_array_by_trials, spike_time_bins, all_clusters_UIDs = bin_spikes_from_all_probes_by_trials(
    synced_timestamp_files = synced_timestamp_files, 
    spike_clusters_files = spike_clusters_files,
    trial_times = trial_times,
    trial_window = trial_window,
    bin_duration = bin_duration,
    normalize = False)

#%% Half-Gaussian convolution
convoluted_binned_array = halfgaussian_filter1d(
    all_probes_binned_array_by_trials, 
    bin_duration=bin_duration, 
    sigma_ms = sigma_ms, 
    axis=-1, 
    output=None,
    mode="constant", 
    cval=0.0, 
    truncate=4.0) # truncate the window at 4SD
# %% Combining spikes rates and scored

spike_fr_xr = xr.DataArray(
    all_probes_binned_array_by_trials,
    name = 'spikes_FR',
    coords={'trial_nb': list(range(1,len(trial_times)+1)), 'event_time':spike_time_bins[1:] - (bin_duration/2), 'UID': all_clusters_UIDs},
    dims=('trial_nb', 'event_time', 'UID')
)

# Create the xr dataset
xr_spikes_trials = xr.merge([spike_fr_xr, xr_cell_metrics, xr_photometry])
xr_spikes_trials.attrs['bin_duration'] = bin_duration
# Save it
xr_spikes_trials.to_netcdf(Path(sinput.xr_session).parent / 'xr_spikes_trials.nc', engine='h5netcdf')

# %%
