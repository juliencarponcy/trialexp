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
import quantities as pq

from sklearn.preprocessing import StandardScaler

import seaborn as sns
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns

from elephant import statistics, kernels

from snakehelper.SnakeIOHelper import getSnake

from workflow.scripts import settings
from trialexp.process.pyphotometry.utils import *
from trialexp.process.pycontrol import event_filters
from trialexp.process.ephys.spikes_preprocessing import \
    merge_cell_metrics_and_spikes, get_spike_trains, \
    get_max_timestamps_from_probes, extract_trial_data


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

bin_duration = 10 # ms for binning spike timestamps
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
trial_window = (500, 500) # time before and after timestamps to extract

#%% Gathering trial outcomes and timestamps of different phases

trial_onsets = df_trials[df_trials.valid == True].timestamp

# Defining filters for different triggering time point for behavioral phases
behav_phases_filters = {
    # 'trial_onset' : None,
    'first_bar_off' : event_filters.get_first_bar_off,
    'last_bar_off' : event_filters.get_last_bar_off_before_first_spout,
    'spout' : event_filters.get_first_spout
}
trial_outcomes = df_conditions.trial_outcome


df_aggregated = pd.concat([trial_outcomes, trial_onsets], axis=1)

for ev_name, filter in behav_phases_filters.items():
    # add timestamp of particuliar behavioral phases
    print(ev_name)
    df_aggregated = pd.concat([df_aggregated, event_filters.extract_event_time(df_events_cond, filter, dict())], axis=1)

# rename the columns
df_aggregated.columns = ['trial_outcome', 'trial_onset',  *behav_phases_filters.keys()]
# Adding reward phase
df_aggregated['reward'] = df_aggregated.spout + 500 # Hard coded, 500ms delay, perhaps adapt to a parameter?
# Adding rest phase
df_aggregated['rest'] = df_aggregated.trial_onset - 2000 # Hard coded, 2000ms resting period, perhaps adapt to a parameter?

# %% Extract instantaneous rates (continuous) from spike times (discrete)
behav_phases = df_aggregated.columns[1:] # exclude trial outcome column
trial_outcomes = df_conditions.trial_outcome.unique()

# Use SpikeTrain class from neo.core
spike_trains, all_clusters_UIDs = get_spike_trains(
                    synced_timestamp_files = synced_timestamp_files, 
                    spike_clusters_files = spike_clusters_files)

t_stop = get_max_timestamps_from_probes(synced_timestamp_files)

kernel = kernels.ExponentialKernel(sigma=sigma_ms*pq.ms)

inst_rates = statistics.instantaneous_rate(
                    spiketrains= spike_trains,
                    sampling_period = bin_duration * pq.ms, 
                    kernel=kernel, 
                    cutoff=5.0, 
                    t_start=None, 
                    t_stop=t_stop * pq.ms, 
                    trim=False, 
                    center_kernel=True, 
                    border_correction=False)
# constructing time vector
session_time_vector = np.linspace(bin_duration,inst_rates.shape[0]*bin_duration,inst_rates.shape[0]) - bin_duration/2
# z-scoring firing rate
scaler = StandardScaler()
z_inst_rates = scaler.fit_transform(inst_rates)
#%% Extracting instantaneous rates by trial for all behavioural phases

trial_rates = dict()
trial_zrates = dict()

for ev_idx, ev_name in enumerate(behav_phases): 

    timestamps = df_aggregated[ev_name]
    # Binning by trials, 3D output (trial, time, cluster)
    trial_rates[ev_name], trial_time_vec = extract_trial_data(inst_rates, session_time_vector, timestamps, trial_window, bin_duration)

    trial_zrates[ev_name], trial_time_vec = extract_trial_data(z_inst_rates, session_time_vector, timestamps, trial_window, bin_duration)


#%% Convert trial firing rates to DataArrays

# empty dictionaries of DataArrays
spike_fr_xr = list()
spike_zfr_xr = list()

for ev_idx, ev_name in enumerate(behav_phases): 

    spike_fr_xr.append(xr.DataArray(
        trial_rates[ev_name],
        name = f'spikes_FR.{ev_name}',
        coords={'trial_nb': df_aggregated.index, 'event_time': trial_time_vec, 'UID': all_clusters_UIDs},
        dims=('trial_nb', 'event_time', 'UID')
        )
    )

    spike_zfr_xr.append(xr.DataArray(
        trial_zrates[ev_name],
        name = f'spikes_zFR.{ev_name}',
        coords={'trial_nb': df_aggregated.index, 'event_time': trial_time_vec, 'UID': all_clusters_UIDs},
        dims=('trial_nb', 'event_time', 'UID')
        )
    )

#%% Adding trials and clusters metadata

# Merge CellExplorer metrics with all clusters in kilosort data
session_ce_cell_metrics = merge_cell_metrics_and_spikes(ce_cell_metrics_files, all_clusters_UIDs)
# Combine all cell metrics (SpikeInterface + CellExplorer) 
all_cell_metrics = pd.merge(si_cell_metrics_df,session_ce_cell_metrics, left_index=True, right_index=True, how='outer')
# discard cell metrics without 'x' position (trick which give same nb of clusters as binned data)
all_cell_metrics.dropna(axis=0, subset='x', inplace=True)
# Create the xr dataset
xr_cell_metrics= all_cell_metrics.to_xarray()

# trial outcomes
trial_out = xr.DataArray(
    df_aggregated.trial_outcome,
    name= 'trial_outome',
    coords= {'trial_nb': df_aggregated.index},
    dims = ('trial_nb')

)

# trial timestamps
trial_ts = xr.DataArray(
    df_aggregated.iloc[:,1:],
    name= 'trial_timestamps',
    coords= {'trial_nb': df_aggregated.index, 'trial_phase': df_aggregated.columns[1:]},
    dims = ('trial_nb', 'trial_phase')

)

xr_spikes_trials = xr.merge([*spike_fr_xr, *spike_zfr_xr, xr_cell_metrics, trial_out, trial_ts])
xr_spikes_trials.attrs['bin_duration'] = bin_duration
xr_spikes_trials.attrs['sigma_ms'] = sigma_ms

# Save
xr_spikes_trials.to_netcdf(Path(sinput.xr_session).parent / f'xr_spikes_trials.nc', engine='h5netcdf')

#%% Saving similar xarray Dataset but this time with the behavioral phase as an extra dimension
# instead of saving one DataArray per behavioral phase (this to leave implementation choice for future processing)

# stacking the 3D arrays for firing rates (trial_nb x event_time x UID) onto a 4th dimension (trial_phase)
trial_rates_phases = np.stack([trial_rate_array for trial_rate_array in trial_rates.values()],axis=3)
trial_zrates_phases = np.stack([trial_zrate_array for trial_zrate_array in trial_zrates.values()],axis=3)

#%%
spike_fr_xr_phases = xr.DataArray(
    trial_rates_phases,
    name = f'spikes_FR',
    coords={'trial_nb': df_aggregated.index, 'event_time': trial_time_vec, 'UID': all_clusters_UIDs, 'trial_phase': df_aggregated.columns[1:]},
    dims=('trial_nb', 'event_time', 'UID', 'trial_phase')
    )

spike_zfr_xr_phases = xr.DataArray(
    trial_zrates_phases,
    name = f'spikes_zFR',
    coords={'trial_nb': df_aggregated.index, 'event_time': trial_time_vec, 'UID': all_clusters_UIDs, 'trial_phase': df_aggregated.columns[1:]},
    dims=('trial_nb', 'event_time', 'UID', 'trial_phase')
    )    

xr_spikes_trials_phases = xr.merge([spike_fr_xr_phases, spike_zfr_xr_phases, xr_cell_metrics, trial_out, trial_ts])
xr_spikes_trials_phases.attrs['bin_duration'] = bin_duration
xr_spikes_trials_phases.attrs['sigma_ms'] = sigma_ms

# Save
xr_spikes_trials_phases.to_netcdf(Path(sinput.xr_session).parent / f'xr_spikes_trials_phases.nc', engine='h5netcdf')
#%%

# Preview of cluster responses to trigger
# Sort clusters by peak time in the trial
# idx_max_FR = np.argmax(convoluted_binned_array,1)

# cluster_ID_sorted_max = np.argsort(idx_max_FR)
