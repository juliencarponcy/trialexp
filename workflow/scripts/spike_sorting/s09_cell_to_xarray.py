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

from elephant import statistics, kernels

from snakehelper.SnakeIOHelper import getSnake

from workflow.scripts import settings
from trialexp.process.pyphotometry.utils import *
from trialexp.process.ephys.spikes_preprocessing import \
    build_evt_fr_xarray, make_evt_dataframe, merge_cell_metrics_and_spikes, get_spike_trains, \
    get_max_timestamps_from_probes, extract_trial_data

from trialexp.process.ephys.utils import dataframe_cleanup
#%% Load inputs


(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [settings.debug_folder + r'/processed/xr_spikes_full_session.nc'],
  'cells_to_xarray')

# %% Path definitions

verbose = True

root_path = Path(os.environ['SESSION_ROOT_DIR'])

# Get probe names from folder path
probe_names = [folder.stem for folder in list(Path(sinput.sorting_path).glob('*'))]

# Fetch file paths from all probes
synced_timestamp_files = list(Path(sinput.sorting_path).glob('*/rsync_corrected_spike_times.npy'))
spike_clusters_files = list(Path(sinput.sorting_path).glob('*/spike_clusters.npy'))
ce_cell_metrics_full= Path(sinput.cell_matrics_full)

# session outputs
session_figure_path = Path(sinput.xr_session).parent / 'figures'
session_waveform_path = Path(sinput.xr_session).parent / 'waveforms'

#%% Variables definition

bin_duration = 10 # ms for binning spike timestamps
sigma_ms = 200 # ms for half-gaussian kernel size (1SD duration)
trial_window = (500, 500) # time before and after timestamps to extract

#%% File loading

xr_cell_metrics = xr.load_dataset(ce_cell_metrics_full)
xr_session = xr.open_dataset(sinput.xr_session)
session_root_path = Path(sinput.xr_session).parent
df_events_cond = pd.read_pickle(session_root_path / 'df_events_cond.pkl')
df_conditions = pd.read_pickle(session_root_path / 'df_conditions.pkl')
df_trials = pd.read_pickle(session_root_path / 'df_trials.pkl')

# trigger = df_events_cond.attrs['triggers'][0]


#%% Gathering trial outcomes and timestamps of different phases


df_aggregated = make_evt_dataframe(df_trials, df_conditions, df_events_cond)

behav_phases = list(df_aggregated.columns)
behav_phases.remove('trial_outcome')# exclude trial outcome column

#%% Extract instantaneous rates (continuous) from spike times (discrete)

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
#%% Aggregating Clusters metadata

# Merge CellExplorer metrics with all clusters in kilosort data
# session_ce_cell_metrics = merge_cell_metrics_and_spikes(ce_cell_metrics_full, all_clusters_UIDs)
# Combine all cell metrics (SpikeInterface + CellExplorer) 
# all_cell_metrics = pd.merge(ce_cell_metrics_full,session_ce_cell_metrics, left_index=True, right_index=True, how='outer')
# discard cell metrics without 'x' position (trick which give same nb of clusters as binned data)
# all_cell_metrics.dropna(axis=0, subset='x', inplace=True)
# Clean dataframe turning object columns into text columns 
# all_cell_metrics = dataframe_cleanup(all_cell_metrics)
# Create the xr dataset
# all_cell_metrics = ce_cell_metrics_df_full.set_index('UID')
# xr_cell_metrics= all_cell_metrics.to_xarray()

# all_cell_metrics['peakVoltage_sorted'].to_xarray()

#%% Building session-wide xarray dataset

# Session_time_vector is already syncronized with pycontrol
spike_fr_xr_session = xr.DataArray(
    inst_rates,
    name = f'spikes_FR_session',
    coords={'time': session_time_vector, 'cluID': all_clusters_UIDs},
    dims=('time', 'cluID')
    )

spike_zfr_xr_session = xr.DataArray(
    z_inst_rates,
    name = f'spikes_zFR_session',
    coords={'time': session_time_vector, 'cluID': all_clusters_UIDs},
    dims=('time', 'cluID')
    )

# Take reference only from the cells included in Cell Explorer
xr_spikes_session = xr.merge([xr_cell_metrics, spike_fr_xr_session, spike_zfr_xr_session], join='inner')
xr_spikes_session.attrs['bin_duration'] = bin_duration
xr_spikes_session.attrs['sigma_ms'] = sigma_ms
xr_spikes_session.attrs['kernel'] = 'ExponentialKernel'


#%% Extracting instantaneous rates by trial for all behavioural phases

da_list = []
for ev_idx, ev_name in enumerate(behav_phases): 

    timestamps = df_aggregated[ev_name]
    # # Binning by trials, 3D output (trial, time, cluster)
    da_list.append(build_evt_fr_xarray(spike_fr_xr_session, timestamps,
                                       df_aggregated.index, f'spikes_FR.{ev_name}', 
                                        trial_window, bin_duration))
    
    da_list.append(build_evt_fr_xarray(spike_zfr_xr_session,
                                       timestamps, df_aggregated.index, f'spikes_zFR.{ev_name}',
                                        trial_window, bin_duration))



#%% Adding trials metadata

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

xr_spikes_trials = xr.merge([xr_cell_metrics, trial_out, trial_ts, *da_list], join='inner')
xr_spikes_trials.attrs['bin_duration'] = bin_duration
xr_spikes_trials.attrs['sigma_ms'] = sigma_ms
xr_spikes_trials.attrs['kernel'] = 'ExponentialKernel'

#%% Save
xr_spikes_trials.to_netcdf(Path(soutput.xr_spikes_trials), engine='h5netcdf')
xr_spikes_trials.close()

#%%
xr_spikes_trials