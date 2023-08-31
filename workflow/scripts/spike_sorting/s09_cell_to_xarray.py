'''
Script to convert raw timestamps to xarray datasets based on sorting and 
other previous step for behaviour and photometry
'''
#%%
import os
import pickle
from pathlib import Path

import neo
import numpy as np
import pandas as pd
import quantities as pq
import xarray as xr
from elephant import kernels, statistics
from elephant.conversion import BinnedSpikeTrain
from sklearn.preprocessing import StandardScaler
from snakehelper.SnakeIOHelper import getSnake

from trialexp.process.ephys.spikes_preprocessing import (
    build_evt_fr_xarray, extract_trial_data, get_max_timestamps_from_probes,
    get_spike_trains, make_evt_dataframe, merge_cell_metrics_and_spikes)
from trialexp.process.ephys.utils import (binned_firing_rate, compare_fr_with_random)
from trialexp.process.pyphotometry.utils import *
from workflow.scripts import settings

#%% Load inputs


(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [settings.debug_folder + r'/processed/xr_spikes_trials.nc'],
  'cells_to_xarray')

# %% Path definitions

verbose = True

root_path = Path(os.environ['SESSION_ROOT_DIR'])

# Get probe names from folder path
sorting_path = Path(sinput.xr_session).parent/'kilosort'
probe_names = [folder.stem for folder in list(sorting_path.glob('*'))]

# Fetch file paths from all probes
synced_timestamp_files = list(sorting_path.glob('*/sorter_output/rsync_corrected_spike_times.npy'))
spike_clusters_files = list(sorting_path.glob('*/sorter_output/spike_clusters.npy'))
ce_cell_metrics_full= Path(sinput.cell_matrics_full)

# session outputs
session_figure_path = Path(sinput.xr_session).parent / 'figures'
session_waveform_path = Path(sinput.xr_session).parent / 'waveforms'

#%% Variables definition

bin_duration = 10 # ms for binning spike timestamps
#ms for half-gaussian kernel size (1SD duration), 
# 20ms as suggest from Martin et al (1999) https://www.sciencedirect.com/science/article/pii/S0165027099001272#FIG4
sigma_ms = 20 #
trial_window = (500, 2000) # time before and after timestamps to extract

#%% File loading

xr_cell_metrics = xr.load_dataset(ce_cell_metrics_full)
xr_session = xr.load_dataset(sinput.xr_session)
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

# An expontentail kernel may give a false sense of decreasing firing rate
# A Gaussian kernel can better capture the uncertainty of the firing rate estimation
# For details for various methods to estimate firing rate, see
# Rimjhim (2019)
# https://www.sciencedirect.com/science/article/pii/S0303264719301492?via%3Dihub
# kernel = kernels.ExponentialKernel(sigma=sigma_ms*pq.ms)
kernel = kernels.GaussianKernel(sigma=sigma_ms*pq.ms)


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

#%%
# constructing time vector
session_time_vector = np.linspace(bin_duration,inst_rates.shape[0]*bin_duration,inst_rates.shape[0]) - bin_duration/2
# z-scoring firing rate
scaler = StandardScaler()
z_inst_rates = scaler.fit_transform(inst_rates)

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
xr_spikes_fr = xr.merge([spike_fr_xr_session, spike_zfr_xr_session], join='inner')
xr_spikes_fr = xr_spikes_fr.sel(cluID=xr_cell_metrics.cluID) #only choose the 'good' cell from kilosort
xr_spikes_fr.attrs['bin_duration'] = bin_duration
xr_spikes_fr.attrs['sigma_ms'] = sigma_ms
xr_spikes_fr.attrs['kernel'] = 'ExponentialKernel'

xr_spikes_fr.to_netcdf(Path(soutput.xr_spikes_fr), engine='h5netcdf')
xr_spikes_fr.close()

# also save the neo spike train object
with open(Path(soutput.neo_spike_train),'wb') as f:
    pickle.dump(spike_trains, f)


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
    name= 'trial_outcome',
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
xr_spikes_trials.attrs['trial_window'] = trial_window


xr_spikes_trials.to_netcdf(Path(soutput.xr_spikes_trials), engine='h5netcdf')
xr_spikes_trials.close()

