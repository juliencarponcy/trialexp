'''
Script to synchronize the spike_time with pycontrol
'''
#%%
import os
import warnings
import shutil

from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from snakehelper.SnakeIOHelper import getSnake
from trialexp import config
from trialexp.utils.ephys_utilities import create_ephys_rsync


#%% Load inputs
ephys_sync_done_path = str(Path(config.debug_folder) / 'processed' / 'ephys_sync.done')

(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
 [ephys_sync_done_path], 'ephys_sync')


# %% Get path to important files

verbose = True
sorter_name = 'kilosort4'

kilosort_folder = Path(sinput.df_quality_metrics).parent / sorter_name
pycontrol_path = (kilosort_folder.parents[1]/'pycontrol')
pycontrol_path = (list(pycontrol_path.glob('*.txt')) + list(pycontrol_path.glob('*.tsv')))[0]
sync_path = kilosort_folder.parents[1]/'ephys'

# %%

for probe_dir in  kilosort_folder.glob('Probe*'):
    # change the spike timing based on the rsync signal from both pycontrol and openephys
    # event time from open ephys count from the beginning of the acquisition, not recording
    # kilosort time always start from the beginning of the recording

    rec_prop = pd.read_csv(probe_dir/f'../rec_prop_{probe_dir.name}.csv').iloc[0]
    rsync = create_ephys_rsync(str(pycontrol_path), sync_path, rec_prop.tstart)
    ks_path = probe_dir /'spike_times.npy'
    
    if not rsync:
        raise ValueError('Error: cannot create rsync')
    
    if not ks_path.exists():
        raise FileNotFoundError('Error: cannnot find the spike_times.npy from kilosort.')
    
    spike_times = np.load(ks_path)

    # Careful with the forced 30 value, sampling rate slightly diverging from 30kHz
    # should be dealt-with by open-ephys/ks3, but need to be checked
    # Note: anything that happen before the first ephys rsync pulse and after the last ephys rsync
    # pulse will not be synchronized and result in nan
    # Will keep the nan in synced_spike_times to be consistent with CellExplorer spike count
    spike_times_converted = spike_times/(rec_prop.sample_rate/1000) # /1000 to be in ms
    synced_spike_times = rsync.A_to_B(spike_times_converted) #convert ephys to pycontrol time
    
    # do some sanity check to make sure we are syncing the correct file
    # Unlikely that there is no spike for 1 minute
    lastspike2end = abs(np.nanmax(synced_spike_times)/1000 - rec_prop.duration)
    # assert lastspike2end<60, f'Error: last spike appears too far away from end of recording {lastspike2end}. Potential sync issues'

    # the synced spike time is in ms
    np.save(ks_path.parent / 'rsync_corrected_spike_times.npy', synced_spike_times)


# %%
