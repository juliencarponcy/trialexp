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
from workflow.scripts import settings

from workflow.scripts import settings
from trialexp.utils.ephys_utilities import create_ephys_rsync


#%% Load inputs
ephys_sync_done_path = str(Path(settings.debug_folder) / 'processed' / 'ephys_sync.done')

(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
 [ephys_sync_done_path], 'ephys_sync')


# %% Get path to important files

verbose = True
sorter_name = 'kilosort3'

kilosort_folder = Path(sinput.kilosort_path)
pycontrol_path = (kilosort_folder.parents[1]/'pycontrol')
pycontrol_path = list(pycontrol_path.glob('*.txt'))[0]
sync_path = kilosort_folder.parents[1]/'ephys'

# %%

for probe_dir in  kilosort_folder.glob('Probe*'):
    # change the spike timing based on the rsync signal from both pycontrol and openephys

    rec_prop = pd.read_csv(probe_dir/'rec_prop.csv').iloc[0]
    rsync = create_ephys_rsync(str(pycontrol_path), sync_path, rec_prop.tstart)
    ks3_path = probe_dir /'spike_times.npy'
    
    if not rsync:
        raise ValueError('Error: cannot create rsync')
    
    if not ks3_path.exists():
        raise FileNotFoundError('Error: cannnot find the spike_times.npy from kilosort.')
    
    spike_times = np.load(ks3_path)

    # Careful with the forced 30 value, sampling rate slightly diverging from 30kHz
    # should be dealt-with by open-ephys/ks3, but need to be checked
    spike_times_converted = spike_times/(rec_prop.sample_rate/1000) # /1000 to be in ms
    synced_spike_times = rsync.B_to_A(spike_times_converted)

    np.save(ks3_path.parent / 'rsync_corrected_spike_times.npy', synced_spike_times)
    

# %%
