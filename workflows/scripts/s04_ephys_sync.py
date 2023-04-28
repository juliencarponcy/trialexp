'''
Script to compute cell metrics by CellExplorer from Kilosort3 results
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
from workflows.scripts import settings

from workflows.scripts import settings
from trialexp.utils.ephys_utilities import create_ephys_rsync


#%% Load inputs
ephys_sync_done_path = str(Path(settings.debug_folder) / 'processed' / 'ephys_sync.done')

(sinput, soutput) = getSnake(locals(), 'workflows/spikesort.smk',
 [ephys_sync_done_path], 'ephys_sync')


# %%

verbose = True
rec_properties_path = Path(sinput.rec_properties)
pycontrol_path = rec_properties_path.parent.parent / 'pycontrol'
pycontrol_path = list(pycontrol_path.glob('*.txt'))[0]
sync_path = rec_properties_path.parent

rec_properties = pd.read_csv(rec_properties_path, index_col= None)

# Only select longest syncable recordings to sort
idx_to_sort = rec_properties[rec_properties.longest == True].index.values

# %%

for idx_rec in  idx_to_sort:

    exp_nb = rec_properties.exp_nb.iloc[idx_rec]
    rec_nb = rec_properties.rec_nb.iloc[idx_rec]
    AP_stream = rec_properties.AP_stream.iloc[idx_rec]

    exp_start_datetime = datetime.strptime(rec_properties.exp_datetime.iloc[idx_rec], '%Y-%m-%d %H:%M:%S')
    tstart = rec_properties.tstart.iloc[idx_rec]


    # TODO: modify the move_to_server shell command to avoid double nesting kilosort3/kilosort3
    ks3_path = rec_properties_path.parent.parent / 'processed' / 'kilosort3'
    if 'ProbeA' in AP_stream:
        ks3_path = ks3_path / 'ProbeA' / 'sorter_output' / 'spike_times.npy'
    elif 'ProbeB' in AP_stream:
        ks3_path = ks3_path / 'ProbeB' / 'sorter_output' / 'spike_times.npy'
    else:
        raise Exception(f'There is an issue with the stream name: {AP_stream}')

    if not ks3_path.exists():
        raise Exception(f'Cannot find the spike_times.npy file at: {ks3_path}')


   
    rsync = create_ephys_rsync(str(pycontrol_path), sync_path, tstart)
    if rsync:
        spike_times = np.load(ks3_path)
        # print(np.nanmin(spike_times), np.nanmax(spike_times))

        # Careful with the forced 30 value, sampling rate slightly diverging from 30kHz
        # should be dealt-with by open-ephys/ks3, but need to be checked
        spike_times_converted = spike_times/(rec_properties.sample_rate.iloc[idx_rec]/1000) # /1000 to be in ms
        synced_spike_times = rsync.B_to_A(spike_times_converted)
        # print(np.nanmin(synced_spike_times), np.nanmax(synced_spike_times))

        np.save(ks3_path.parent / 'rsync_corrected_spike_times.npy', synced_spike_times)
    else:
        raise Warning(f'{pycontrol_path} and {sync_path} with tstart = {tstart}sec could not be synchronized, no corrected spike times saved')
    # %%
