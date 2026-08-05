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
from trialexp import config
from trialexp import config
from trialexp.utils.ephys_utilities import create_ephys_rsync


#%% Load inputs
ephys_sync_done_path = str(Path(config.debug_folder) / 'processed' / 'ephys_sync.done')

(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
 [ephys_sync_done_path], 'ephys_sync')


# %%

verbose = True
sorter_name = 'kilosort3'

rec_properties_path = Path(sinput.rec_properties)
pycontrol_path = rec_properties_path.parent.parent / 'pycontrol'
pycontrol_path = list(pycontrol_path.glob('*.txt'))[0]
sync_path = rec_properties_path.parent

rec_properties = pd.read_csv(rec_properties_path, index_col = 0)

rec_properties['synced'] = False

processed_folder = rec_properties_path.parent.parent / 'processed' / sorter_name
# Only select probe folders where the results of the sorting can be found.
probe_folders = [str(processed_folder / probe_folder / 'sorter_output') 
                  for probe_folder in os.listdir(processed_folder)
                  if 'spike_clusters.npy' in os.listdir(processed_folder / probe_folder / 'sorter_output')]

# Only select longest syncable recordings to sort
if 'sorting_error' in rec_properties.columns.values.tolist():
    idx_to_sort = rec_properties[(rec_properties.longest == True) & 
                                (rec_properties.sorting_error == False)].index.values
else:
    idx_to_sort = rec_properties[rec_properties.longest == True].index.values
# %%

for idx_rec in  idx_to_sort:

    exp_nb = rec_properties.exp_nb.iloc[idx_rec]
    rec_nb = rec_properties.rec_nb.iloc[idx_rec]
    AP_stream = rec_properties.AP_stream.iloc[idx_rec]

    if len(rec_properties.exp_datetime.iloc[idx_rec]) == 19: # for datetime including seconds
        exp_start_datetime = datetime.strptime(rec_properties.exp_datetime.iloc[idx_rec], '%Y-%m-%d %H:%M:%S')
    elif len(rec_properties.exp_datetime.iloc[idx_rec]) == 16: # for datetime with seconds == 00, happened for kms058-2023-03-20-132658
        exp_start_datetime = datetime.strptime(rec_properties.exp_datetime.iloc[idx_rec], '%d/%m/%Y %H:%M')
    
    tstart = rec_properties.tstart.iloc[idx_rec]

    # TODO: extract probe names from folder structure
    ks3_path = rec_properties_path.parent.parent / 'processed' / sorter_name
    if 'ProbeA' in AP_stream:
        ks3_path = ks3_path / 'ProbeA' / 'sorter_output' / 'spike_times.npy'
    elif 'ProbeB' in AP_stream:
        ks3_path = ks3_path / 'ProbeB' / 'sorter_output' / 'spike_times.npy'
    else:
        raise Exception(f'There is an issue with the stream name: {AP_stream}')

  
    rsync = create_ephys_rsync(str(pycontrol_path), sync_path, tstart)
    if rsync and ks3_path.exists():

        spike_times = np.load(ks3_path)
        # print(np.nanmin(spike_times), np.nanmax(spike_times))

        # Careful with the forced 30 value, sampling rate slightly diverging from 30kHz
        # should be dealt-with by open-ephys/ks3, but need to be checked
        spike_times_converted = spike_times/(rec_properties.sample_rate.iloc[idx_rec]/1000) # /1000 to be in ms
        synced_spike_times = rsync.B_to_A(spike_times_converted)
        # print(np.nanmin(synced_spike_times), np.nanmax(synced_spike_times))

        np.save(ks3_path.parent / 'rsync_corrected_spike_times.npy', synced_spike_times)
        
        # Indicate that the clusters timestamps were succesfully synced
        rec_properties.loc['synced', idx_rec] = True

    elif not ks3_path.exists():
        if verbose:
            print(f'{str(Path(*ks3_path.parts[-8:]))} \n' +
                   'does not exist, syncing impossible because sorting has not been done/possible')
        # Set by default but just in case..
        rec_properties.loc[idx_rec, 'synced'] = False 
    else:
        # Set by default but just in case..
        rec_properties.loc[idx_rec, 'synced'] = False 
        if verbose:
            print(f'{pycontrol_path} and {sync_path} with tstart = {tstart}sec could not be synchronized, no corrected spike times saved')
#
#  %% Save the updated rec_properties.csv file
# disabled as it changes input files of all rules ;( need to log on table differently
# rec_properties.to_csv(rec_properties_path)
# %%
