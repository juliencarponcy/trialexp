'''
Script to create the session folder structure
'''
#%%
import os
import warnings

from pathlib import Path

import pandas as pd

from snakehelper.SnakeIOHelper import getSnake
from workflows.scripts import settings
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
from spikeinterface.core import select_segment_recording


from trialexp.process.ephys.spikesort import sort


#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflows/spikesort.smk',
  [settings.debug_folder + 'spike_templates.npy'],
  'spikesort')


# %%
session_path = '/home/MRC.OX.AC.UK/phar0732/ettin/Data/head-fixed/by_sessions/reaching_go_spout_bar_nov22/kms058-2023-03-25-184034'
# rec_path = 
sorter_name = 'kilosort3'
rec_properties_path = Path(session_path) / 'ephys' / 'rec_properties.csv'
rec_properties = pd.read_csv(rec_properties_path, index_col= None)

output_folder = rec_properties_path.parent / 'sorting'

# Only select longest syncable recordings to sort
idx_to_sort = rec_properties[rec_properties.longest == True].index.values


# %%
for idx_rec in idx_to_sort:
    exp_nb = rec_properties.exp_nb.iloc[idx_rec]
    rec_nb = rec_properties.rec_nb.iloc[idx_rec]
    AP_stream = rec_properties.AP_stream.iloc[idx_rec]
    ephys_path = Path(rec_properties.full_path.iloc[idx_rec]).parent.parent.parent.parent.parent

    recordings = se.read_openephys(ephys_path, block_index=exp_nb-1, stream_name=AP_stream)
    recording = select_segment_recording(recordings, segment_indices= int(rec_nb-1))

    sorting = ss.run_sorter(
            sorter_name = sorter_name,
            recording = recording, 
            output_folder = output_folder / AP_stream,
            remove_existing_folder = True, 
            delete_output_folder = False, 
            verbose = True)

# %%
