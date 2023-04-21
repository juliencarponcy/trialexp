'''
Script to compute cell metrics by CellExplorer from Kilosort3 results
'''
#%%
import os
import warnings

import shutil

from pathlib import Path

import pandas as pd

from snakehelper.SnakeIOHelper import getSnake
from workflows.scripts import settings
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
from spikeinterface.core import select_segment_recording

from workflows.scripts import settings
from trialexp.process.ephys.spikesort import sort


#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflows/spikesort.smk',
  [settings.debug_folder + r'/processed/spike_sorting.done'],
  'spike_sorting')


# %%

sorter_name = 'kilosort3'
verbose = True
rec_properties_path = Path(sinput.rec_properties)
rec_properties = pd.read_csv(rec_properties_path, index_col= None)

temp_output_folder = rec_properties_path.parent / 'temp'

# Only select longest syncable recordings to sort
idx_to_sort = rec_properties[rec_properties.longest == True].index.values

root_data_path = os.environ['SORTING_ROOT_DATA_PATH']

si_sorted_folder = Path(sinput.rec_properties).parent / 'si_sorted'
sorter_specific_folder = Path(sinput.rec_properties).parent / 'sorter'

# %%
for idx_rec in idx_to_sort:
    exp_nb = rec_properties.exp_nb.iloc[idx_rec]
    rec_nb = rec_properties.rec_nb.iloc[idx_rec]
    AP_stream = rec_properties.AP_stream.iloc[idx_rec]
    
    # symplifying folder names for each probe
    if 'ProbeA' in AP_stream:    
        probe_name = 'ProbeA'
    elif 'ProbeB' in AP_stream:
        probe_name = 'ProbeB'
    else:
        raise ValueError(f'invalid probe name rec: {rec_properties_path.parent}')

    # Define outputs folder, specific for each probe and sorter
    output_sorter_specific_folder = sorter_specific_folder / sorter_name / probe_name
    output_si_sorted_folder = si_sorted_folder / sorter_name / probe_name

    ephys_path = Path(rec_properties.full_path.iloc[idx_rec]).parent.parent.parent.parent.parent
    relative_ephys_path = os.path.join(*ephys_path.parts[5:])
    ephys_path = os.path.join(root_data_path, relative_ephys_path)
    
    experiments_nb = rec_properties.exp_nb.unique()
    if len(experiments_nb) == 1:
        recordings = se.read_openephys(ephys_path, block_index=exp_nb-1, stream_name=AP_stream) # nb-based
    else:
        recordings = se.read_openephys(ephys_path, block_index=exp_nb, stream_name=AP_stream) # nb-based
    
    recording = select_segment_recording(recordings, segment_indices= int(rec_nb-1)) # index-based
    if verbose:
        print(f'{Path(ephys_path).parts[-1]}, {probe_name}, exp_nb:{exp_nb}, rec_nb:{rec_nb}. recording duration: {recording.get_total_duration()}s')   

    sorter_specific_params = {
        # 'n_jobs': 24, 
        # 'total_memory': 512000000000, 
        # 'chunk_size': None, 
        # 'chunk_memory': 12800000000,
        'chunk_duration': '10s', 
        'progress_bar': True}

    sorting = ss.run_sorter(
            sorter_name = sorter_name,
            recording = recording, 
            output_folder = output_sorter_specific_folder,
            remove_existing_folder = True, 
            delete_output_folder = False, 
            verbose = True,
            **sorter_specific_params)


    # delete previous output_sorting_folder and its contents if it exists,
    # this prevent the save method to crash.
    if output_si_sorted_folder.exists():
        shutil.rmtree(output_si_sorted_folder)
    
    sorting.save(folder = output_si_sorted_folder)

# %%
