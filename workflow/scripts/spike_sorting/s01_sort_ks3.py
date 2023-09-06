'''
Script to compute cell metrics by CellExplorer from Kilosort3 results
'''
#%%
import os
import warnings

import shutil

from pathlib import Path
import numpy as np

import pandas as pd

from snakehelper.SnakeIOHelper import getSnake
from workflow.scripts import settings
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
from spikeinterface.core import select_segment_recording

from workflow.scripts import settings

#%% Load inputs
spike_sorting_done_path = str(Path(settings.debug_folder) / 'processed' / 'spike_sorting.done')
# print(spike_sorting_done_path)
(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
 [spike_sorting_done_path], 'spike_sorting')

# %%

sorter_name = 'kilosort3'
verbose = True
rec_properties_path = Path(sinput.rec_properties)
session_path = rec_properties_path.parents[1]/'processed'
session_id = rec_properties_path.parents[1].stem
rec_properties = pd.read_csv(rec_properties_path, index_col = 0)
rec_properties['sorting_error'] = False
# Only select longest syncable recordings to sort
idx_to_sort = rec_properties[(rec_properties.syncable == True) & (rec_properties.longest==True)].index.values

root_data_path = os.environ['SORTING_ROOT_DATA_PATH']
temp_sorter_folder = Path(os.environ['TEMP_DATA_PATH']) /session_id
output_si_sorted_folder = Path(soutput.si_output_folder)

# %%
for idx_rec in idx_to_sort:
    exp_nb = rec_properties.exp_nb.iloc[idx_rec]
    rec_nb = rec_properties.rec_nb.iloc[idx_rec]
    AP_stream = rec_properties.AP_stream.iloc[idx_rec]
    duration = rec_properties.duration.iloc[idx_rec]
    
    # symplifying folder names for each probe
    if 'ProbeA' in AP_stream:    
        probe_name = 'ProbeA'
    elif 'ProbeB' in AP_stream:
        probe_name = 'ProbeB'
    else:
        raise ValueError(f'invalid probe name rec: {rec_properties_path.parent}')

    # Define outputs folder, specific for each probe and sorter
    # output_sorter_specific_folder = sorter_specific_folder / sorter_name / probe_name
    temp_output_sorter_specific_folder = temp_sorter_folder / sorter_name / probe_name

    ephys_path = Path(rec_properties.full_path.iloc[idx_rec]).parents[4]
    
    # Maybe not the best method to get it
    # has introduced some bugs for forgotten reason related to folder changes
    # TODO improve to join just before relative_ephys_path and root_data_path overlap
    relative_ephys_path = os.path.join(*ephys_path.parts[5:])
    ephys_path = os.path.join(root_data_path, relative_ephys_path)
    
    experiments_nb = rec_properties.exp_nb.unique()
    
    '''
    Neo doesn't care about the actual name of the experiment and recording folder, it just sorts them and assign them a block index and
    segment index sequentially. Therefore, the experiment number from the open ephys data structure may not have direct 
    corresondance with the block and segment index.
    In case where all the experiments are recordings are present, it doesn't matter. But in case where some experiments or recordings
    are delete. It will lead to errors.
    '''
    
    # The indices of experiments seems to differ depending on whether they are a single or multiple experiments
    if len(experiments_nb) == 1:
        recordings = se.read_openephys(ephys_path, block_index=exp_nb-1, stream_name=AP_stream) # nb-based
    else:
        recordings = se.read_openephys(ephys_path, block_index=exp_nb, stream_name=AP_stream) # nb-based
    
    # Segment indices are not the same as the recording number, they are based on the number of recording folders
    # e.g. recording1,recording3,recording4 -> segment indices will always be 0,1,2
    # Will lead to issue if some recording folders are omitted
    # find the correct segment for the recording
    rec_sorted = rec_properties.sort_values('rec_nb')
    rec_sorted = rec_sorted[rec_sorted.AP_stream==AP_stream]
    segment_no = np.where(rec_sorted.rec_nb == rec_nb)[0]
    
    recording = recordings.select_segments(segment_no)
    
    # make sure we have selected the correct recording
    assert int(recording.get_total_duration()) == int(duration), 'Error: recording durations do not match!'
    
    if verbose:
        print(f'{Path(ephys_path).parts[-1]}, {probe_name}, exp_nb:{exp_nb}, rec_nb:{rec_nb}. recording duration: {recording.get_total_duration()}s')   

    sorter_specific_params = {
        'n_jobs': 32, 
        # 'total_memory': 512000000000, 
        # 'chunk_size': None, 
        # 'chunk_memory': 12800000000,
        'chunk_duration': '10s', 
        'progress_bar': False}
    
    # TODO: Add try / catch with warnings and logging of the failed sorting (in rec_properties.csv for instance)
    # In order to cleanly skip dodgy recordings and keep the pipeline running
    # try:
#     sorting = ss.run_sorter(
#             sorter_name = sorter_name,
#             recording = recording, 
#             output_folder = temp_output_sorter_specific_folder,
#             remove_existing_folder = True, 
#             delete_output_folder = False, 
#             verbose = True,
#             **sorter_specific_params)
    
# #     # delete previous output_sorting_folder and its contents if it exists,
# #     # this prevent the save method to crash.
    
#     # clear any existing folder content
#     # if output_si_sorted_folder.exists():
#     #     shutil.rmtree(output_si_sorted_folder)
        
#     # output_si_sorted_folder.parent.mkdir(parents=True, exist_ok=True) #make sure the parent directory exist
        
#     sorting.save(folder =  output_si_sorted_folder/probe_name) # very small, can save directly
    
# #     # also save the rec_properties for this particular recording
#     # record_path = session_path/'kilosort'/probe_name
#     # if not record_path.exists():
#     #     record_path.mkdir(parents=True)
        
#     rec2save = rec_properties.iloc[[idx_rec]].copy()
#     rec2save['segment_no'] = segment_no
#     rec2save.to_csv(temp_output_sorter_specific_folder/'sorter_output'/'rec_prop.csv', index=False) #also save the recording property


# %%
