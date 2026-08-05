'''
Script to compute cell metrics by CellExplorer from Kilosort3 results
'''
#%%
import os

import shutil

from pathlib import Path
import numpy as np

import pandas as pd

from snakehelper.SnakeIOHelper import getSnake
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
from spikeinterface.core import select_segment_recording
from kilosort import run_kilosort
from trialexp import config
import torch
from trialexp.process.ephys.artifact_removal import filter_artifact_sensor
from kilosort.io import BinaryFiltered
from open_ephys.analysis import Session
from scipy import signal
import matplotlib.pylab as plt
import numpy as np 
from loguru import logger
from trialexp.process.ephys.artifact_removal import plot_spectrogram

#%% Load inputs
spike_sorting_done_path = str(Path(config.debug_folder) / 'processed' / 'spike_sorting.done')
(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
 [spike_sorting_done_path], 'spike_sorting')

# %%

sorter_name = 'kilosort4'
verbose = True
skip_sorted_session = True
rec_properties_path = Path(sinput.rec_properties)
session_path = rec_properties_path.parents[1]/'processed'
session_id = rec_properties_path.parents[1].stem
rec_properties = pd.read_csv(rec_properties_path, index_col = 0)
rec_properties['sorting_error'] = False
# Only select longest syncable recordings to sort
idx_to_sort = rec_properties[(rec_properties.syncable == True) & (rec_properties.longest==True)].index.values

root_data_path = config.SORTING_ROOT_DATA_PATH
temp_sorter_folder = Path(config.TEMP_DATA_PATH) /session_id
output_si_sorted_folder = Path(soutput.si_output_folder)

#%% plot spectrogram to check for artifacts

# if skip_sorted_session and not Path(soutput.spectrogram).exists():
#     fig,ax =plot_spectrogram(rec_properties.iloc[idx_to_sort[0]], 30000, freq_pool_ratio=100);
#     fig.suptitle(session_id)
#     fig.savefig(soutput.spectrogram, dpi=300)
# %%
for idx_rec in idx_to_sort:

    AP_stream = rec_properties.AP_stream.iloc[idx_rec]
    recording_path = rec_properties.full_path[idx_rec]
    
    # symplifying folder names for each probe
    if 'ProbeA' in AP_stream:    
        probe_name = 'ProbeA'
    elif 'ProbeB' in AP_stream:
        probe_name = 'ProbeB'
    else:
        raise ValueError(f'invalid probe name rec: {rec_properties_path.parent}')

    
    # Maybe not the best method to get it
    # has introduced some bugs for forgotten reason related to folder changes
    # # TODO improve to join just before relative_ephys_path and root_data_path overlap
    ephys_path = Path(rec_properties.full_path.iloc[idx_rec]).parts[-10:]
    
    recording_path = os.path.join(root_data_path, *ephys_path)
    
    if not (output_si_sorted_folder/probe_name).exists():
        (output_si_sorted_folder/probe_name).mkdir()
    else:
        # skip if the sorting has already been done
        # used when we need to manually sort a certain session
        if skip_sorted_session:
            logger.info(f'Skipping {session_id} as it has already been sorted')
            continue
    
    # use kilosort4 directly
    device = torch.device('cuda:0')
    settings = {'data_dir': recording_path, 
                'n_chan_bin': 384, 
                'batch_size' : 30000*4, # 8*Fs for speeding up, for bad session, use a smaller batch_size
                # 'tmax':900, #for bad session, set the tmax manually
                'results_dir': output_si_sorted_folder/probe_name}
    
    #####
    # artifact removal. Only activate this on bad session
    # BinaryFiltered.filter = filter_artifact_sensor 
    
    run_kilosort(settings=settings, 
                 save_extra_vars=True,
                probe_name='neuropixPhase3B1_kilosortChanMap.mat', device=device)
    
        
    rec2save = rec_properties.iloc[[idx_rec]].copy()
    # cannot save in the probe folder otherwise spikeinterface will complain
    rec2save.to_csv(output_si_sorted_folder/f'rec_prop_{probe_name}.csv', index=False) #also save the recording property



# free GPU memory
torch.cuda.empty_cache()
Path(soutput.sorting_complete).touch() # for use during manual pipeline run

# %% Manual clean up of unsorted sessions
# base_search_folder = os.path.join(config.SESSION_ROOT_DIR, 
#                             '2024_August_cohort',
#                             'by_sessions',
#                             'reaching_go_spout_bar_VR_April24')

# # search to see if processed/kilosort4/ProbeA exists in the session folder
# session_folders = Path(base_search_folder).glob('*')
# for s in session_folders:
#     if (s/'processed'/'kilosort4').exists() and not (s/'processed'/'kilosort4'/'ProbeA').exists():
#         print(f'{s.stem} is not yet sorted. cleaning up')
#         if (s/'processed'/'spike_sorting.done').exists():
#             # remove the spike_sorting.done file
#             (s/'processed'/'spike_sorting.done').unlink()
        
#         if (s/'processed'/'spike_workflow.done').exists():
#             (s/'processed'/'spike_workflow.done').unlink()
        
# %%
