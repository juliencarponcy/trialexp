'''
Script to create the session folder structure
'''
#%%
import os
from pathlib import Path

from neo.rawio.openephysbinaryrawio import explore_folder

#%% 
ephys_base_path = Path('/home/MRC.OX.AC.UK/phar0732/ettin/Data/head-fixed/_Other/test_folder_ephys')
sorted_base_path = Path('/home/MRC.OX.AC.UK/phar0732/ettin/Data/head-fixed/_Other/test_folder_ephys/sorted')

ephys_folders = os.listdir(ephys_base_path)


#%% 

# Loop for all open_ephys base folders
for ephys_folder in ephys_folders:
    
    # Explore folder with neo utilities for openephys
    folder_structure, all_streams, nb_block, nb_segment_per_block,\
    possible_experiment_names = explore_folder(ephys_base_path / ephys_folder)
    
    print(f'Nb of Experiments (blocks): {nb_block}\nNb of segments per block: {nb_segment_per_block}\nDefault exp name: {possible_experiment_names}')

    # List continuous streams names
    continuous_streams = list(folder_structure['Record Node 101']['experiments'][1]['recordings'][1]['streams']['continuous'].keys())
    # Only select action potentials streams
    AP_streams = [AP_stream for AP_stream in continuous_streams if 'AP' in AP_stream]
    print(f'Spike streams:{AP_streams}\n')
    for rec_nb in range(1,nb_segment_per_block[0]+1):
        for AP_stream in AP_streams:
            target_sorted_folder = sorted_base_path / ephys_folder / ('rec' + str(rec_nb)) / AP_stream
            if not target_sorted_folder.exists():
                target_sorted_folder.mkdir(parents=True)

   
# %%
