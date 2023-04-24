'''
Script to create the session folder structure
'''
#%%
import os
from pathlib import Path

import matlab.engine
from snakehelper.SnakeIOHelper import getSnake

from workflows.scripts import settings

#%% Load inputs


(sinput, soutput) = getSnake(locals(), 'workflows/spikesort.smk',
  [settings.debug_folder + r'/processed/spike_metrics.done'],
  'spike_metrics_ks3')
    

# Delete temporary copies of the recording and whitening matrix
    rec_copy_path = temp_output_sorter_specific_folder / 'sorter_output' / 'recording.dat'
    whitening_mat_path = temp_output_sorter_specific_folder / 'sorter_output' / 'temp_wh.dat'
    rec_copy_path.unlink()
    whitening_mat_path.unlink()

    # copy content of the ks3 folder to the server
    files_to_copy = rec_copy_path.parent.glob('*')
    for file_to_copy in files_to_copy:
        shutil.copy(file_to_copy, output_sorter_specific_folder/ 'sorter_output' /  file_to_copy.name)
    
    # delete all local files 
    shutil.rmtree(temp_output_sorter_specific_folder.parent)