'''
Script to create the session folder structure
'''
#%%
import os
from pathlib import Path
import shutil
import matlab.engine
from snakehelper.SnakeIOHelper import getSnake

from trialexp import config
#%% Load inputs


(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [config.debug_folder + r'/processed/spike_metrics.done'],
  'spike_metrics_ks3')


# %% Load Metadata and folders

sorter_name = 'kilosort3'
verbose = True


rec_properties_path = Path(sinput.rec_properties)
session_id = rec_properties_path.parents[1].stem


# use the temporary folder for processing with Cell Explorer
# Doing processong in local file should be much faster than over the network, 
# since Cell Explorer probably does very frequent disk I/O via memmap
sorter_specific_path = Path(config.TEMP_DATA_PATH) /session_id/ sorter_name
assert sorter_specific_path.exists(), 'Sorted data do not exist!'
probe_folders = [str(sorter_specific_path / probe_folder/'sorter_output') for probe_folder in os.listdir(sorter_specific_path)]

kilosort_path = Path(soutput.kilosort_folder)

# %% Start Matlab engine and add paths
eng = matlab.engine.start_matlab()

# importing matlabplot while the eng is active may have some library conflicts issues
s = eng.genpath(os.environ['CORTEX_LAB_SPIKES_PATH']) # maybe unnecessary, just open all ks3 results
n = eng.genpath(os.environ['NPY_MATLAB_PATH'])
c = eng.genpath(os.environ['CELL_EXPLORER_PATH'])

eng.addpath(s, nargout=0)
eng.addpath(n, nargout=0)
eng.addpath(c, nargout=0)

# %% Process CellExplorer Cell metrics

for probe_folder in probe_folders:

    #TODO: add sessionTemplate_nxp path to matlab instead of relying on copying file
    # need to copy the sessionTemplate_nxp to the cellexplorer folder first
    cell_exp_session = eng.sessionTemplate_nxp(probe_folder, 'showGUI', False)

    # Process Cell Metrics
    # loading spike is the most time-consuming step
    cell_metrics = eng.ProcessCellMetrics('session', cell_exp_session, \
        'showGUI', False, 'showWaveforms', False, 'showFigures', False, \
        'manualAdjustMonoSyn', False, 'summaryFigures', False)

    
    # copy the kilosort output and cell metrics back to the session folder
    print('I will now copy the Cell metrics and sorting output back to the session folder')
    probe_path = Path(probe_folder).parent
    shutil.copytree(probe_path, kilosort_path/probe_path.stem, dirs_exist_ok=True)
    
    print('Folders copied. I will now delete the temp folders')
    # delete the temporary folders
    shutil.rmtree(probe_path)
  
# %% Stop Matlab engine
eng.quit()

# also remove the root session path in temp
shutil.rmtree(probe_path.parents[1])
