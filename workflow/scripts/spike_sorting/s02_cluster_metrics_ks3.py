'''
Script to create the session folder structure
'''
#%%
import os
from pathlib import Path

import matlab.engine
from snakehelper.SnakeIOHelper import getSnake
from trialexp.process.ephys.utils import prepare_mathlab_path

from workflow.scripts import settings

#%% Load inputs


(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [settings.debug_folder + r'/processed/spike_metrics.done'],
  'spike_metrics_ks3')


# %% Load Metadata and folders

sorter_name = 'kilosort3'
verbose = True

sorter_specific_path = Path(sinput.rec_properties).parent.parent / 'processed' / sorter_name

probe_folders = [str(sorter_specific_path / probe_folder) for probe_folder in os.listdir(sorter_specific_path)]

# %% Start Matlab engine and add paths
eng = matlab.engine.start_matlab()


prepare_mathlab_path(eng)
# %% Process CellExplorer Cell metrics
# TODO do this analyze on the temp folder to improve speed
for probe_folder in probe_folders:

    cell_exp_session = eng.sessionTemplate_nxp(probe_folder, 'showGUI', False)
    
    ### Important:
    # The following defaults have been modified in CellExplorer base code
    # sessionTemplate.m
    # to match NeuroPixels defaults, as setting them afterwards induced bugs
    
    # adjusting wrong default params
    # sampling rate
    # cell_exp_session['extracellular']['sr'] = matlab.double([30000])
    # nb of channels: WARNING: This change of parameter introduced an error in Matlab
    # cell_exp_session['extracellular']['nChannels'] = matlab.double([384])

    # Process Cell Metrics
    cell_metrics = eng.ProcessCellMetrics('session', cell_exp_session, \
        'showGUI', False, 'showWaveforms', False, 'showFigures', False, \
        'manualAdjustMonoSyn', False, 'forceReloadSpikes', True, 'forceReload', True, \
        'summaryFigures', False)

    # Dump cell metrics dict in pkl
    # with open(probe_folder / 'cell_metrics.pkl', 'wb') as handle: 
    #     pickle.dump(cell_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

  
# %% Stop Matlab engine
eng.quit()
# %%
