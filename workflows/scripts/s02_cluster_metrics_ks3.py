'''
Script to create the session folder structure
'''
#%%
import os

from pathlib import Path

import pandas as pd
import matlab.engine

from snakehelper.SnakeIOHelper import getSnake

import spikeinterface.full as si

from workflows.scripts import settings

#%% Load inputs


(sinput, soutput) = getSnake(locals(), 'workflows/spikesort.smk',
  [settings.debug_folder + r'/processed/spike_metrics.done'],
  'spike_metrics_ks3')


# %% Load Metadata and folders

sorter_name = 'kilosort3'
verbose = True
rec_properties_path = Path(sinput.rec_properties)
rec_properties = pd.read_csv(rec_properties_path, index_col= None)

sorter_specific_path = Path(sinput.ks_3_spike_templates_A).parent.parent.parent

probe_folders = [str(sorter_specific_path / probe_folder / 'sorter_output') for probe_folder in os.listdir(sorter_specific_path)]

# %% Start Matlab engine and add paths
eng = matlab.engine.start_matlab()
s = eng.genpath('/home/MRC.OX.AC.UK/phar0732/Documents/GitHub/spikes')
n = eng.genpath('/home/MRC.OX.AC.UK/phar0732/Documents/GitHub/npy-matlab')
c = eng.genpath('/home/MRC.OX.AC.UK/phar0732/Documents/GitHub/CellExplorer')

eng.addpath(s, nargout=0)
eng.addpath(n, nargout=0)
eng.addpath(c, nargout=0)

# %% Process CellExplorer Cell metrics
for probe_folder in probe_folders:

    cell_exp_session = eng.sessionTemplate(probe_folder, 'showGUI', False)
    
    ### Important:
    # The following defaults have been modified in CellExplorer base code
    # to match NeuroPixels defaults, has setting them afterwards induced bugs
    
    # adjusting wrong default params
    # sampling rate
    # cell_exp_session['extracellular']['sr'] = matlab.double([30000])
    # nb of channels: WARNING: This change of parameter introduced an error in Matlab
    # cell_exp_session['extracellular']['nChannels'] = matlab.double([384])

    # Process Cell Metrics
    cell_metrics = eng.ProcessCellMetrics('session', cell_exp_session, \
        'showGUI', False, 'showWaveforms', False, 'showFigures', False, \
        'manualAdjustMonoSyn', False, 'forceReloadSpikes', True, 'forceReload', True, \
        'summaryFigures', True)

    # Dump cell metrics dict in pkl
    # with open(probe_folder / 'cell_metrics.pkl', 'wb') as handle:
    #     pickle.dump(cell_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

  
# %% Stop Matlab engine
eng.quit()
# %%
