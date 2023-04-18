'''
Script to create the session folder structure
'''
#%%
import os
import warnings

import shutil
import pickle

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

sorter_specific_path = Path(sinput.sorter_specific_folder) / sorter_name

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

    cell_exp_session = eng.sessionTemplate(probe_folders[0], 'showGUI', False)
    
    # adjusting wrong default params
    # sampling rate
    cell_exp_session['extracellular']['sr'] = 30000
    # nb of channels
    cell_exp_session['extracellular']['nChannels'] = 384

    # Process Cell Metrics
    cell_metrics = eng.ProcessCellMetrics('session', cell_exp_session, \
        'showGUI', False, 'showWaveforms', False, 'showFigures', False, \
        'manualAdjustMonoSyn', False)

    # Dump cell metrics dict in pkl
    # with open(probe_folder / 'cell_metrics.pkl', 'wb') as handle:
    #     pickle.dump(cell_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)


# %%
