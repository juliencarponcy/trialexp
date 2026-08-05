'''
Script to explore cell metrics from CellExplorer and SpikeInterface

This measures are then further combined as cluster associated data
'''
#%%
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from scipy.io import loadmat
import xarray as xr
from dotenv import load_dotenv
load_dotenv()

import spikeinterface.full as si
import spikeinterface.extractors as se

from spikeinterface.core import select_segment_recording

from snakehelper.SnakeIOHelper import getSnake
from trialexp import config
from trialexp.process.ephys.utils import cellmat2xarray, denest_string_cell, session_and_probe_specific_uid, cellmat2dataframe

xr.set_options(display_expand_attrs=False) # attrs is too long
#%% Load inputs
cell_metrics_processing_done_path = str(Path(config.debug_folder) / 'processed' /'cell_metrics_full.nc')

(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
 [cell_metrics_processing_done_path], 'cell_metrics_processing')


# %% Define path of processed probes

verbose = True
sorter_name = 'kilosort3'
ks3_path = Path(sinput.ephys_sync_complete).parent / sorter_name
rec_properties_path = Path(sinput.rec_properties)
session_ID = ks3_path.parent.parent.stem
processed_folder = rec_properties_path.parent.parent / 'processed' 


rec_properties = pd.read_csv(rec_properties_path, header=0, index_col=0)

kilosort_path = Path(sinput.ephys_sync_complete).parent/'kilosort'
# %% Get the path of CellExplorer outputs

xr_metrics_all = []
probe_names = []
for probe_folder in kilosort_path.glob('Probe*'):
    probe_name = probe_folder.stem

    mat_files = list((probe_folder/'sorter_output').glob('*.mat'))

    cell_metrics_path = [mat_file for mat_file in mat_files if ('cell_metrics.cellinfo' in str(mat_file))][0]
    spikes_path = [mat_file for mat_file in mat_files if ('spikes.cellinfo' in str(mat_file))][0]

    # Load .mat files of CellExplorer outputs
    cell_metrics_dict = loadmat(cell_metrics_path, simplify_cells=True)
    cell_metrics = cell_metrics_dict['cell_metrics']
    spikes_dict = loadmat(spikes_path, simplify_cells=True)
    spikes = spikes_dict['spikes']

    # Convert CellExplorer data structure to xarray
    clusID_prefix = f'{session_ID}_{probe_name}_'
    dataset_spike = cellmat2xarray(spikes, clusID_prefix)
    dataset_metrics = cellmat2xarray(cell_metrics,clusID_prefix)
    dataset = xr.merge([dataset_metrics, dataset_spike])
    
    

    # # Prepare the DataFrame so it can be aggregated with other animals, sessions, or probes
    dataset.attrs['subject_ID'] = session_ID.split('-')[0]
    dataset.attrs['session_ID'] = session_ID
    dataset.attrs['task_folder'] = rec_properties_path.parents[2].stem

    # subject_ID replace animal columns as it is random bad auto extraction from cellExplorer, deleting animal
    dataset=dataset.drop_vars(['animal'])
    xr_metrics_all.append(dataset)
    probe_names.append(probe_name)

xr_metrics_all = xr.merge(xr_metrics_all)
xr_metrics_all.attrs['probe_names'] = probe_names
xr_metrics_all.to_netcdf(soutput.cell_matrics_full, engine='h5netcdf',invalid_netcdf=True) 

# %%
