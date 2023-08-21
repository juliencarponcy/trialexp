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
from workflow.scripts import settings
from trialexp.process.ephys.utils import cellmat2xarray, denest_string_cell, session_and_probe_specific_uid, cellmat2dataframe

xr.set_options(display_expand_attrs=False) # attrs is too long
#%% Load inputs
cell_metrics_processing_done_path = str(Path(settings.debug_folder) / 'processed' / 'kilosort3' /'cell_metrics_full.nc')

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

kilosort_path = Path(sinput.kilosort_path)
# %% Get the path of CellExplorer outputs

xr_metrics_all = []
for probe_folder in kilosort_path.glob('Probe*'):
    probe_name = probe_folder.stem

    mat_files = list(probe_folder.glob('*.mat'))

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
    dataset.attrs['probe_name'] = probe_folder.stem

    # Drop useless or badly automatically filled column for DataFrame cleaning

    # subject_ID replace animal columns as it is random bad auto extraction from cellExplorer, deleting animal
    dataset=dataset.drop_vars(['animal'])
    
    dataset = dataset.expand_dims({'probe_name':[probe_folder.stem]})

    xr_metrics_all.append(dataset)

    # # Get name and shapes of variables

    # spikes_var_names = np.array(spikes.dtype.names)
    # cell_var_names = np.array(cell_metrics.dtype.names)
    # nb_clusters = max(cell_metrics['UID'][0][0][0])

    # spikes_var_shapes = [spikes[spikes_var_name][0][0][0].shape for spikes_var_name in spikes_var_names]
    # cell_metrics_shapes = [cell_metrics[cell_var_name][0][0][0].shape for cell_var_name in cell_var_names]

    # # A lot of reformatting is needed to get from matlab structure to tabular DataFrame
    # # Here we only take varables which have a single value by cluster

    # uni_dim_vars_idx = [idx for idx, shape in enumerate(cell_metrics_shapes) if shape == (nb_clusters,)]
    # cell_metrics_df = pd.DataFrame(index = cell_metrics['UID'][0][0][0], columns = cell_var_names[uni_dim_vars_idx])
    
    
    # for col in cell_var_names[uni_dim_vars_idx]:

    #     cell_metrics_df[col] = cell_metrics[col][0][0][0]

    #     # correct for string columns which are in 1 element arrays (e.g. labels)
    #     if type(cell_metrics[col][0][0][0][0]) == np.ndarray:
    #         cell_metrics_df[col] = cell_metrics_df[col].apply(denest_string_cell)


    # # Adding peakVoltage_expFitLengthConstant as a cell metric
    # cell_metrics_df['peakVoltage_expFitLengthConstant'] = spikes['peakVoltage_expFitLengthConstant'][0][0][0]

    # # Computing max of std of waveforms and std of the std of waveforms as indicators of possible artifacts
    # max_std_wf = [np.nanmax(std_wf.squeeze()) for std_wf in spikes['rawWaveform_std'][0][0][0]]
    # std_std_wf = [np.nanstd((std.squeeze())) for std in spikes['rawWaveform_std'][0][0][0]]

    # cell_metrics_df['max_std_waveform'] = max_std_wf
    # cell_metrics_df['std_std_waveform'] = std_std_wf

    # # Adding computation about the average peak amplitude computed across all channels,
    # # which might help for artifact detection
    # peak_average_all_wf = [max(abs(np.nanmean(rawWaveform_all,0))) for rawWaveform_all in spikes['rawWaveform_all'][0][0][0]]
    # cell_metrics_df['peak_average_all_wf'] = peak_average_all_wf


xr_metrics_all = xr.merge(xr_metrics_all)
xr_metrics_all.to_netcdf(soutput.cell_matrics_full, engine='h5netcdf',invalid_netcdf=True) 
