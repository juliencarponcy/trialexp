'''
Script to explore cell metrics from CellExplorer to find artifacts
'''
#%%
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from scipy.io import loadmat

# is this necessary?
from dotenv import load_dotenv

load_dotenv()

from snakehelper.SnakeIOHelper import getSnake
from workflow.scripts import settings


#%% Load inputs
cell_metrics_processing_done_path = str(Path(settings.debug_folder) / 'processed' / 'cell_metrics_processing.done')

(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
 [cell_metrics_processing_done_path], 'cell_metrics_processing')


# %% Define path of processed probes

verbose = True
sorter_name = 'kilosort3'
ks3_path = Path(sinput.ephys_sync_complete).parent / sorter_name
rec_properties_path = Path(sinput.rec_properties)

session_ID = ks3_path.parent.parent.stem

processed_folder = rec_properties_path.parent.parent / 'processed' 
# Only select probe folders where the results of the sorting can be found.
probe_folders = [processed_folder / sorter_name / probe_folder / 'sorter_output'
                  for probe_folder in os.listdir(processed_folder / sorter_name)
                  if 'spike_clusters.npy' in os.listdir(processed_folder / sorter_name / probe_folder / 'sorter_output')]


# %% little helper to make IDs of clusters session and probe specific
def session_and_probe_specific_uid(session_ID: str, probe_name: str, uid: int):
    return session_ID + '_' + probe_name + '_' + str(uid)


# %% Get the path of CellExplorer outputs
for probe_folder in probe_folders:
    probe_name = probe_folder.parent.stem

    mat_files = list(probe_folder.glob('*.mat'))

    cell_metrics_path = [mat_file for mat_file in mat_files if ('cell_metrics.cellinfo' in str(mat_file))][0]
    spikes_path = [mat_file for mat_file in mat_files if ('spikes.cellinfo' in str(mat_file))][0]

    # Load .mat files of CellExplorer outputs
    cell_metrics_dict = loadmat(cell_metrics_path)
    cell_metrics = cell_metrics_dict['cell_metrics']
    spikes_dict = loadmat(spikes_path)
    spikes = spikes_dict['spikes']

    # Get name and shapes of variables

    spikes_var_names = np.array(spikes.dtype.names)
    cell_var_names = np.array(cell_metrics.dtype.names)
    nb_clusters = max(cell_metrics['UID'][0][0][0])

    spikes_var_shapes = [spikes[spikes_var_name][0][0][0].shape for spikes_var_name in spikes_var_names]
    cell_metrics_shapes = [cell_metrics[cell_var_name][0][0][0].shape for cell_var_name in cell_var_names]

    # A lot of reformatting is needed to get from matlab structure to tabular DataFrame
    # Here we only take varables which have a single value by cluster

    uni_dim_vars_idx = [idx for idx, shape in enumerate(cell_metrics_shapes) if shape == (nb_clusters,)]
    cell_metrics_df = pd.DataFrame(index = cell_metrics['UID'][0][0][0], columns = cell_var_names[uni_dim_vars_idx])
    
    def denest_string_cell(cell):
        if len(cell) == 0: 
            return 'ND'
        else:
            return str(cell[0])
    
    for col in cell_var_names[uni_dim_vars_idx]:

        cell_metrics_df[col] = cell_metrics[col][0][0][0]

        # correct for string columns which are in 1 element arrays (e.g. labels)
        if type(cell_metrics[col][0][0][0][0]) == np.ndarray:
            cell_metrics_df[col] = cell_metrics_df[col].apply(denest_string_cell)


    # Adding peakVoltage_expFitLengthConstant as a cell metric
    cell_metrics_df['peakVoltage_expFitLengthConstant'] = spikes['peakVoltage_expFitLengthConstant'][0][0][0]

    # Computing max of std of waveforms and std of the std of waveforms as indicators of possible artifacts
    max_std_wf = [np.nanmax(std_wf.squeeze()) for std_wf in spikes['rawWaveform_std'][0][0][0]]
    std_std_wf = [np.nanstd((std.squeeze())) for std in spikes['rawWaveform_std'][0][0][0]]

    cell_metrics_df['max_std_waveform'] = max_std_wf
    cell_metrics_df['std_std_waveform'] = std_std_wf

    # Adding computation about the average peak amplitude computed across all channels,
    # which might help for artifact detection
    peak_average_all_wf = [max(abs(np.nanmean(rawWaveform_all,0))) for rawWaveform_all in spikes['rawWaveform_all'][0][0][0]]
    cell_metrics_df['peak_average_all_wf'] = peak_average_all_wf


    # Prepare the DataFrame so it can be aggregated with other animals, sessions, or probes
    cell_metrics_df['subject_ID'] = session_ID.split('-')[0]
    
    # Drop useless or badly automatically filled column for DataFrame cleaning

    # subject_ID replace animal columns as it is random bad auto extraction from cellExplorer, deleting animal
    cell_metrics_df.drop(columns='animal', inplace=True)
    
    cell_metrics_df['datetime'] = datetime.strptime(session_ID.split('-', maxsplit=1)[1],'%Y-%m-%d-%H%M%S')
    cell_metrics_df['task_folder'] = rec_properties_path.parent.parent.parent.stem
    cell_metrics_df['probe_name'] = probe_folder.parent.stem

    # Turn UID into real UID with session name and date and cellID and set as index
    
    cell_metrics_df['UID'] = cell_metrics_df['UID'].apply(lambda x: session_and_probe_specific_uid(session_ID, probe_name, x))
    cell_metrics_df.set_index(keys = 'UID', drop = True, inplace = True)

    # Define variables meaningless for clustering
    invalid_cols_for_clustering = ['UID','animal', 'brainRegion','cellID', 'cluID',
                                    'electrodeGroup', 'labels', 'maxWaveformCh', 'maxWaveformCh1', 
                                    'maxWaveformChannelOrder', 'putativeCellType', 'sessionName', 
                                    'shankID', 'synapticConnectionsIn', 'synapticConnectionsOut', 
                                    'synapticEffect', 'thetaModulationIndex', 'total', 'trilat_x', 
                                    'trilat_y', 'deepSuperficial', 'deepSuperficialDistance',
                                    'subject_ID',	'datetime']

    clustering_cols = [col for col in cell_metrics_df.columns if col not in invalid_cols_for_clustering]

    # Save a dataframe with partial information specifically for later clustering
    cell_metrics_df[clustering_cols].to_pickle(probe_folder / 'cell_metrics_df_clustering.pkl')
    
    # Save a full version of the cell-metrics dataframe  
    cell_metrics_df.to_pickle(probe_folder / 'cell_metrics_df_full.pkl')

    # Storing raw waveforms of all channels, dim  N channels x M timestamps x L clusters
    all_raw_wf = np.ndarray((spikes['rawWaveform_all'][0][0][0][0].shape[0], spikes['rawWaveform_all'][0][0][0][0].shape[1], spikes['rawWaveform_all'][0][0][0].shape[0]))
    for clu_idx, rawWaveforms in enumerate(spikes['rawWaveform_all'][0][0][0]):
        all_raw_wf[:,:, clu_idx] = rawWaveforms
    
    np.save(probe_folder / 'all_raw_waveforms.npy', all_raw_wf)
# %%
