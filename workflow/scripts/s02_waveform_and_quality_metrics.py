'''
Script to extract waveforms and quality metrics

methods from:
https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quality_metrics.html
https://spikeinterface.readthedocs.io/en/latest/modules/qualitymetrics.html#quality-metrics-module

from: https://raw.githubusercontent.com/SpikeInterface/spikeinterface/main/doc/modules/qualitymetrics.rst

Quality Metrics module
======================

Quality metrics allows one to quantitatively assess the *goodness* of a spike sorting output.
The :py:mod:`~spikeinterface.qualitymetrics` module includes functions to compute a large variety of available metrics.
All of the metrics currently implemented in spikeInterface are *per unit* (pairwise metrics do appear in the literature).

Each metric aims to identify some quality of the unit.
Contamination metrics (also sometimes called 'false positive' or 'type I' metrics) aim to identify the amount of noise present in the unit.
Examples include: ISI violations, sliding refractory period violations, SNR, NN-hit rate.
Completeness metrics (or 'false negative'/'type II' metrics) aim to identify whether any of the spiking activity is missing from a unit.
Examples include: presence ratio, amplitude cutoff, NN-miss rate.
Drift metrics aim to identify changes in waveforms which occur when spike sorters fail to successfully track neurons in the case of electrode drift.

Some metrics make use of principal component analysis (PCA) to reduce the dimensionality of computations.
Various approaches to computing the principal components are possible, and choice should be carefully considered in relation to the recording equipment used.
The following metrics make use of PCA: isolation distance, L-ratio, D-prime, Silhouette score and NN-metrics.
By contrast, the following metrics are based on spike times only: firing rate, ISI violations, presence ratio.
And amplitude cutoff and SNR are based on spike times as well as waveforms.

For more details about each metric and it's availability and use within SpikeInterface, see the individual pages for each metrics.
'''
#%%
import os
from pathlib import Path

import pandas as pd
import numpy as np

import spikeinterface.full as si
import spikeinterface.extractors as se

from spikeinterface.core import select_segment_recording
from spikeinterface.postprocessing import compute_principal_components
from spikeinterface.qualitymetrics import get_quality_metric_list, compute_quality_metrics, get_default_qm_params
from spikeinterface.curation import remove_excess_spikes
from snakehelper.SnakeIOHelper import getSnake


from workflow.scripts import settings
from trialexp.process.ephys.utils import denest_string_cell, session_and_probe_specific_uid

#%% Load inputs


(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [settings.debug_folder + r'/processed/si_quality.done'],
  'waveform_and_quality_metrics')


# %% Load Metadata and folders

sorter_name = 'kilosort3'
verbose = True

rec_properties_path = Path(sinput.rec_properties)
processed_folder = rec_properties_path.parent.parent / 'processed' 

# Folder where to store outputs of waveforms info
waveform_results_folder = rec_properties_path.parent.parent / 'processed' / 'waveforms'
if not waveform_results_folder.exists():
    waveform_results_folder.mkdir()

probe_folders = [processed_folder / sorter_name / probe_folder / 'sorter_output'
                  for probe_folder in os.listdir(processed_folder / sorter_name)
                  if 'spike_clusters.npy' in os.listdir(processed_folder / sorter_name / probe_folder / 'sorter_output')]

rec_properties = pd.read_csv(rec_properties_path, header=0, index_col=0)
# Filter only longest syncable files
rec_properties = rec_properties[(rec_properties['longest'] == True)]

experiments_nb = rec_properties.exp_nb.unique()


session_ID = rec_properties_path.parent.parent.stem
#%% Process SpikeInterface metrics

# will hold all probes cluster locations
session_si_df = pd.DataFrame()
# will hold all probes cluster metrics
session_si_metrics_df = pd.DataFrame()

for probe_folder in probe_folders:
    probe_name = probe_folder.parent.stem

        # Part dealing with extracting info from SpikeInterface (cluster location and else)
    AP_stream = [AP_stream for AP_stream in rec_properties.AP_stream if probe_name in AP_stream]
    if len(AP_stream) > 1:
        raise Exception('More than one Action Potential stream correspond to the current probe') 
    else:
        AP_stream = AP_stream[0]

    rec_path = rec_properties[rec_properties.AP_stream == AP_stream].full_path.values[0]

    rec_path = Path(rec_path).parent.parent.parent.parent.parent

    sorting = si.read_sorter_folder(probe_folder.parent)
    exp_nb = rec_properties.exp_nb.values[0]
    rec_nb = int(rec_properties.rec_nb.values[0])

    if len(experiments_nb) == 1:
        recordings = se.read_openephys(rec_path, block_index=exp_nb-1, stream_name=AP_stream) # nb-based
    else:
        recordings = se.read_openephys(rec_path, block_index=exp_nb, stream_name=AP_stream) # nb-based

    recording = select_segment_recording(recordings, segment_indices= rec_nb-1) # index-based
    
    # recording = remove_excess_spikes(recording)
    recording.annotate(is_filtered=True)
    
    
    waveform_folder = Path(probe_folder) / 'waveforms'

    if verbose:
        print(f'Now extracting waveforms for {session_ID}, {probe_name}')

    # Waveform extraction
    we = si.extract_waveforms(recording, 
                            sorting, 
                            folder=waveform_folder,
                            overwrite=True,
                            sparse=True, 
                            max_spikes_per_unit=100, 
                            ms_before=1,
                            ms_after=3,
                            allow_unfiltered=True,
                            chunk_duration = '10s'
                            )
    # Quality Metrics


    # pca = compute_principal_components(we, n_components=5, mode='by_channel_local')
    # pca.fit()
    # all_projections, all_labels = pca.get_all_projections()

    # run for all spikes in the SortingExtractor
    # pca.run_for_all_spikes(file_path= waveform_results_folder / f'all_pca_projections_{probe_name}.npy')
    metric_names =  get_quality_metric_list()
    # issue with computing sliding_rp_violation so removing it
    metric_names = [metric_name for metric_name in metric_names if (metric_name != 'sliding_rp_violation')]
    # probe_metrics_df = pd.DataFrame()
    metrics = compute_quality_metrics(we, 
        progress_bar = True,                               
        metric_names = metric_names)

        
    si_noise_levels = si.compute_noise_levels(we)
    np.save(waveform_results_folder / f'chan_noise_levels_{probe_name}.npy', si_noise_levels)

    si_correlograms, correlo_bins = si.compute_correlograms(we)
    np.save(waveform_results_folder / f'cluster_correlograms_{probe_name}.npy', si_correlograms)
    np.save(waveform_results_folder / f'correlo_bins.npy', correlo_bins)

    si_template_similarities = si.compute_template_similarity(we)
    np.save(waveform_results_folder / f'cluster_template_similarities_{probe_name}.npy', si_template_similarities)

    si_locations = si.compute_unit_locations(we)

    probe_si_df = pd.DataFrame(data=si_locations , columns=['x','y'])
    probe_si_df['session_ID'] = session_ID
    probe_si_df['probe_name'] = probe_name
    probe_si_df['cluster_id'] = probe_si_df.index.values
    probe_si_df['UID'] = probe_si_df['cluster_id'].apply(lambda i: session_and_probe_specific_uid(session_ID = session_ID, probe_name = probe_name, uid = i))

    # aggregate over all probes metrics from CellExplorer and SpikeInterface
    # SpikeInterface
    session_si_df = pd.concat([session_si_df, probe_si_df], axis=0)
    session_si_metrics_df = pd.concat([session_si_metrics_df, metrics], axis=0)



# %% Merge channel locations and SpikeInterface quality metrics
si_metadata = pd.concat([session_si_df, session_si_metrics_df], axis=1)
# Save results
si_metadata.to_pickle(waveform_results_folder / 'si_metrics_df.pkl' )
# %%
