from glob import glob
from pathlib import Path
import os 
from dotenv import load_dotenv

load_dotenv()

# temp attempt, to move to .env 
sorter_name = 'kilosort3'

def rec_properties_input(wildcards):
    # determine if there is an ephys recording for that folder
    recording_csv = glob(f'{wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/ephys/rec_properties.csv')
    if len(recording_csv) > 0:
        return f'{wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/ephys/rec_properties.csv'
    else:
        return []

def gather_metrics_to_aggregate(wildcards):
    # determine if there is processed cell metrics in that folder
    # cell_metrics_df_clustering = glob(f'{wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/processed/kilosort3/{wildcards.probe_folder}/sorter_output/cell_metrics_df_clustering.pkl')
    # if len(recording_csv) > 0:
    #     return f'{wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/processed/kilosort3/{wildcards.probe_folder}/sorter_output/cell_metrics_df_clustering.pkl'
    # else:
    #     return []
    ...

rule spike_all:
    input: expand('{sessions_root}/processed/spike_workflow.done', sessions_root = Path(os.environ.get('SESSION_ROOT_DIR')).glob('*/*'))

rule spike_sorting:
    input:
        rec_properties = rec_properties_input

    output:
        sorting_complete = touch('{sessions}/{task_path}/{session_id}/processed/spike_sorting.done'),       
    
    threads: 32

    script:
        "scripts/s01_sort_ks3.py"



rule move_to_server:
    input: 
        sorting_complete = '{sessions}/{task_path}/{session_id}/processed/spike_sorting.done'

    params:
        local_root_sorting_folder = Path(os.environ['TEMP_DATA_PATH'])

    output:
        move_complete = touch('{sessions}/{task_path}/{session_id}/processed/move_to_server.done')

    threads: 32

    priority: 30

    run:
        shell('mv {params.local_root_sorting_folder}/kilosort3 {wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/processed --remove-source-files')
        shell('mv {params.local_root_sorting_folder}/si {wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/processed/si --remove-source-files')
        shell('rm -rf {params.local_root_sorting_folder}/kilosort3')
        shell('rm -rf {params.local_root_sorting_folder}/si')

rule spike_metrics_ks3:
    input:
        rec_properties = rec_properties_input,
        move_complete = '{sessions}/{task_path}/{session_id}/processed/move_to_server.done',

    output:
        metrics_complete = touch('{sessions}/{task_path}/{session_id}/processed/spike_metrics.done')
    
    threads: 32
    
    priority: 10

    script:
        "scripts/s02_cluster_metrics_ks3.py"


rule waveform_and_quality_metrics:
    input:
        rec_properties = rec_properties_input,
        move_complete = '{sessions}/{task_path}/{session_id}/processed/move_to_server.done'

    output:
        si_quality_complete = touch('{sessions}/{task_path}/{session_id}/processed/si_quality.done')
    
    threads: 32
    
    priority: 11

    script:
        "scripts/s02_waveform_and_quality_metrics.py"


rule ephys_sync:
    input:
        rec_properties = rec_properties_input,
        metrics_complete = '{sessions}/{task_path}/{session_id}/processed/spike_metrics.done'

    output:
        ephys_sync_complete = touch('{sessions}/{task_path}/{session_id}/processed/ephys_sync.done')
    
    threads: 32

    priority: 40

    script:
        "scripts/s04_ephys_sync.py"

rule cell_metrics_processing:
    input:
        rec_properties = rec_properties_input,
        ephys_sync_complete = '{sessions}/{task_path}/{session_id}/processed/ephys_sync.done'

    output:
        cell_metrics_processing_complete = touch('{sessions}/{task_path}/{session_id}/processed/cell_metrics_processing.done')

    threads: 32

    priority: 50

    script:
        "scripts/s05_cell_metrics_processing.py"


rule cell_metrics_aggregation:
    input:
        cell_metrics_processing_complete = '{sessions}/{task_path}/{session_id}/processed/cell_metrics_processing.done'

    output:
        cell_metrics_aggregation_complete =  touch('{sessions}/{task_path}/{session_id}/processed/cell_metrics_aggregation.done')

    threads: 32

    priority: 60

    script:
        "scripts/s06_cell_metrics_aggregation.py"

rule cell_metrics_dim_reduction:
    input:
        cell_metrics_aggregation_complete = '{sessions}/{task_path}/{session_id}/processed/cell_metrics_aggregation.done'

    output:
        cell_metrics_dim_reduction_complete =  touch('{sessions}/{task_path}/{session_id}/processed/cell_metrics_dim_reduction.done')

    threads: 32

    priority: 70

    script:
        "scripts/s07_cell_metrics_dim_reduction.py"

rule cell_metrics_clustering:
    input:
        cell_metrics_dim_reduction_complete = '{sessions}/{task_path}/{session_id}/processed/cell_metrics_dim_reduction.done'
    
    output:
        cell_metrics_clustering_complete =  touch('{sessions}/{task_path}/{session_id}/processed/cell_metrics_clustering.done')

    threads: 32

    priority: 80

    script:
        "scripts/s08_cell_metrics_clustering.py"


rule cells_to_xarray:
    input:
        ephys_sync_complete = '{sessions}/{task_path}/{session_id}/processed/ephys_sync.done',
        xr_session = '{sessions}/{task_path}/{session_id}/processed/xr_session.nc',
        sorting_path = Path('{sessions}/{task_path}/{session_id}/processed/kilosort3/')
    
    output:
        xr_spikes_trials = '{sessions}/{task_path}/{session_id}/processed/xr_spikes_trials.nc',
        xr_spikes_session = '{sessions}/{task_path}/{session_id}/processed/xr_spikes_session.nc'

    threads: 32

    priority: 85

    script:
        "scripts/s09_cell_to_xarray.py"

rule session_correlations:
    input:
        xr_spikes_session = '{sessions}/{task_path}/{session_id}/processed/xr_spikes_session.nc'

    output:
        xr_session_correlations = '{sessions}/{task_path}/{session_id}/processed/xr_session_correlations.nc'
    
    threads: 32

    priority: 90

    script:
        "scripts/s10_session_correlations.py"

rule spike_final:
    input:
        cell_metrics_clustering_complete = '{session_path}/{task_path}/{session_id}/processed/cell_metrics_clustering.done'
        
    output:
        done = touch('{session_path}/{task_path}/{session_id}/processed/spike_workflow.done')
