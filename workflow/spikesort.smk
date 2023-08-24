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
        return f'{wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/processed/spikesort.done'
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

def task2analyze(tasks:list=None):
    #specify the list of task to analyze to save time.
    total_sessions = []

    if tasks is None:
        tasks=['*']

    for t in tasks:
        total_sessions+=expand('{sessions}/processed/spike_workflow.done', sessions = Path(os.environ.get('SESSION_ROOT_DIR')).glob(f'{t}/TT*'))        

    return total_sessions

rule spike_all:
     # input: task2analyze(['reaching_go_spout_bar_nov22', 'reaching_go_spout_incr_break2_nov22','pavlovian_spontanous_reaching_march23'])
    input: task2analyze(['reaching_go_spout_bar_nov22'])

rule spike_sorting:
    input:
        rec_properties = '{sessions}/{task_path}/{session_id}/ephys/rec_properties.csv',
    output:
        sorting_complete = touch('{sessions}/{task_path}/{session_id}/processed/spike_sorting.done'),           
    threads: 32
    script:
        "scripts/spike_sorting/s01_sort_ks3.py"


rule spike_metrics_ks3:
    input:
        rec_properties = '{sessions}/{task_path}/{session_id}/ephys/rec_properties.csv',
        # move_complete = '{sessions}/{task_path}/{session_id}/processed/move_to_server.done',
        sorting_complete = '{sessions}/{task_path}/{session_id}/processed/spike_sorting.done'
    output:
        metrics_complete = touch('{sessions}/{task_path}/{session_id}/processed/spike_metrics.done')
    threads: 32
    priority: 10
    script:
        "scripts/spike_sorting/s02_cluster_metrics_ks3.py"


rule waveform_and_quality_metrics:
    input:
        rec_properties = '{sessions}/{task_path}/{session_id}/ephys/rec_properties.csv',
        sorting_complete = '{sessions}/{task_path}/{session_id}/processed/spike_sorting.done'
    output:
        si_quality_complete = touch('{sessions}/{task_path}/{session_id}/processed/si_quality.done')
    threads: 32
    priority: 11
    script:
        "scripts/spike_sorting/s02b_waveform_and_quality_metrics.py"


rule ephys_sync:
    input:
        kilosort_path = '{sessions}/{task_path}/{session_id}/processed/kilosort3',
        metrics_complete = '{sessions}/{task_path}/{session_id}/processed/spike_metrics.done'
    output:
        ephys_sync_complete = touch('{sessions}/{task_path}/{session_id}/processed/ephys_sync.done')
    threads: 32
    priority: 40
    script:
        "scripts/spike_sorting/s04_ephys_sync.py"

rule cell_metrics_processing:
    input:
        rec_properties = '{sessions}/{task_path}/{session_id}/ephys/rec_properties.csv',
        kilosort_path = '{sessions}/{task_path}/{session_id}/processed/kilosort3',
        ephys_sync_complete = '{sessions}/{task_path}/{session_id}/processed/ephys_sync.done',
    output:
        cell_matrics_full= '{sessions}/{task_path}/{session_id}/processed/kilosort3/cell_metrics_full.nc'
    threads: 32
    priority: 50
    script:
        "scripts/spike_sorting/s05_cell_metrics_processing.py"


rule cell_metrics_aggregation:
    input:
        cell_matrics_full= '{sessions}/{task_path}/{session_id}/processed/kilosort3/cell_metrics_full.pkl'
    output:
        cell_metrics_aggregation_complete =  touch('{sessions}/{task_path}/{session_id}/processed/cell_metrics_aggregation.done')
    threads: 32
    priority: 60
    script:
        "scripts/spike_sorting/s06_cell_metrics_aggregation.py"

rule cell_metrics_dim_reduction:
    input:
        cell_metrics_aggregation_complete = '{sessions}/{task_path}/{session_id}/processed/cell_metrics_aggregation.done'
    output:
        cell_metrics_dim_reduction_complete =  touch('{sessions}/{task_path}/{session_id}/processed/cell_metrics_dim_reduction.done')
    threads: 32
    priority: 70
    script:
        "scripts/spike_sorting/s07_cell_metrics_dim_reduction.py"

rule cell_metrics_clustering:
    input:
        cell_metrics_dim_reduction_complete = '{sessions}/{task_path}/{session_id}/processed/cell_metrics_dim_reduction.done' 
    output:
        cell_metrics_clustering_complete =  touch('{sessions}/{task_path}/{session_id}/processed/cell_metrics_clustering.done')
    threads: 32
    priority: 80
    script:
        "scripts/spike_sorting/s08_cell_metrics_clustering.py"


rule cells_to_xarray:
    input:
        ephys_sync_complete = '{sessions}/{task_path}/{session_id}/processed/ephys_sync.done',
        xr_session = '{sessions}/{task_path}/{session_id}/processed/xr_session.nc',   
        sorting_path = '{sessions}/{task_path}/{session_id}/processed/kilosort3',   
        cell_matrics_full= '{sessions}/{task_path}/{session_id}/processed/kilosort3/cell_metrics_full.nc' 
    output:
        xr_spikes_trials = '{sessions}/{task_path}/{session_id}/processed/xr_spikes_trials.nc',
        xr_spikes_fr = '{sessions}/{task_path}/{session_id}/processed/xr_spikes_fr.nc',
        neo_spike_train = '{sessions}/{task_path}/{session_id}/processed/neo_spiketrain.pkl',
        spike_sort_done = touch('{sessions}/{task_path}/{session_id}/processed/spikesort.done'),
    threads: 32
    priority: 85
    script:
        "scripts/spike_sorting/s09_cell_to_xarray.py"

# rule cell_anatomy:
#     input:
#         xr_spikes_trials = '{sessions}/{task_path}/{session_id}/processed/xr_spikes_trials.nc',
#         xr_spikes_trials_phases = '{sessions}/{task_path}/{session_id}/processed/xr_spikes_trials_phases.nc',
#         xr_spikes_full_session = '{sessions}/{task_path}/{session_id}/processed/xr_spikes_full_session.nc'
#     output:
#         cell_anatomy_complete = '{sessions}/{task_path}/{session_id}/processed/ephys_anatomy.done'

#     threads: 32

#     script:
#         "scripts/spike_sorting/s10_cell_anatomy.py"

rule cell_trial_responses_plot:
    input:
        xr_spikes_trials = '{sessions}/{task_path}/{session_id}/processed/xr_spikes_trials.nc',
        xr_spikes_fr = '{sessions}/{task_path}/{session_id}/processed/xr_spikes_fr.nc',
        pycontrol_dataframe = '{sessions}/{task_path}/{session_id}/processed/df_pycontrol.pkl',
        xr_session = '{sessions}/{task_path}/{session_id}/processed/xr_session.nc',       
    output:
        figures_path = directory('{sessions}/{task_path}/{session_id}/processed/figures/ephys'),
        cell_trial_responses_complete = touch('{sessions}/{task_path}/{session_id}/processed/cell_trial_responses.done')
    threads: 32

    script:
        "scripts/spike_sorting/s11_cell_trial_responses_plot.py"

rule session_correlations:
    input:
        # xr_session = '{sessions}/{task_path}/{session_id}/processed/xr_session.nc',
        xr_spike_fr = '{sessions}/{task_path}/{session_id}/processed/xr_spikes_fr.nc',
    output:
        xr_session_correlations = '{sessions}/{task_path}/{session_id}/processed/xr_session_correlations.nc'
    threads: 32
    priority: 90
    script:
        "scripts/spike_sorting/s10_session_correlations.py"

rule spike_final:
    input:
        rec_properties_input
        # xr_spikes_trials = '{session_path}/{task_path}/{session_id}/processed/xr_spikes_trials.nc'
        
    output:
        done = touch('{sessions}/{task_path}/{session_id}/processed/spike_workflow.done')
