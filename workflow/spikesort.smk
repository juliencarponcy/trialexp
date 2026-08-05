from glob import glob
from pathlib import Path
import os 
from dotenv import load_dotenv
from trialexp.config import SESSION_ROOT_DIR

load_dotenv()

include: 'pycontrol.smk'

def rec_properties_input(wildcards):
    # determine if there is an ephys recording for that folder
    recording_csv = glob(f'{wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/ephys/states.npy')
    if len(recording_csv) > 0:
        return f'{wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/processed/spikesort.done'
    else:
        return []


def check_photometry_exists(wildcards):
    ppd_files = glob(f'{wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/pyphotometry/*.ppd')
    if len(ppd_files) > 0:
        return True
    else:
        return False

def session_correlations_input(wildcards):
    # only run if photometry file is present
    ppd_files = glob(f'{wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/pyphotometry/*.ppd')
    if len(ppd_files)>0:
        return f'{wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/processed/xr_corr.nc'
    else:
        return []

def session2analyze(tasks:list=None, cohort:list = None):
    #specify the list of task to analyze to save time.
    total_sessions = []

    if tasks is None:
        tasks=['*']

    if cohort is None:
        cohort = ['*']

    for c in cohort:
        for t in tasks:
            total_sessions+=expand('{sessions}/processed/spike_workflow.done', sessions = Path(SESSION_ROOT_DIR).glob(f'{c}/by_sessions/{t}/*'))        

    return total_sessions

rule spike_all:
     # input: task2analyze(['reaching_go_spout_bar_nov22', 'reaching_go_spout_incr_break2_nov22','pavlovian_spontanous_reaching_march23'])
    input: session2analyze(cohort=['2024_April_cohort'], tasks=['reaching_go_spout_bar_VR_April24'])

rule spike_sorting:
    input:
        rec_properties = '{sessions}/{task_path}/{session_id}/ephys/rec_properties.csv',
    log:
        '{sessions}/{task_path}/{session_id}/processed/log/spike_sorting.log'
    output:
        sorting_complete = touch('{sessions}/{task_path}/{session_id}/processed/spike_sorting.done'), 
        si_output_folder = directory('{sessions}/{task_path}/{session_id}/processed/kilosort4'),
        # spectrogram = '{sessions}/{task_path}/{session_id}/processed/figures/spectrogram.png',
    threads: 96
    script:
        "scripts/spike_sorting/s01_sort_ks4.py"


rule waveform_and_quality_metrics:
    input:
        kilosort_folder = '{sessions}/{task_path}/{session_id}/processed/kilosort4',
        rec_properties = '{sessions}/{task_path}/{session_id}/ephys/rec_properties.csv',
        sorting_complete = '{sessions}/{task_path}/{session_id}/processed/spike_sorting.done'
    log:
        '{sessions}/{task_path}/{session_id}/processed/log/waveform_and_quality_metrics.log'
    output:
        df_quality_metrics = '{sessions}/{task_path}/{session_id}/processed/df_quality_metrics.pkl',
    threads: 32
    script:
        "scripts/spike_sorting/s03_waveform_and_quality_metrics.py"


rule ephys_sync:
    input:
        df_quality_metrics = '{sessions}/{task_path}/{session_id}/processed/df_quality_metrics.pkl',
    log:
        '{sessions}/{task_path}/{session_id}/processed/log/ephys_sync.log'
    output:
        ephys_sync_complete = touch('{sessions}/{task_path}/{session_id}/processed/ephys_sync.done')
    script:
        "scripts/spike_sorting/s04_ephys_sync.py"


rule cells_to_xarray:
    input:
        df_quality_metrics = '{sessions}/{task_path}/{session_id}/processed/df_quality_metrics.pkl',
        ephys_sync_complete = '{sessions}/{task_path}/{session_id}/processed/ephys_sync.done',
        xr_session = '{sessions}/{task_path}/{session_id}/processed/xr_session.nc',   
    log:
        '{sessions}/{task_path}/{session_id}/processed/log/cells_to_xarray.log'
    output:
        xr_spikes_trials = '{sessions}/{task_path}/{session_id}/processed/xr_spikes_trials.nc',
        xr_spikes_fr = '{sessions}/{task_path}/{session_id}/processed/xr_spikes_fr.nc',
        neo_spike_train = '{sessions}/{task_path}/{session_id}/processed/neo_spiketrain.pkl',
        df_aggregate = '{sessions}/{task_path}/{session_id}/processed/df_aggregate.pkl',
    script:
        "scripts/spike_sorting/s09_cell_to_xarray.py"

rule cell_overview_plot:
    input:
        xr_spikes_trials = '{sessions}/{task_path}/{session_id}/processed/xr_spikes_trials.nc',
        xr_spikes_fr = '{sessions}/{task_path}/{session_id}/processed/xr_spikes_fr.nc',
        pycontrol_dataframe = '{sessions}/{task_path}/{session_id}/processed/df_pycontrol.pkl',
        xr_session = '{sessions}/{task_path}/{session_id}/processed/xr_session.nc',  
    log:
        '{sessions}/{task_path}/{session_id}/processed/log/cell_overview_plot.log'     
    output:
        figures_path = directory('{sessions}/{task_path}/{session_id}/processed/figures/ephys/overview'),
        cell_overview_complete = touch('{sessions}/{task_path}/{session_id}/processed/cell_overview.done'),
    threads: 8
    script:
        "scripts/spike_sorting/s11_cell_overview_plot.py"

rule cell_response_comparison:
    input:
        cell_overview_complete = '{sessions}/{task_path}/{session_id}/processed/cell_overview.done',
        xr_spikes_trials = '{sessions}/{task_path}/{session_id}/processed/xr_spikes_trials.nc',
        xr_spikes_fr = '{sessions}/{task_path}/{session_id}/processed/xr_spikes_fr.nc',
        pycontrol_dataframe = '{sessions}/{task_path}/{session_id}/processed/df_pycontrol.pkl',
        xr_session = '{sessions}/{task_path}/{session_id}/processed/xr_session.nc',   
    log:
        '{sessions}/{task_path}/{session_id}/processed/log/cell_response_comparison.log'     
    output:
        response_curves_path = directory('{sessions}/{task_path}/{session_id}/processed/figures/ephys/response_curves'),
        df_cell_prop = '{sessions}/{task_path}/{session_id}/processed/df_cell_prop.pkl',
        cell_trial_responses_complete = touch('{sessions}/{task_path}/{session_id}/processed/cell_response_comparison.done'),
    threads: 96
    script:
        "scripts/spike_sorting/s11b_cell_response_comparison.py"



rule session_correlations:
    input: 
        df_quality_metrics = '{sessions}/{task_path}/{session_id}/processed/df_quality_metrics.pkl',
        xr_spike_fr = '{sessions}/{task_path}/{session_id}/processed/xr_spikes_fr.nc',
        xr_spikes_trials = '{sessions}/{task_path}/{session_id}/processed/xr_spikes_trials.nc',
    log:
        '{sessions}/{task_path}/{session_id}/processed/log/session_correlations.log'   
    output:
        xr_corr = '{sessions}/{task_path}/{session_id}/processed/xr_corr.nc',
        corr_plots = directory('{sessions}/{task_path}/{session_id}/processed/figures/ephys/correlations'),
    threads: 32
    script:
        "scripts/spike_sorting/s12_session_correlations.py"


rule spike_timewarp:
    input:
        xr_spikes_fr = '{sessions}/{task_path}/{session_id}/processed/xr_spikes_fr.nc',
        condition_dataframe = '{sessions}/{task_path}/{session_id}/processed/df_conditions.pkl',
        event_dataframe = '{sessions}/{task_path}/{session_id}/processed/df_events_cond.pkl',
    log:
        '{sessions}/{task_path}/{session_id}/processed/log/spike_timewarp.log'     
    output:
        xr_timewarpped = '{sessions}/{task_path}/{session_id}/processed/xr_spikes_timewarped.nc',
    script:
        "scripts/spike_sorting/s13_time_warping.py"


rule spike_timewarp_figures:
    input:
        xr_spike_timewarp = '{sessions}/{task_path}/{session_id}/processed/xr_spikes_timewarped.nc',
        xr_session = '{sessions}/{task_path}/{session_id}/processed/xr_session.nc',
        xr_photom_timewarp = '{sessions}/{task_path}/{session_id}/processed/xr_photom_timewarped.nc',
        xr_corr = '{sessions}/{task_path}/{session_id}/processed/xr_corr.nc',
    log:
        '{sessions}/{task_path}/{session_id}/processed/log/spike_timewarp_figures.log'     
    output:
        figure_dir= directory('{sessions}/{task_path}/{session_id}/processed/figures/ephys/timewarp'),
        done = touch('{sessions}/{task_path}/{session_id}/processed/spike_timewarp_figures.done')
    script:
        "scripts/spike_sorting/s13b_time_warping_figures.py"

rule cell_classification:
    input: 
        df_quality_metrics = '{sessions}/{task_path}/{session_id}/processed/df_quality_metrics.pkl',
    log:
        '{sessions}/{task_path}/{session_id}/processed/log/cell_classification.log'   
    output:
        df_cell_types = '{sessions}/{task_path}/{session_id}/processed/df_celltypes.pkl'
    script:
        "scripts/spike_sorting/s14_cell_classification.py"

rule localization:
    input: 
        xr_spikes_trials = '{sessions}/{task_path}/{session_id}/processed/xr_spikes_trials.nc',
    log:
        '{sessions}/{task_path}/{session_id}/processed/log/localization.log'   
    output:
        xr_local = '{sessions}/{task_path}/{session_id}/processed/xr_localization.nc'
    script:
        "scripts/spike_sorting/s16_localization.py"

rule curation:
    input: 
        df_quality_metrics = '{sessions}/{task_path}/{session_id}/processed/df_quality_metrics.pkl',
    log:
        '{sessions}/{task_path}/{session_id}/processed/log/curation.log'   
    output:
        df_bombcell = '{sessions}/{task_path}/{session_id}/processed/df_bombcell.pkl',
        df_qm_table = '{sessions}/{task_path}/{session_id}/processed/df_qm_table.pkl',
        curation_plots = directory('{sessions}/{task_path}/{session_id}/processed/figures/ephys/curation')
    script:
        "scripts/spike_sorting/s15_curation.py"


rule spikesort_done:
    input:
        corr_plot = branch(check_photometry_exists, 
                            then = '{sessions}/{task_path}/{session_id}/processed/xr_corr.nc', 
                            otherwise = []),
        spike_timewarp_figures = branch(check_photometry_exists,
                                    then = '{sessions}/{task_path}/{session_id}/processed/spike_timewarp_figures.done',
                                    otherwise = []),
        df_bombcell = '{sessions}/{task_path}/{session_id}/processed/df_bombcell.pkl',
        xr_local = '{sessions}/{task_path}/{session_id}/processed/xr_localization.nc',
        df_cell_types = '{sessions}/{task_path}/{session_id}/processed/df_celltypes.pkl',
        comparison_done = '{sessions}/{task_path}/{session_id}/processed/cell_response_comparison.done',
        cell_trial_responses_complete = '{sessions}/{task_path}/{session_id}/processed/cell_overview.done',
        time_warp = '{sessions}/{task_path}/{session_id}/processed/xr_spikes_timewarped.nc',
    
    priority: 20
    output:
        spike_sort_done = touch('{sessions}/{task_path}/{session_id}/processed/spikesort.done'),

rule spike_workflow_final:
    input:
        rec_properties_input
    output:
        done = touch('{sessions}/{task_path}/{session_id}/processed/spike_workflow.done')
