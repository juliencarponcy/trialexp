from glob import glob
from pathlib import Path
import os 
from dotenv import load_dotenv

load_dotenv()

def rec_properties_input(wildcards):
    # determine if there is an ephys recording for that folder
    recording_csv = glob(f'{wildcards.session_path}/{wildcards.task_path}/{wildcards.session_id}/ephys/rec_properties.csv')
    if len(recording_csv) > 0:
        return f'{wildcards.session_path}/{wildcards.task_path}/{wildcards.session_id}/ephys/rec_properties.csv'
    else:
        return []

rule spike_sort_all:
    input: expand('{sessions}/processed/spike_workflow.done', sessions = Path(os.environ.get('SESSION_ROOT_DIR')).glob('*/*'))

rule spike_sorting:
    input:
        rec_properties = rec_properties_input
    output:
        rule_complete = touch('{session_path}/{task_path}/{session_id}/processed/spike_sorting.done'),       
        ks_3_spike_templates_A = '{session_path}/{task_path}/{session_id}/ephys/sorter/kilosort3/probeA/sorter_output/spike_templates.npy',
        ks_3_spike_templates_B = '{session_path}/{task_path}/{session_id}/ephys/sorter/kilosort3/probeB/sorter_output/spike_templates.npy'
    threads: 64
    log:
        '{session_path}/{task_path}/{session_id}/processed/log/process_spike_sorting.log'
    script:
        "scripts/s01_sort_ks3.py"

rule spike_metrics_ks3:
    input:
        sorting_complete = '{session_path}/{task_path}/{session_id}/processed/spike_sorting.done',
        ks_3_spike_templates_A = '{session_path}/{task_path}/{session_id}/ephys/sorter/kilosort3/probeA/sorter_output/spike_templates.npy',
        ks_3_spike_templates_B = '{session_path}/{task_path}/{session_id}/ephys/sorter/kilosort3/probeB/sorter_output/spike_templates.npy'

    output:
        spike_metrics_A = '{session_path}/{task_path}/{session_id}/ephys/sorter/kilosort3/ProbeA/sorter_output/recording.cell_metrics.cellinfo.mat',
        spike_metrics_B = '{session_path}/{task_path}/{session_id}/ephys/sorter/kilosort3/ProbeB/sorter_output/recording.cell_metrics.cellinfo.mat',
        rule_complete = touch('{session_path}/{task_path}/{session_id}/processed/spike_metrics.done')
    threads: 64
    log:
        '{session_path}/{task_path}/{session_id}/processed/log/process_spike_metrics.log'
    script:
        "scripts/s02_cluster_metrics_ks3.py"


rule spike_sort_final:
    input:
        spike_metrics_done = '{session_path}/{task_path}/{session_id}/processed/spike_metrics.done'
    output:
        done = touch('{session_path}/{task_path}/{session_id}/processed/spike_workflow.done')