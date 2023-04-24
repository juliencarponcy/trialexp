from glob import glob
from pathlib import Path
import os 
from dotenv import load_dotenv

load_dotenv()

envvars: 
    'TEMP_DATA_PATH'

def rec_properties_input(wildcards):
    # determine if there is an ephys recording for that folder
    recording_csv = glob(f'{wildcards.session_path}/{wildcards.task_path}/{wildcards.session_id}/ephys/rec_properties.csv')
    if len(recording_csv) > 0:
        return f'{wildcards.session_path}/{wildcards.task_path}/{wildcards.session_id}/ephys/rec_properties.csv'
    else:
        return []

rule all:
    input: expand('{sessions}/processed/spike_workflow.done', sessions = Path(os.environ.get('SESSION_ROOT_DIR')).glob('*/*'))

rule spike_sorting:
    input:
        rec_properties = rec_properties_input

    # params:
    #     local_root_sorting_folder = os.environ['TEMP_DATA_PATH']

    output:
        rule_complete = touch('{session_path}/{task_path}/{session_id}/processed/spike_sorting.done'),       
        ks_3_dir_A = '{session_path}/{task_path}/{session_id}/ProbeA/sorter_output',
        ks_3_dir_B = '{session_path}/{task_path}/{session_id}/ProbeB/sorter_output'
        # ks_3_dir_A = directory('{params.local_root_sorting_folder}/probeA/sorter_output'),
        # ks_3_dir_B = directory('{params.local_root_sorting_folder}/probeB/sorter_output')
    
    threads: 64

    script:
        "scripts/s01_sort_ks3.py"

rule spike_metrics_ks3:
    input:
        sorting_complete = '{params.local_root_sorting_folder}/{task_path}/{session_id}/processed/spike_sorting.done',
        ks_3_dir_A = '{params.local_root_sorting_folder}/{task_path}/{session_id}/probeA/sorter_output',
        ks_3_dir_B = '{params.local_root_sorting_folder}/{task_path}/{session_id}/probeB/sorter_output'

    # params:
    #     local_root_sorting_folder = os.environ['TEMP_DATA_PATH']

    output:
        spike_metrics_A = '{params.local_root_sorting_folder}/{task_path}/{session_id}/probeA/sorter_output/recording.cell_metrics.cellinfo.mat',
        spike_metrics_B = '{params.local_root_sorting_folder}/{task_path}/{session_id}/probeB/sorter_output/recording.cell_metrics.cellinfo.mat',
        metrics_complete = touch('{params.local_root_sorting_folder}/{task_path}/{session_id}/processed/spike_metrics.done')
    
    threads: 64
    
    priority: 10

    script:
        "scripts/s02_cluster_metrics_ks3.py"

rule move_to_server:
    input: 
        rec_properties = rules.spike_sorting.input.rec_properties,
        # metrics_complete = '{params.local_root_sorting_folder}/{task_path}/{session_id}/processed/spike_metrics.done',
        ks_3_dir_A = '{params.local_root_sorting_folder}/{task_path}/{session_id}/ProbeA/sorter_output',
        ks_3_dir_B = '{params.local_root_sorting_folder}/{task_path}/{session_id}/ProbeB/sorter_output',

    # params:
    #     local_root_sorting_folder = Path(os.environ['TEMP_DATA_PATH'])

    output:
        move_complete = touch('{session_path}/{task_path}/{session_id}/processed/move_to_server.done')

    threads: 64

    priority: 30

    shell: 
        'mv {input.ks_3_dir_A} {wildcards.session_path}/{wildcards.task_path}/{wildcards.session_id}/ephys/sorter/kilosort3/ProbeA'
        'mv {input.ks_3_dir_B} {wildcards.session_path}/{wildcards.task_path}/{wildcards.session_id}/ephys/sorter/kilosort3/ProbeB'


rule final:
    input:
        move_complete = '{session_path}/{task_path}/{session_id}/processed/move_to_server.done'
    output:
        done = touch('{session_path}/{task_path}/{session_id}/processed/spike_workflow.done')