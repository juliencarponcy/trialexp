from glob import glob
from pathlib import Path
import os 
from dotenv import load_dotenv

load_dotenv()

def rec_properties_input(wildcards):
    # determine if there is an ephys recording for that folder
    recording_csv = glob(f'{wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/ephys/rec_properties.csv')
    if len(recording_csv) > 0:
        print(f'{wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/ephys/rec_properties.csv')
        return f'{wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/ephys/rec_properties.csv'
    else:
        return []

rule all:
    input: expand('{sessions_root}/processed/spike_workflow.done', sessions_root = Path(os.environ.get('SESSION_ROOT_DIR')).glob('*/*'))

rule spike_sorting:
    input:
        rec_properties = rec_properties_input

    output:
        sorting_complete = touch('{sessions}/{task_path}/{session_id}/processed/spike_sorting.done'),       
    
    threads: 64

    script:
        "scripts/s01_sort_ks3.py"

rule spike_metrics_ks3:
    input:
        sorting_complete = '{sessions}/{task_path}/{session_id}/processed/spike_sorting.done',

    output:
        metrics_complete = touch('{sessions}/{task_path}/{session_id}/processed/spike_metrics.done')
    
    threads: 64
    
    priority: 10

    script:
        "scripts/s02_cluster_metrics_ks3.py"

rule move_to_server:
    input: 
        metrics_complete = '{sessions}/{task_path}/{session_id}/processed/spike_metrics.done'

    params:
        local_root_sorting_folder = Path(os.environ['TEMP_DATA_PATH'])

    output:
        move_complete = touch('{sessions}/{task_path}/{session_id}/processed/move_to_server.done')

    threads: 64

    priority: 30

    shell:
        'mkdir -p {wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/processed/kilosort3'
        'mv {params.local_root_sorting_folder}/kilosort3 {wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/processed/kilosort3'
        'mkdir -p {wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/processed/si'
        'mv {params.local_root_sorting_folder}/si {wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/processed/si'


rule final:
    input:
        move_complete = '{session_path}/{task_path}/{session_id}/processed/move_to_server.done'
    output:
        done = touch('{session_path}/{task_path}/{session_id}/processed/spike_workflow.done')