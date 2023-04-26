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

    run:
        shell('mkdir -p {wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/processed/kilosort3')
        #modify to avoid double nesting of processed/kilosort3/kilosort3 not done as pipeline would take so much time to run again
        shell('mv {params.local_root_sorting_folder}/kilosort3 {wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/processed/kilosort3')
        shell('mkdir -p {wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/processed/si')
        shell('mv {params.local_root_sorting_folder}/si {wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/processed/si')
        # The following should not be necessary but did not worked properly, maybe because of filesystem on ettin
        shell('rm -rf {params.local_root_sorting_folder}/kilosort3')
        shell('rm -rf {params.local_root_sorting_folder}/si')

rule ephys_sync:
    input:
        rec_properties = rec_properties_input,
        move_complete = '{sessions}/{task_path}/{session_id}/processed/move_to_server.done'

    output:
        ephys_sync_complete = touch('{sessions}/{task_path}/{session_id}/processed/ephys_sync.done')

    script:
        "scripts/s04_ephys_sync.py"

rule final:
    input:
        ephys_sync_complete = '{session_path}/{task_path}/{session_id}/processed/ephys_sync.done'
    output:
        done = touch('{session_path}/{task_path}/{session_id}/processed/spike_workflow.done')