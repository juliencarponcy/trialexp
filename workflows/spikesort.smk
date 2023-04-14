from glob import glob
from pathlib import Path
import os 

rule all:
    input: expand('{sessions}/processed/spike_workflow.done', sessions = Path(os.environ.get('SESSION_ROOT_DIR')).glob('*/*'))


# rule create_folder:
#     output:
#         create_folder_done = touch('{session_path}/{session_id}/spike_sorting.done')
#     script:
#         'scripts/s00_create_session_folders.py'



rule spike_sorting:
    input:
        rec_properties = '{session_path}/{task_path}/{session_id}/ephys/rec_properties.csv'
    output:
        output_folder = directory('{session_path}/{task_path}/{session_id}/ephys/output'),
        sorting_folder = directory('{session_path}/{task_path}/{session_id}/ephys/sorting'),
        rule_complete = touch(r'{session_path}/{task_path}/{session_id}/processed/spike_sorting.done')
    log:
        '{session_path}/{task_path}/{session_id}/processed/log/process_spike_sorting.log'
    script:
        'scripts/s01_sort_ks3.py'

def rec_properties_input(wildcards):
    # determine if there is an ephys recording for that folder
    recording_csv = glob(f'{wildcards.session_path}/{wildcards.task_path}/{wildcards.session_id}/ephys/rec_properties.csv')
    if len(recording_csv) > 0:
        return f'{wildcards.session_path}/{wildcards.task_path}/{wildcards.session_id}/processed/spike_sorting.done'
    else:
        return []

rule final:
    input:
        spike_sorting_done = rec_properties_input,
    output:
        done = touch('{session_path}/{task_path}/{session_id}/processed/spike_workflow.done')