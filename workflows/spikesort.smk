from glob import glob
from pathlib import Path

configfile : 'workflows/config/config.yaml'
# report: 'report/workflow.rst'


rule all:
    input: expand('{sessions}/processed/spike_sorting.done', sessions = Path(config['session_root_dir']).glob('*/*'))


# rule create_folder:
#     output:
#         create_folder_done = touch('{session_path}/{session_id}/spike_sorting.done')
#     script:
#         'scripts/s00_create_session_folders.py'

def rec_properties_input(wildcards):
    # determine if there is an ephys recording for that folder
    recording_csv = glob(f'{wildcards.session_path}/{wildcards.task_path}/{wildcards.session_id}/ephys/rec_properties.csv')
    if len(recording_csv) > 0:
        return f'{wildcards.session_path}/{wildcards.task_path}/{wildcards.session_id}/ephys/rec_properties.csv'
    else:
        return []

rule spike_sorting:
    input:
        rec_properties = rec_properties_input
    output:
        spike_templateA = '{session_path}/{task_path}/{session_id}/ephys/sorted/probeA/spike_templates.npy',
        spike_templateB = '{session_path}/{task_path}/{session_id}/ephys/sorted/probeB/spike_templates.npy',
        rule_complete = touch(r'{session_path}/{task_path}/{session_id}/processed/spike_sorting.done')
    threads: 64
    log:
        '{session_path}/{task_path}/{session_id}/processed/log/process_spike_sorting.log'
    shell:
        "python workflows/scripts/s01_sort_ks3.py --log {log} {input} {output}"

rule final:
    input:
        spike_sorting_done = '{session_path}/{task_path}/{session_id}/processed/spike_sorting.done'
    output:
        done = touch('{session_path}/{task_path}/{session_id}/processed/spike_workflow.done')