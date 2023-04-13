from glob import glob
from pathlib import Path
import os

configfile: 'workflows/config/config.yaml'
# report: 'report/workflow.rst'


def check_input_exists(wildcards):
    # determine if there is an ephys recording for that folder
    recording_csv = glob(f'{wildcards.session_path}/{wildcards.task_path}/{wildcards.session_id}/ephys/rec_properties.csv')
    if len(recording_csv) > 0:
        return recording_csv[0]  # Return the first matching file path
    else:
        return ''  # Return an empty string if no file found


rule all:
    input:
        expand('{sessions}/processed/spike_sorting.done', sessions=Path(config['session_root_dir']).glob('*/*'))

rule spike_sorting:
    input:
        rec_properties=check_input_exists
    output:
        spike_templateA='{session_path}/{task_path}/{session_id}/ephys/sorting/probeA/spike_templates.npy',
        spike_templateB='{session_path}/{task_path}/{session_id}/ephys/sorting/probeB/spike_templates.npy',
        rule_complete=touch('{session_path}/{task_path}/{session_id}/processed/spike_sorting.done')
    script:
        'scripts/s01_sort_ks3.py'

rule final:
    input:
        spike_sorting_done='{session_path}/{task_path}/{session_id}/processed/spike_sorting.done'
    output:
        done=touch('{session_path}/{task_path}/{session_id}/processed/spike_workflow.done')
    run:
        # Your final rule logic here
        pass
