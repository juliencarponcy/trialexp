from glob import glob
from pathlib import Path

configfile : 'workflows/config/config.yaml'
# report: 'report/workflow.rst'

rule all:
    input: expand('{session_path}/{task_path}/{session_id}/processed/spike_sorting.done', sessions = Path(config['session_root_dir']).glob('*'))


# rule create_folder:
#     output:
#         create_folder_done = touch('{session_path}/{session_id}/spike_sorting.done')
#     script:
#         'scripts/s00_create_session_folders.py'


rule spike_sorting:
    input:
        rec_properties = '{session_path}/{task_path}/{session_id}/ephys/rec_properties.csv'
    output:
        spike_templateA = '{session_path}/{task_path}/{session_id}/ephys/probeA/spike_templates.npy',
        spike_templateB = '{session_path}/{task_path}/{session_id}/ephys/probeB/spike_templates.npy',
        rule_complete = touch('{session_path}/{task_path}/{session_id}/processed/spike_sorting.done')

