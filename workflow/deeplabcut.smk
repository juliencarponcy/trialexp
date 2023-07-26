from glob import glob
from pathlib import Path
import os 
from dotenv import load_dotenv

load_dotenv()

def task2analyze(tasks:list=None):
    #specify the list of task to analyze to save time.
    total_sessions = []

    if tasks is None:
        tasks=['*']

    for t in tasks:
        total_sessions+=expand('{sessions}/processed/deeplabcut_workflow.done', sessions = Path(os.environ.get('SESSION_ROOT_DIR')).glob(f'{t}/*'))        

    return total_sessions

rule deeplabcut_all:
    input: task2analyze(['reaching_go_spout_bar_nov22', 'reaching_go_spout_incr_break2_nov22','pavlovian_spontanous_reaching_march23'])


def deeplabcut_input(wildcards):
    # determine if there is video file recording for that folder
    video_list = glob(f'{wildcards.session_path}/{wildcards.task_path}/{wildcards.session_id}/video/video_list.txt')
    if len(video_list) > 0:
        return f'{wildcards.session_path}/{wildcards.task_path}/{wildcards.session_id}/processed/deeplabcut.done'
    else:
        return []

rule analyze_video:
    input: 
        video_list = '{session_path}/{task_path}/{session_id}/video/video_list.txt'
    output: 
        dlc_result = '{session_path}/{task_path}/{session_id}/processed/dlc_results.h5'
    script:
        'scripts/deeplabcut/02_analyze_video.py'

rule dlc_preprocess:
    input:
        video_list = '{session_path}/{task_path}/{session_id}/video/video_list.txt',
        dlc_result = '{session_path}/{task_path}/{session_id}/processed/dlc_results.h5'
    output:
        dlc_processed ='{session_path}/{task_path}/{session_id}/processed/dlc_results_clean.pkl'
    script:
        'scripts/deeplabcut/03_dlc_preprocess.py'

rule sync_video:
    input:
        dlc_processed ='{session_path}/{task_path}/{session_id}/processed/dlc_results_clean.pkl',
        pycontrol_dataframe = '{session_path}/{task_path}/{session_id}/processed/df_pycontrol.pkl',
        xr_session = '{session_path}/{task_path}/{session_id}/processed/xr_session.nc',
    output:
        xr_dlc ='{session_path}/{task_path}/{session_id}/processed/xr_session_dlc.nc',
        synced_video = '{session_path}/{task_path}/{session_id}/processed/dlc_synced_video.mp4'
    script:
        'scripts/deeplabcut/04_sync_video.py'

rule deeplabcut_final:
    input:
        spike_metrics_done = deeplabcut_input
    output:
        done = touch('{session_path}/{task_path}/{session_id}/processed/deeplabcut_workflow.done')