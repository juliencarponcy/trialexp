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
        total_sessions+=expand('{sessions}/processed/pycontrol_workflow.done', sessions = Path(os.environ.get('SESSION_ROOT_DIR')).glob(f'{t}/*'))        

    return total_sessions

rule pycontrol_all:
    input: task2analyze(['reaching_go_spout_bar_nov22', 'reaching_go_spout_incr_break2_nov22','pavlovian_spontanous_reaching_march23'])

rule process_pycontrol:
    input:
        session_path = '{session_path}/{task}/{session_id}'
    output:
        event_dataframe = '{session_path}/{task}/{session_id}/processed/df_events_cond.pkl',
        condition_dataframe = '{session_path}/{task}/{session_id}/processed/df_conditions.pkl',
        pycontrol_dataframe = '{session_path}/{task}/{session_id}/processed/df_pycontrol.pkl',
        trial_dataframe = '{session_path}/{task}/{session_id}/processed/df_trials.pkl'
    log:
        '{session_path}/{task}/{session_id}/processed/log/process_pycontrol.log'
    script:
        'scripts/01_process_pycontrol.py'

rule pycontrol_figures:
    input:
        event_dataframe = '{session_path}/{task}/{session_id}/processed/df_events_cond.pkl'
    log:
        '{session_path}/{task}/{session_id}/processed/log/pycontrol_figures.log'
    output:
        event_histogram = report('{session_path}/{task}/{session_id}/processed/figures/event_histogram_{session_id}.png', 
                                    caption='report/event_histogram.rst', category='Plots' ),
        rule_complete = touch('{session_path}/{task}/{session_id}/processed/log/pycontrol.done')
    script:
        'scripts/02_plot_pycontrol_data.py'


rule export_spike2:
    input:
        pycontrol_folder = '{session_path}/{session_id}/pycontrol',
        pycontrol_dataframe = '{session_path}/{session_id}/processed/df_pycontrol.pkl',
        photometry_folder = '{session_path}/{session_id}/pyphotometry'
    output:
        spike2_export_done = touch('{session_path}/{session_id}/processed/spike2_export.done'),
    script:
        'scripts/03_export_spike2.py'

rule import_pyphotometry:
    input:
        pycontrol_dataframe = '{session_path}/{task}/{session_id}/processed/df_pycontrol.pkl',
        trial_dataframe = '{session_path}/{task}/{session_id}/processed/df_trials.pkl',
        event_dataframe = '{session_path}/{task}/{session_id}/processed/df_events_cond.pkl',
        condition_dataframe = '{session_path}/{task}/{session_id}/processed/df_conditions.pkl',
        photometry_folder = '{session_path}/{task}/{session_id}/pyphotometry'
    output:
        xr_photometry = '{session_path}/{task}/{session_id}/processed/xr_photometry.nc',
        xr_session = '{session_path}/{task}/{session_id}/processed/xr_session.nc',
        pycontrol_aligner = '{session_path}/{task}/{session_id}/processed/pycontrol_aligner.pkl'
    script:
        'scripts/04_import_pyphotometry.py'

rule task_specifc_analysis:
    input:
        event_dataframe = '{session_path}/{task}/{session_id}/processed/df_events_cond.pkl',
        xr_photometry = '{session_path}/{task}/{session_id}/processed/xr_photometry.nc',
        xr_session = '{session_path}/{task}/{session_id}/processed/xr_session.nc',
    output:
        rule_complete = touch('{session_path}/{task}/{session_id}/processed/log/task_specific_analysis.done')
    script:
        'scripts/task_specific/common.py'

rule photometry_figure:
    input:
        task_specific = '{session_path}/{task}/{session_id}/processed/log/task_specific_analysis.done',
        xr_session = '{session_path}/{task}/{session_id}/processed/xr_session.nc',
    output:
        trigger_photo_dir= directory('{session_path}/{task}/{session_id}/processed/figures/photometry'),
        rule_complete = touch('{session_path}/{task}/{session_id}/processed/log/photometry.done')
    script:
        'scripts/05_plot_pyphotometry.py'

def photometry_input(wildcards):
    #determine if photometry needs to run in the current folder
    ppd_files = glob(f'{wildcards.session_path}/{wildcards.task}/{wildcards.session_id}/pyphotometry/*.ppd')
    if len(ppd_files)>0:
        return f'{wildcards.session_path}/{wildcards.task}/{wildcards.session_id}/processed/log/photometry.done'
    else:
        return []

rule pycontrol_final:
    input:
        photometry_done = photometry_input,
        pycontrol_done = '{session_path}/{task}/{session_id}/processed/log/pycontrol.done',
        spike2='{session_path}/{task}/{session_id}/processed/spike2_export.done'
    output:
        done = touch('{session_path}/{task}/{session_id}/processed/pycontrol_workflow.done')