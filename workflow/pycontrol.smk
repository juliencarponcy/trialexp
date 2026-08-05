from glob import glob
from pathlib import Path
import os
from dotenv import load_dotenv
from trialexp.config import SESSION_ROOT_DIR #cannot import the root config module in snakefile

load_dotenv()

def session2analyze(tasks:list=None, cohort:list = None, animals:list = None):
    #specify the list of task to analyze to save time.
    total_sessions = []

    if tasks is None:
        tasks=['*']

    if cohort is None:
        cohort = ['*']

    if animals is None:
        animals = ['*']
    else:
        animals = [f'{a}*' for a in animals]
    

    for c in cohort:
        for a in animals:
            for t in tasks:
                session_paths = list(Path(SESSION_ROOT_DIR).glob(f'{c}/by_sessions/{t}/{a}'))
                for session_path in session_paths:
                    target_file = f"{str(session_path)}/processed/pycontrol_workflow.done"
                    total_sessions.append(target_file)  
                
    return total_sessions

rule pycontrol_all:
    input: session2analyze(cohort=['2025_July_cohort','2025_June_cohort','2025_October_cohort','2025_September_cohort','2026_January_cohort','2026_April_cohort'])

'''
Ach/DA cohort
2024_April_cohort
2024_August_cohort
2025_Feb_cohort

HC:
2025_January_cohort
2025_October_cohort
'''


rule process_pycontrol:
    input:
        session_path = '{session_path}/{task}/{session_id}'
    output:
        event_dataframe = '{session_path}/{task}/{session_id}/processed/df_events_cond.pkl',
        condition_dataframe = '{session_path}/{task}/{session_id}/processed/df_conditions.pkl',
        pycontrol_dataframe = '{session_path}/{task}/{session_id}/processed/df_pycontrol.pkl',
        trial_dataframe = '{session_path}/{task}/{session_id}/processed/df_trials.pkl',
    log:
        '{session_path}/{task}/{session_id}/processed/log/process_pycontrol.log'
    script:
        'scripts/01_process_pycontrol.py'

rule pycontrol_figures:
    input:
        event_dataframe = '{session_path}/{task}/{session_id}/processed/df_events_cond.pkl',
        df_pycontrol = '{session_path}/{task}/{session_id}/processed/df_pycontrol.pkl',
    log:
        '{session_path}/{task}/{session_id}/processed/log/pycontrol_figures.log'
    output:
        event_histogram = report('{session_path}/{task}/{session_id}/processed/figures/event_histogram.png', 
                                    caption='report/event_histogram.rst', category='Plots' ),
        reach_histogram = '{session_path}/{task}/{session_id}/processed/figures/reach_histogram.png',
        discrim_scores  = '{session_path}/{task}/{session_id}/processed/figures/discrim_score.png',
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


rule export_parquet:
    input:
        xr_photometry = '{session_path}/{task}/{session_id}/processed/xr_photometry.nc',
        pycontrol_dataframe = '{session_path}/{task}/{session_id}/processed/df_pycontrol.pkl',
    output:
        photometry_parquet = '{session_path}/{task}/{session_id}/processed/photometry.parquet'
    log:            
        '{session_path}/{task}/{session_id}/processed/log/export_parquet.log'
    script:
        'scripts/03_export_parquet.py'


rule import_pyphotometry:
    input:
        pycontrol_dataframe = '{session_path}/{task}/{session_id}/processed/df_pycontrol.pkl',
        trial_dataframe = '{session_path}/{task}/{session_id}/processed/df_trials.pkl',
        event_dataframe = '{session_path}/{task}/{session_id}/processed/df_events_cond.pkl',
        condition_dataframe = '{session_path}/{task}/{session_id}/processed/df_conditions.pkl',
        photometry_folder = '{session_path}/{task}/{session_id}/pyphotometry'
    log:
        '{session_path}/{task}/{session_id}/processed/log/import_pyphotometry.log'
    output:
        xr_photometry = '{session_path}/{task}/{session_id}/processed/xr_photometry.nc',
        xr_session = '{session_path}/{task}/{session_id}/processed/xr_session.nc',
        pycontrol_aligner = '{session_path}/{task}/{session_id}/processed/pycontrol_aligner.pkl'
    script:
        'scripts/04_import_pyphotometry.py'


rule photometry_figure:
    input:
        xr_session = '{session_path}/{task}/{session_id}/processed/xr_session.nc',
    output:
        trigger_photo_dir= directory('{session_path}/{task}/{session_id}/processed/figures/photometry'),
        done = touch('{session_path}/{task}/{session_id}/processed/log/photometry_figure.done'),
    log:
        '{session_path}/{task}/{session_id}/processed/log/photometry_figure.log'
    script:
        'scripts/05_plot_pyphotometry.py'

rule photometry_pipline:
    input:
        xr_timewarpped = '{session_path}/{task}/{session_id}/processed/xr_photom_timewarped.nc',
        trigger_photo_dir= '{session_path}/{task}/{session_id}/processed/log/photometry_figure.done',
        photometry_parquet = '{session_path}/{task}/{session_id}/processed/photometry.parquet',
    output:
        rule_complete = touch('{session_path}/{task}/{session_id}/processed/log/photometry.done')



rule behavorial_analysis:
    input:
        condition_dataframe = '{session_path}/{task}/{session_id}/processed/df_conditions.pkl',
        event_dataframe = '{session_path}/{task}/{session_id}/processed/df_events_cond.pkl',
    log:
        '{session_path}/{task}/{session_id}/processed/log/behavorial_analysis.log'  
    output:
        xr_behaviour = '{session_path}/{task}/{session_id}/processed/xr_behaviour.nc',
    script:
        'scripts/06_behavorial_analysis.py'


rule time_warping:
    input:
        xr_photometry = '{session_path}/{task}/{session_id}/processed/xr_photometry.nc',
        condition_dataframe = '{session_path}/{task}/{session_id}/processed/df_conditions.pkl',
        event_dataframe = '{session_path}/{task}/{session_id}/processed/df_events_cond.pkl',
    output:
        xr_timewarpped = '{session_path}/{task}/{session_id}/processed/xr_photom_timewarped.nc',
        figure_dir= directory('{session_path}/{task}/{session_id}/processed/figures/timewarp'),
    log:
        '{session_path}/{task}/{session_id}/processed/log/time_warping.log'  
    script:
        'scripts/07_time_warping.py'

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
        xr_session = '{session_path}/{task}/{session_id}/processed/xr_session.nc',
        pycontrol_done = '{session_path}/{task}/{session_id}/processed/log/pycontrol.done',
        xr_behaviour = '{session_path}/{task}/{session_id}/processed/xr_behaviour.nc',
    output:
        done = touch('{session_path}/{task}/{session_id}/processed/pycontrol_workflow.done')