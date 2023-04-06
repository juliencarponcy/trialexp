from glob import glob
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

envvars:
    'SNAKEMAKE_DEBUG_ROOT',
    'SESSION_ROOT_FOLDER'

rule all:
    input: expand('{path}/processed/task.done', path = Path(os.environ['SESSION_ROOT_FOLDER']).glob('*/*'))
    
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
        event_histogram = '{session_path}/{task}/{session_id}/processed/figures/event_histogram.png', 
        rule_complete = touch('{session_path}/{task}/{session_id}/processed/log/pycontrol.done')
    script:
        'scripts/02_plot_pycontrol_data.py'

rule export_spike2:
    input:
        pycontrol_dataframe = '{session_path}/{task}/{session_id}/processed/df_pycontrol.pkl',
    output:
        spike2_file = '{session_path}/{task}/{session_id}/processed/spike2.smrx',
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
    log:
        '{session_path}/{task}/{session_id}/processed/log/import_pyphotometry.log'
    script:
        'scripts/04_import_pyphotometry.py'

rule photometry_figure:
    input:
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

rule final:
    input:
        photometry_done = photometry_input,
        pycontrol_done = '{session_path}/{task}/{session_id}/processed/log/pycontrol.done',
        spike2='{session_path}/{task}/{session_id}/processed/spike2.smrx'
    output:
        done = touch('{session_path}/{task}/{session_id}/processed/task.done')