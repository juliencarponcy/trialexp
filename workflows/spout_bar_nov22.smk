from glob import glob
from pathlib import Path

configfile : 'workflows/config/config.yaml'
# report: 'report/workflow.rst'

rule all:
    input: expand('{sessions}/processed/task.done', sessions = Path(config['session_root_dir']).glob('*'))

rule process_pycontrol:
    input:
        session_path = '{session_path}/{session_id}'
    output:
        event_dataframe = '{session_path}/{session_id}/processed/df_events_cond.pkl',
        condition_dataframe = '{session_path}/{session_id}/processed/df_conditions.pkl',
        pycontrol_dataframe = '{session_path}/{session_id}/processed/df_pycontrol.pkl',
        trial_dataframe = '{session_path}/{session_id}/processed/df_trials.pkl'
    log:
        '{session_path}/{session_id}/processed/log/process_pycontrol.log'
    script:
        'scripts/01_process_pycontrol.py'

rule pycontrol_figures:
    input:
        event_dataframe = '{session_path}/{session_id}//processed/df_events_cond.pkl'
    log:
        '{session_path}/{session_id}/processed/log/pycontrol_figures.log'
    output:
        event_histogram = report('{session_path}/{session_id}//processed/figures/event_histogram_{session_id}.png', 
                                    caption='report/event_histogram.rst', category='Plots' ),
    script:
        'scripts/02_plot_pycontrol_data.py'

rule export_spike2:
    input:
        pycontrol_dataframe = '{session_path}/{session_id}/processed/df_pycontrol.pkl',
        df_photometry = '{session_path}/{session_id}/processed/df_photometry.nc'
    output:
        spike2_file = '{session_path}/{session_id}/processed/spike2.smrx',
    script:
        'scripts/03_export_spike2.py'


rule import_pyphotometry:
    input:
        pycontrol_dataframe = '{session_path}/{session_id}/processed/df_pycontrol.pkl',
        trial_dataframe = '{session_path}/{session_id}/processed/df_trials.pkl',
        event_dataframe = '{session_path}/{session_id}/processed/df_events_cond.pkl',
        condition_dataframe = '{session_path}/{session_id}/processed/df_conditions.pkl',
        photometry_folder = '{session_path}/{session_id}/photometry'
    output:
        df_photometry = '{session_path}/{session_id}/processed/df_photometry.nc',
    script:
        'scripts/04_import_pyphotometry.py'

rule photometry_figure:
    input:
        df_photometry = '{session_path}/{session_id}/processed/df_photometry.nc',
    output:
        trigger_photo_dir= directory('{session_path}/{session_id}/processed/figures/photometry'),
        done = touch('{session_path}/{session_id}/processed/task.done')
    script:
        'scripts/05_plot_pyphotometry.py'
