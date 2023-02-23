from glob import glob

rule all:
    input: expand('{sessions}/processed/task.done', sessions =glob(r'Z:\Teris\ASAP\expt_sessions\*'))

rule process_pycontrol:
    input:
        session_path = '{session_path}'
    output:
        event_dataframe = '{session_path}/processed/df_events_cond.pkl',
        condition_dataframe = '{session_path}/processed/df_conditions.pkl'
    log:
        '{session_path}/processed/log/process_pycontrol.log',
    script:
        'scripts/01_process_pycontrol.py'

rule pycontrol_figures:
    input:
        event_dataframe = '{session_path}/processed/df_events_cond.pkl'
    output:
        event_histogram = '{session_path}/processed/figures/event_histogram.png',
        done = touch('{session_path}/processed/task.done')
    script:
        'scripts/02_plot_pycontrol_data.py'
