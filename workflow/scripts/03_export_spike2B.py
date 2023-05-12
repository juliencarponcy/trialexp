'''
Export event data to spike2
'''
#%%
import pandas as pd 
from trialexp.process.pycontrol.utils import export_session
from snakehelper.SnakeIOHelper import getSnake
# from workflow.scripts import settings
from re import match
from pathlib import Path
from trialexp.process.pyphotometry.utils import *

import os


#%% target sessions

dir_by_sessions = r"\\ettin\Magill_lab\Julien\Data\head-fixed\by_sessions"

def join_task_session(taskname, sessionnames: list):
    return [os.path.join(dir_by_sessions, taskname, ssn) for ssn in sessionnames]

task1 = join_task_session('reaching_go_spout_bar_nov22', [
    'kms058-2023-03-24-151254',
    'kms058-2023-03-25-184034',
    'kms062-2023-02-21-103400',
    'kms062-2023-02-22-150828',
    'kms063-2023-04-09-183115',
    'kms063-2023-04-10-194331',
    'kms064-2023-02-13-104949',
    'kms064-2023-02-15-104438',
    'kms064-2023-02-16-103424',
    'RE602-2023-03-22-121414'])

task2 = join_task_session('reaching_go_spout_bar_dual_dec22', [
    'JC316L-2022-12-07-163252',
    'JC316L-2022-12-08-143046'])


debug_folders = task1 + task2

#%% Loop

for debug_folder in debug_folders:
    
    (sinput, soutput) = getSnake(locals(), 'workflow/pycontrol.smk',
        [debug_folder +'/processed/spike2B.smrx'],
        'export_spike2B')

    # Photometry dict

    #fn = glob(sinput.photometry_folder+'\*.ppd')[0]
    fn = list(Path(sinput.photometry_folder).glob('*.ppd'))
    if fn == []:
        data_photometry = None    
    else:
        fn = fn[0]
        data_photometry = import_ppd(fn)

        data_photometry = denoise_filter(data_photometry)
        data_photometry = motion_correction(data_photometry)
        data_photometry = compute_df_over_f(data_photometry, low_pass_cutoff=0.001)


    # no down-sampling here

    # Load data
    df_pycontrol = pd.read_pickle(sinput.pycontrol_dataframe)

    pycontrol_time = df_pycontrol[df_pycontrol.name == 'rsync'].time

    # assuming just one txt file
    pycontrol_txt = list(Path(sinput.pycontrol_folder).glob('*.txt'))

    with open(pycontrol_txt[0], 'r') as f:
        all_lines = [line.strip() for line in f.readlines() if line.strip()]

    count = 0
    print_lines = []
    while count < len(all_lines):
        # all_lines[count][0] == 'P'
        if bool(match('P\s\d+\s', all_lines[count])):
            print_lines.append(all_lines[count][2:])
            count += 1
            while (count < len(all_lines)) and not (bool(match('[PVD]\s\d+\s', all_lines[count]))):
                print_lines[-1] = print_lines[-1] + \
                    "\n" + all_lines[count]
                count += 1
        else:
            count += 1

    v_lines = [line[2:] for line in all_lines if line[0] == 'V']


    #
    if fn == []:
        photometry_times_pyc = None
    else:
        photometry_aligner = Rsync_aligner(pycontrol_time, data_photometry['pulse_times_2'])
        photometry_times_pyc = photometry_aligner.B_to_A(data_photometry['time'])

    #remove all state change event
    df_pycontrol = df_pycontrol.dropna(subset='name')
    df2plot = df_pycontrol[df_pycontrol.type == 'event']
    # state is handled separately with export_state, whereas parameters are handled vchange_to_text

    keys = df2plot.name.unique()

    photometry_keys =  ['analog_1', 'analog_2',  'analog_1_filt', 'analog_2_filt',
                    'analog_1_est_motion', 'analog_1_corrected', 'analog_1_baseline_fluo', 
                    'analog_1_df_over_f']

    #
    export_session(df_pycontrol, keys, 
        data_photometry = data_photometry,
        photometry_times_pyc = photometry_times_pyc,
        photometry_keys = photometry_keys,
        print_lines = print_lines,
        v_lines = v_lines,
        smrx_filename=soutput.spike2_file)
