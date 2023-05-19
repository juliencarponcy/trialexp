'''
Export event data to spike2
'''
#%%
import pandas as pd 
from trialexp.process.pycontrol.utils import export_session
from snakehelper.SnakeIOHelper import getSnake
from workflow.scripts import settings
from re import match
from pathlib import Path
from trialexp.process.pyphotometry.utils import *
import os
#%%

(sinput, soutput) = getSnake(locals(), 'workflow/pycontrol.smk',
    [settings.debug_folder +'/processed/spike2_export.done'],
    'export_spike2')

#%% Photometry dict

#fn = glob(sinput.photometry_folder+'\*.ppd')[0]
fn = list(Path(sinput.photometry_folder).glob('*.ppd'))
if fn == []:
    data_photometry = None    
else:
    fn = fn[0]
    data_photometry = import_ppd(fn)

    data_photometry = denoise_filter(data_photometry)
    # data_photometry = motion_correction(data_photometry)
    data_photometry = motion_correction_win(data_photometry)
    data_photometry = compute_df_over_f(data_photometry, low_pass_cutoff=0.001)


# no down-sampling here

#%% Load data
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


#%%
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
                  'analog_1_df_over_f','analog_2_baseline_fluo', 'analog_2_df_over_f']

#%%
'''
sonpy holds a reference of the smrx file in memory, this will result in resource busy error
when it is currently opened by someone else, this will result in snakemake error which cannot be skpped. We need to handle 
the exception ourselves here
'''

spike2_path = Path(soutput.spike2_export_done).parent/'spike2.smrx'

try:
    if spike2_path.exists():
        os.remove(spike2_path)
except OSError:
    logging.warning(f'Warning: smrx file is busy. Skipping {spike2_path}')
else:
    export_session(df_pycontrol, keys, 
        data_photometry = data_photometry,
        photometry_times_pyc = photometry_times_pyc,
        photometry_keys = photometry_keys,
        print_lines = print_lines,
        v_lines = v_lines,
        smrx_filename=str(spike2_path))
    