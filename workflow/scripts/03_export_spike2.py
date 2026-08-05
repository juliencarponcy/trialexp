'''
Export event data to spike2
'''
#%%
import pandas as pd 
from trialexp.process.pycontrol.utils import extract_v_line_v2, export_session_v2, get_sync_time
from snakehelper.SnakeIOHelper import getSnake
from trialexp import config
from re import match
from pathlib import Path
from trialexp.process.pyphotometry.utils import *
import os
#%%

(sinput, soutput) = getSnake(locals(), 'workflow/pycontrol.smk',
    [config.debug_folder +'/processed/spike2_export.done'],
    'export_spike2')

#%% Load data
df_pycontrol = pd.read_pickle(sinput.pycontrol_dataframe)
df_variable, df_print = extract_v_line_v2(df_pycontrol)
df_dataformat = pd.read_csv('params/data_format.csv')

#%% Photometry dict

fn = list(Path(sinput.photometry_folder).glob('*.ppd'))

if fn == []:
    data_photometry = None    
else:
    fn = fn[0]
    data_photometry = import_ppd_auto(fn)
    data_photometry = preprocess_photometry(data_photometry, df_pycontrol)
    
#%%
if fn == []:
    photometry_times_pyc = None
    photometry_keys= None
else:
    pycontrol_time = get_sync_time(df_pycontrol)
    photometry_aligner = Rsync_aligner(pycontrol_time, data_photometry['pulse_times_2'])
    photometry_times_pyc = photometry_aligner.B_to_A(data_photometry['time'])
    photometry_keys =  [k for k in data_photometry.keys() if k.startswith('analog') or k.startswith('bleedthrough')]

#%%
#remove all state change event
# df_pycontrol = df_pycontrol.dropna(subset='name')
df2plot = df_pycontrol[(df_pycontrol.type == 'event') &
                        (df_pycontrol.subtype != 'sync')]
# state is handled separately with export_state, whereas parameters are handled vchange_to_text

keys = df2plot.content.unique()

#%%
'''
sonpy holds a reference of the smrx file in memory, this will result in resource busy error
when it is currently opened by someone else, this will result in snakemake error which cannot be skpped. We need to handle 
the exception ourselves here
'''

spike2_path = Path(soutput.spike2_export_done).parent/'spike2.smrx'
import tempfile

# tempfile = tempfile.NamedTemporaryFile()
try:
    if spike2_path.exists():
        os.remove(spike2_path)
except OSError:
    logging.warning(f'Warning: smrx file is busy. Skipping {spike2_path}')
else:
    df_export = df_pycontrol[df_pycontrol.type!='variable']
    export_session_v2(df_export, keys, 
        data_photometry = data_photometry,
        photometry_times_pyc = photometry_times_pyc,
        photometry_keys = photometry_keys,
        df_print = df_print,
        df_variable = df_variable,
        smrx_filename=str(spike2_path))
    

# %%
