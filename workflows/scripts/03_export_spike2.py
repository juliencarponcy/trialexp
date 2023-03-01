'''
Export event data to spike2
'''
#%%
import pandas as pd 
from trialexp.process.pycontrol.utils import export_session
from snakehelper.SnakeIOHelper import getSnake


#%%

(sinput, soutput) = getSnake(locals(), 'workflows/spout_bar_nov22.smk',
#   ['Z:/Julien/Data/head-fixed/_Other/test_folder/by_session_folder/JC316L-2022-12-07-163252\processed/spike2.smrx'],
    ['Z:\Teris\ASAP\expt_sessions\kms064-2023-02-08-100449\processed\spike2.smrx'],
  'export_spike2')

#%% Load data
df_pycontrol = pd.read_pickle(sinput.pycontrol_dataframe)


#%%

#remove all state change event
df_pycontrol = df_pycontrol.dropna(subset='name')
df2plot = df_pycontrol[df_pycontrol.type != 'state']
keys = df2plot.name.unique()

export_session(df_pycontrol, keys,
             smrx_filename=soutput.spike2_file)
# %%
