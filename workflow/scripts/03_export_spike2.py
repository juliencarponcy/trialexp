'''
Export event data to spike2
'''
#%%
import pandas as pd 
from trialexp.process.pycontrol.utils import export_session
from snakehelper.SnakeIOHelper import getSnake
from workflow.scripts import settings

#%%

(sinput, soutput) = getSnake(locals(), 'workflow/pycontrol.smk',
    [settings.debug_folder +'/processed/spike2.smrx'],
  'export_spike2')

#%% Load data
df_pycontrol = pd.read_pickle(sinput.pycontrol_dataframe)

#remove all state change event
df_pycontrol = df_pycontrol.dropna(subset='name')
df2plot = df_pycontrol[df_pycontrol.type != 'state']
keys = df2plot.name.unique()

export_session(df_pycontrol, keys,
             smrx_filename=soutput.spike2_file)
# %%