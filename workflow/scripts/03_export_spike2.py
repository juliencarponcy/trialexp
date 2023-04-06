'''
Export event data to spike2
'''
#%%
import pandas as pd 
from trialexp.process.pycontrol.utils import export_session
from snakehelper.SnakeIOHelper import getSnake
from workflow.scripts import settings
import os 
#%%

(sinput, soutput) = getSnake(locals(), 'workflow/Snakefile',
    [os.path.join(settings.debug_folder,'processed','spike2.smrx')],
  'export_spike2')

#%% Load data
df_pycontrol = pd.read_pickle(sinput.pycontrol_dataframe)

#remove all state change event
df_pycontrol = df_pycontrol.dropna(subset='name')
df2plot = df_pycontrol[df_pycontrol.type != 'state']
keys = df2plot.name.unique()


#%% Check if we need to export photometry data too
if soutput.xr_session is not None:
  print('test')


#%%
export_session(df_pycontrol, keys,
             smrx_filename=soutput.spike2_file)
# %%
