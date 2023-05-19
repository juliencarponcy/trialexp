
'''
Plotting of photometry data
'''
#%%
from snakehelper.SnakeIOHelper import getSnake
from trialexp.process.pyphotometry.utils import *
from glob import glob
import xarray as xr
from trialexp.utils.rsync import *
import pandas as pd 
from scipy.interpolate import interp1d
import seaborn as sns 
import numpy as np
import os
from workflow.scripts import settings

#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflow/pycontrol.smk',
  [settings.debug_folder + '/processed/log/photometry.done'],
  'photometry_figure')


#%%
xr_session = xr.open_dataset(sinput.xr_session)

figure_dir = soutput.trigger_photo_dir

#%% plot all event-related data

sns.set_style("white", {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "font.family": ["Arial"]
    })

for k in xr_session.data_vars.keys():
    da = xr_session[k]
    if 'event_time' in da.coords:
        df2plot = xr_session[[k,'success']].to_dataframe().reset_index()
        
        if not all(df2plot[k].isna()): #make sure data are correct
          
          fig, ax = plt.subplots(1,1,dpi=300, figsize=(6,6))

          ax = sns.lineplot(x='event_time',hue='success', y=k, data=df2plot)
          ax.set(ylabel=k, xlabel='Time (ms)')
          ax.legend(frameon=False)

          fig.savefig(os.path.join(figure_dir, f'{k}.png'), dpi=300, bbox_inches='tight')

xr_session.close()
# %%
