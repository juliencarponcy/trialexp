
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
from workflows.scripts import settings

#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflows/spout_bar_nov22.smk',
  [os.path.join(settings.debug_folder,'processed','log','photometry.done')],
  'photometry_figure')

#%%
xr_session = xr.open_dataset(sinput.xr_session)

figure_dir = soutput.trigger_photo_dir

#%% plot all event-related data
for k in xr_session.data_vars.keys():
    da = xr_session[k]
    if 'event_time' in da.coords:
        df2plot = xr_session[[k,'success']].to_dataframe().reset_index()
        
        #remove invalid data, sometimes an event is not found and everything will be NaN
        df2plot = df2plot.dropna(subset=k)
        
        if len(df2plot)>0:
          fig, ax = plt.subplots(1,1,dpi=300, figsize=(6,6))
          ax = sns.lineplot(x='event_time',hue='success', y=k, data=df2plot)
          ax.set(ylabel=k, xlabel='Delta F/F')

          fig.savefig(os.path.join(figure_dir, f'{k}.png'), dpi=300, bbox_inches='tight')

xr_session.close()