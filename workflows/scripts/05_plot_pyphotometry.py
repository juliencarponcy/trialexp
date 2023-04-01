
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
  [settings.debug_folder + '/processed/log/photometry.done'],
  'photometry_figure')


#%%
xr_session = xr.open_dataset(sinput.xr_session)

figure_dir = soutput.trigger_photo_dir

#%% plot all event-related data
for k in xr_session.data_vars.keys():
    da = xr_session[k]
    if 'event_time' in da.coords:
        df2plot = xr_session[[k,'success']].to_dataframe().reset_index()
        
        fig, ax = plt.subplots(1,1,dpi=300, figsize=(6,6))

        ax = sns.lineplot(x='event_time',hue='success', y=k, data=df2plot)
        ax.set(xlabel=k, ylabel='Delta F/F')

        fig.savefig(os.path.join(figure_dir, f'{k}.png'), dpi=300, bbox_inches='tight')

#%%
da = xr_session[['hold_for_water_analog_1_df_over_f', 'success']]

df2plot = da.to_dataframe()

#%%
sns.lineplot(x='event_time', y='hold_for_water_analog_1_df_over_f', data=df2plot)
# %%
xr_session.close()