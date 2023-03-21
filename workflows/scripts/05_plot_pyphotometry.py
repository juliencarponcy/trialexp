
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
#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflows/spout_bar_nov22.smk',
 ['//ettin/Magill_Lab/Julien/Data/head-fixed/_Other/test_folder/by_session_folder/JC316L-2022-12-09-171925/processed/figures/photometry'],
#   ['Z:/Teris/ASAP/expt_sessions/kms064-2023-02-08-100449/processed/figures/photometry'],
  'photometry_figure')


#%%
dataset = xr.open_dataset(sinput.df_photometry)
%timeit dataset = xr.open_dataset(sinput.df_photometry)


#%%
figure_dir = soutput.trigger_photo_dir

df2plot = dataset[['rel_time_hold_for_water','analog_1_df_over_f']].to_dataframe()
df2plot.rel_time_hold_for_water = df2plot.rel_time_hold_for_water//10*10 # time windows: 100ms

fig, ax = plt.subplots(1,1,dpi=300, figsize=(6,6))
sns.boxplot(x='rel_time_hold_for_water', y='analog_1_df_over_f', data=df2plot, ax=ax)
ax.set(xlabel='Time from hold_for_water', ylabel='Delta F/F')
fig.savefig(os.path.join(figure_dir, 'rel_time_hold_for_water.png'), dpi=300, bbox_inches='tight')

#%%