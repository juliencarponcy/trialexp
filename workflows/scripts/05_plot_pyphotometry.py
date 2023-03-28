
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
#  ['//ettin/Magill_Lab/Julien/Data/head-fixed/_Other/test_folder/by_session_folder/JC316L-2022-12-09-171925/processed/figures/photometry'],
  ['Z:/Teris/ASAP/expt_sessions/kms063-2023-02-27-164426/processed/log/photometry.done'],
  'photometry_figure')


#%%
xr_session = xr.open_dataset(sinput.xr_session)

figure_dir = soutput.trigger_photo_dir

#%% plot all relative time 
for k in xr_session.data_vars.keys():
    if 'rel_time' in k:
        fig, ax = plt.subplots(1,1,dpi=300, figsize=(6,6))

        ax = sns.lineplot(x=k,hue='success',
                    y='analog_1_df_over_f', data=xr_session)
        ax.set(xlabel=k, ylabel='Delta F/F')

        fig.savefig(os.path.join(figure_dir, f'{k}.png'), dpi=300, bbox_inches='tight')


# %%
xr_session.close()