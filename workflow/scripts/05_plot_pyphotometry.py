
'''
Plotting of photometry data
'''
#%%
from snakehelper.SnakeIOHelper import getSnake
from trialexp.process.pyphotometry.plotting_utils import annotate_trial_number, plot_and_handler_error
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
    })

sns.set_context('paper')


for k in xr_session.data_vars.keys():
    da = xr_session[k]
    if 'event_time' in da.coords: # choose data varialbes that are event related
        df2plot = xr_session[[k,'trial_outcome']].to_dataframe().reset_index()
        trial_outcome = df2plot['trial_outcome'].unique()
        
        g = sns.FacetGrid(df2plot, col='trial_outcome', col_wrap=3, hue='trial_outcome')
        g.map_dataframe(plot_and_handler_error, sns.lineplot, x='event_time', y=k)
        g.map_dataframe(annotate_trial_number)
        g.set_titles(col_template='{col_name}')
        g.set_xlabels('Time (ms)')
            
        g.figure.savefig(os.path.join(figure_dir, f'{k}.png'), dpi=300, bbox_inches='tight')

xr_session.close()
# %%
