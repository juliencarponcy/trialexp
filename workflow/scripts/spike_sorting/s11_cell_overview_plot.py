#%%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm
import xarray as xr
from matplotlib import gridspec
from snakehelper.SnakeIOHelper import getSnake
from trialexp.process.ephys.spikes_preprocessing import build_evt_fr_xarray
from trialexp.process.ephys.utils import *
from trialexp.process.group_analysis.plot_utils import style_plot
from joblib import Parallel, delayed

from trialexp import config 

#%% Load inputs


(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [config.debug_folder + r'/processed/cell_overview.done'],
  'cell_overview_plot')


# %% Define variables and folders


figures_path = Path(soutput.figures_path)

#%% Opening datasets
# load_dataset will load the file into memory and automatically close it
# open_dataset will not load everything into memory at once
# load_dataset is better for analysis pipeline as mulitple script may work on the same file

xr_spikes_trials = xr.load_dataset(Path(sinput.xr_spikes_trials)) 
xr_fr = xr.load_dataset(Path(sinput.xr_spikes_fr))
session_ID = xr_spikes_trials.attrs['session_id']
df_pycontrol = pd.read_pickle(sinput.pycontrol_dataframe)

# only include good units
good_idx = (xr_spikes_trials.ks_labels=='good')
xr_spikes_trials = xr_spikes_trials.sel(cluID=good_idx)
xr_fr = xr_fr.sel(cluID=good_idx)

#%% Overall firing rate plot
# need to get the channel map and plot them in the correct depth
xr_fr_coord = xr_fr.merge(xr_spikes_trials[['ks_chan_pos_x', 'ks_chan_pos_y','probe_name']])
xr_fr_coord = xr_fr_coord.sortby('ks_chan_pos_y')

# netCDF flatten length 1 list automatically, 
probe_names = np.unique(xr_fr_coord['probe_name'].data)

for probe_name in probe_names:
    cluID_probe = [probe_name in id for id in xr_fr_coord.cluID.data]
    pos_y = xr_fr_coord.ks_chan_pos_y.sel(cluID=cluID_probe)
     # plot distribution of cell in depth
    fig, ax= plt.subplots(1,1,figsize=(4,4))
    style_plot()
    sns.histplot(y=pos_y,bins=50,ax=ax)
    ax.set(ylabel='Depth (um)', title=f'{probe_name}')
    ax.set_ylim([0,4000])
    fig.savefig(figures_path/f'cluster_depth_distribution_{probe_name}.png',dpi=200)

#%% Align the photometry time to the firing rate time

xr_session = xr.load_dataset(Path(sinput.xr_session))
xr_session = xr_session.interp(time=xr_fr_coord.time)

#%% Firing rate map 
sns.set_context('paper')

for probe_name in probe_names:
    cluID_probe = [probe_name in id for id in xr_fr_coord.cluID.data]

    xr_fr_coord_probe = xr_fr_coord.sel(cluID=cluID_probe)
    fig = plot_firing_rate(xr_fr_coord_probe, xr_session, df_pycontrol, ['hold_for_water', 'spout','bar_off','aborted']);
    fig.savefig(figures_path/f'firing_map_{probe_name}.png',dpi=200)

    # a zoomed in version
    fig = plot_firing_rate(xr_fr_coord_probe, xr_session, df_pycontrol,
                        ['hold_for_water', 'spout','bar_off','aborted'],
                        xlim=[10*60*1000, 12*60*1000]);

    fig.savefig(figures_path/f'firing_map_{probe_name}_2min.png',dpi=200)
    
    
# %%
