#%%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib import gridspec
from snakehelper.SnakeIOHelper import getSnake

from workflow.scripts import settings

#%% Load inputs


(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [settings.debug_folder + r'/processed/cell_trial_responses.done'],
  'cell_trial_responses_plot')


# %% Define variables and folders
# xr_session_path = Path(sinput.xr_session)
# xr_spikes_trials_path = Path(sinput.xr_spikes_trials_anat)
# xr_spikes_trials_phases_path = Path(sinput.xr_spikes_trials_phases_anat)
# xr_spikes_session_path = Path(sinput.xr_spikes_full_session_anat)

# figures_path = xr_spikes_trials_path.parent / 'figures' / 'ephys'
figures_path = Path(soutput.figures_path)
# session_path = xr_spikes_session_path.parent.parent
# session_ID = session_path.stem

#%% Opening datasets
# xr_session = xr.open_dataset(Path(xr_session_path)
xr_spikes_trials = xr.open_dataset(Path(sinput.xr_spikes_trials))
session_ID = xr_spikes_trials.attrs['session_ID']


#%% Define trials of interest

# Plot the distrubtion of cluster in each brain regions
# trial_types_to_plot = ('water by spout')

# structs, struct_count = np.unique(xr_spikes_trials_phases['brain_region_short'].values.astype(str), return_counts=True)

# # create grid for different subplots
# spec = gridspec.GridSpec(ncols=2, nrows=1,
#                          width_ratios=[2, 1], wspace=0.1)

# # create a figure
# fig = plt.figure(figsize=(20,5))
# # to change size of subplot's
# # set height of each subplot as 8
# # fig.set_figheight(8)
 
# # set width of each subplot as 8
# fig.set_figwidth
# axes = list()
# axes.append(fig.add_subplot(spec[0]))
# plt.suptitle(f'Clusters anatomical distribution: {session_ID}')

# sns.barplot(x=structs, y=struct_count, ax=axes[0])
# axes[0].set_xlabel('Brain structure acronym')
# axes[0].set_ylabel('Number of clusters')

# axes.append(fig.add_subplot(spec[1]))

# sns.histplot(data=xr_spikes_trials_phases[['probe_name_x','anat_depth']].to_dataframe(),x='anat_depth', hue='probe_name_x', kde=True)
# axes[1].set_xlabel('Anatomical depth (micrometers)')
# axes[1].set_ylabel('Number of clusters')

# fig.savefig(figures_path / 'cluster_anat_distrib.png')
# %%