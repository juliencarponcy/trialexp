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
# xr_session= xr.open_dataset(Path(sinput.xr_session))
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
xr_fr = xr.open_dataset(Path(sinput.xr_spikes_fr))
session_ID = xr_spikes_trials.attrs['session_ID']
df_pycontrol = pd.read_pickle(sinput.pycontrol_dataframe)

#%%

onset = xr_spikes_trials['spikes_FR.trial_onset']
df2plot = onset.to_dataframe()

#%% Overall firing rate plot
# need to get the channel map and plot them in the correct depth

# find out the location of each cluster
# the total shank length of 1.0 NXP probe is 10mm
probe_name = 'ProbeA'
probe = xr_spikes_trials.sel(probe_name=probe_name)
waveform_chan = probe.maxWaveformCh.to_dataframe()
chanCoords_x = probe.attrs['chanCoords_x']
chanCoords_y = probe.attrs['chanCoords_y']
waveform_chan['pos_x'] = chanCoords_x[waveform_chan.maxWaveformCh]
waveform_chan['pos_y'] = chanCoords_y[waveform_chan.maxWaveformCh]

xr_fr_coord = xr_fr.merge(waveform_chan)

# plot distribution of cell in depth
a = sns.histplot(waveform_chan, y='pos_y',bins=50)
a.set(ylabel='Depth (um)', title='ProbeA')
plt.savefig(figures_path/f'cluster_depth_distribution_{probe_name}.png',dpi=200)


#%%

xr_session = xr.open_dataset(Path(sinput.xr_session))
xr_session = xr_session.interp(time=xr_fr_coord.time)

#%%
def plot_firing_rate(xr_fr_coord, xr_session, df_pycontrol, events2plot):
    spike_rates = xr_fr_coord.spikes_zFR_session.data
    
    fig,ax = plt.subplots(3,1,figsize=(20,15),dpi=200, sharex=True)
    
    # firing rate map
    image = ax[0].imshow(spike_rates.T, vmax=2, vmin=-2,cmap='icefire')
    ax[0].set_aspect('auto')
    
    yticks = np.arange(0, spike_rates.shape[1],50 ) #where we want to show the
    
    
    ax[0].set_yticks(yticks)
    ax[0].set_yticklabels(xr_fr_coord.pos_y.data[yticks]); #the cooresponding label for the tick
    ax[0].invert_yaxis()
    
    xticks = np.linspace(0,spike_rates.shape[0]-10,10).astype(int)
    ax[0].set_xticks(xticks)
    xticklabels = (xr_fr_coord.time[xticks].data/1000).astype(int)
    ax[0].set_xticklabels(xticklabels)

    
    ax[0].set_ylabel('Distance from tip (um)')
    ax[0].set_xlabel('Time (s)')

    # also plot the important pycontrol events
    
    events2plot = df_pycontrol[df_pycontrol.name.isin(events2plot)]

    ## Event
    evt_colours =['r','g','b','w']
    # Note: the time coordinate of the firing map corresponds to the time bins
    bin_duration = xr_fr_coord.attrs['bin_duration']
    for i, event in enumerate(events2plot.name.unique()):
        evt_time = events2plot[events2plot.name==event].time
        evt_time_idx = [np.searchsorted(xr_fr_coord.time, t) for t in evt_time]
        # evt_time = evt_time/bin_duration
        ax[1].eventplot(evt_time_idx, lineoffsets=80+20*i, linelengths=20,label=event, color=evt_colours[i])
    
    ax[1].legend(loc='upper left', prop = { "size": 7 }, ncol=4)
    
    
    ax[2].plot(xr_session.zscored_df_over_f.data)
    
    cbar_ax = fig.add_axes([0.95, 0.55, 0.02, 0.35]) 
    fig.colorbar(image, cax=cbar_ax)
    
    return fig

fig = plot_firing_rate(xr_fr_coord, xr_session, df_pycontrol, ['hold_for_water', 'spout','bar_off','aborted']);
# fig.savefig(figures_path/f'firing_map_{probe_name}.png',dpi=200)

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