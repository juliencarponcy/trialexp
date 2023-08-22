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
from trialexp.process.ephys.spikes_preprocessing import build_evt_fr_xarray
from trialexp.process.ephys.utils import plot_firing_rate
from trialexp.process.group_analysis.plot_utils import style_plot

from workflow.scripts import settings
from scipy.special import kl_div
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

#%% Firing rate map 
sns.set_context('paper')
fig = plot_firing_rate(xr_fr_coord, xr_session, df_pycontrol, ['hold_for_water', 'spout','bar_off','aborted']);
fig.savefig(figures_path/f'firing_map_{probe_name}.png',dpi=200)

# a zoomed in version
fig = plot_firing_rate(xr_fr_coord, xr_session, df_pycontrol,
                       ['hold_for_water', 'spout','bar_off','aborted'],
                       xlim=[180*1000, 240*1000]);

fig.savefig(figures_path/f'firing_map_{probe_name}_1min.png',dpi=200)


#%%
# choose some random event
timestamps = sorted(np.random.choice(xr_fr.time, size=300, replace=False))
trial_nb = np.arange(len(timestamps))
trial_window = (500, 500) # time before and after timestamps to extract
bin_duration = xr_fr.attrs['bin_duration']

da_rand = build_evt_fr_xarray(xr_fr.spikes_FR_session, timestamps, trial_nb, f'spikes_FR.random', 
                                        trial_window, bin_duration)

#%% Use KL divergence to measure the difference between the random and event triggered response

x = da_rand.isel(cluID=0)
y = da.sel(cluID=x.cluID)

x = x.data
y = y.data


# compute the pdf
a = np.concatenate([x.ravel(),y.ravel()])
a = a[~np.isnan(a)]
bins = np.linspace(a.min(), a.max(), 20)
tIdx = 0
kl_d = np.zeros((x.shape[1]))

# for tIdx in range(len(kl_d)):
#     x_pdf = np.histogram(x[:,tIdx], bins=bins)[0]
#     y_pdf = np.histogram(y[:,tIdx], bins=bins)[0]


#     kl_d[tIdx] = kl_div(y_pdf,x_pdf).sum()
    
#%%
from scipy.stats import ttest_ind
var_name = 'spikes_FR.spout'
da = xr_spikes_trials[var_name]

pvalue_ratio = np.zeros((len(da.cluID),))
for i, cluID in enumerate([da.cluID[2]]):
    
    x = da_rand.sel(cluID=cluID).data
    y = da.sel(cluID=cluID).data
        
    pvalue = ttest_ind(x,y,axis=0, nan_policy='omit').pvalue #Note: can be nan in the data if the event cannot be found
    pvalue_ratio[i] = np.mean(pvalue<0.05)

#%%
cluIdx = 1
df2plot = da.isel(cluID=cluIdx).to_dataframe().reset_index()
df2plot['type'] = 'real'
df2plotR = da_rand.isel(cluID=cluIdx).to_dataframe().reset_index()
df2plotR['type'] = 'random'

df2plot = pd.concat([df2plot, df2plotR])
sns.lineplot(df2plot, y=var_name, x='spk_event_time', hue='type', ci=100)

#%%
# calculate the modulation index of neurons
var_name = 'spikes_FR.spout'
da = xr_spikes_trials[var_name]
fr_std = da.std(dim='trial_nb').mean(dim='spk_event_time')
meanCurve = da.mean(dim='trial_nb')
p2p = meanCurve.max(dim='spk_event_time') -  meanCurve.min(dim='spk_event_time')
f2 = p2p/fr_std
f2 = f2.sortby(f2,ascending=False)

#%%
df2plot = da.sel(cluID=f2.cluID[0]).to_dataframe().reset_index()
df2plot['type'] = 'real'
df2plotR = da_rand.sel(cluID=f2.cluID[0]).to_dataframe().reset_index()
df2plotR['type'] = 'random'

df2plot = pd.concat([df2plot, df2plotR])


sns.lineplot(df2plot, y=var_name, x='spk_event_time', hue='type',n_boot=100)

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