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
from trialexp.process.ephys.utils import combine2dataframe, compare_fr_with_random, compute_tuning_prop, get_max_sig_region_size, get_pvalue_random_events, plot_firing_rate
from trialexp.process.group_analysis.plot_utils import style_plot
from joblib import Parallel, delayed

from workflow.scripts import settings
#%% Load inputs


(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [settings.debug_folder + r'/processed/cell_trial_responses.done'],
  'cell_trial_responses_plot')


# %% Define variables and folders


figures_path = Path(soutput.figures_path)

#%% Opening datasets
# load_dataset will load the file into memory and automatically close it
# open_dataset will not load everything into memory at once
# load_dataset is better for analysis pipeline as mulitple script may work on the same file

xr_spikes_trials = xr.load_dataset(Path(sinput.xr_spikes_trials)) 
xr_fr = xr.load_dataset(Path(sinput.xr_spikes_fr))
session_ID = xr_spikes_trials.attrs['session_ID']
df_pycontrol = pd.read_pickle(sinput.pycontrol_dataframe)


#%% Overall firing rate plot
# need to get the channel map and plot them in the correct depth
waveform_chan = xr_spikes_trials.maxWaveformCh.to_dataframe()
chanCoords_x = xr_spikes_trials.attrs['chanCoords_x']
chanCoords_y = xr_spikes_trials.attrs['chanCoords_y']
waveform_chan['pos_x'] = chanCoords_x[waveform_chan.maxWaveformCh.astype(int)]
waveform_chan['pos_y'] = chanCoords_y[waveform_chan.maxWaveformCh.astype(int)]
xr_fr_coord = xr_fr.merge(waveform_chan)
xr_fr_coord.attrs['probe_names'] = xr_spikes_trials.attrs['probe_names']
xr_fr_coord = xr_fr_coord.sortby('pos_y')

# netCDF flatten length 1 list automatically, 
probe_names = xr_fr_coord.attrs['probe_names']
probe_names = [probe_names] if type(probe_names) is str else probe_names

for probe_name in probe_names:
    cluID_probe = [probe_name in id for id in xr_fr_coord.cluID.data]
    pos_y = xr_fr_coord.pos_y.sel(cluID=cluID_probe)
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
                        xlim=[180*1000, 240*1000]);

    fig.savefig(figures_path/f'firing_map_{probe_name}_1min.png',dpi=200)

#%% compute tuning prop
var2plot = [x for x in xr_spikes_trials if x.startswith('spikes_FR')]
bin_duration = xr_fr.attrs['bin_duration']
trial_window = xr_spikes_trials.attrs['trial_window']

#%%
style_plot()

def draw_response_curve(var_name):
    print(f'Drawing the response curve for {var_name}')
    da = xr_spikes_trials[var_name]

    da_rand, pvalues, pvalue_ratio = get_pvalue_random_events(da, xr_fr, trial_window, bin_duration)
    
    max_region_size = get_max_sig_region_size(pvalues, p_threshold=0.05)

    max_region_size[(max_region_size>100)|(max_region_size<10)] = 0 #only focus on region between 1s and 100ms
    sortIdx = np.argsort(max_region_size)[::-1]
    cluID_sorted = da.cluID[sortIdx]
    pvalues_sorted = pvalues[sortIdx,:]


    fig, ax = plt.subplots(4,4,dpi=200, figsize=(4*3,4*3))

    for cellIdx2plot in range(len(ax.flat)):
        compare_fr_with_random(da, da_rand, 
                            cluID_sorted[cellIdx2plot], pvalues_sorted[cellIdx2plot,:],
                            ax=ax.flat[cellIdx2plot])

    fig.tight_layout()
    fig.savefig(figures_path/f'event_response_{var_name}.png',dpi=200)
    
    return {var_name:{
        'pvalues': pvalues.tolist(),
        'pvalue_ratio': pvalue_ratio,
        'max_region_size':max_region_size
    }}
    
    #TODO also draw the heatmaps
    #TODO: save the resutls of the significance analysis to another dataframe
    
# use joblib to speed up the processing
# generator expression
result = Parallel(n_jobs=len(var2plot))(delayed(draw_response_curve)(var_name) for var_name in var2plot)
df_tuning = combine2dataframe(result)
df_tuning['cluID'] = xr_spikes_trials.cluID

# %%
# Extract additional cell characteristics
df_chan_coords  = xr_fr_coord[['pos_x','pos_y','maxWaveformCh']].to_dataframe()

# save
df_cell_prop = df_tuning.merge(df_chan_coords, on='cluID')
df_cell_prop.to_pickle(Path(soutput.df_cell_prop))
