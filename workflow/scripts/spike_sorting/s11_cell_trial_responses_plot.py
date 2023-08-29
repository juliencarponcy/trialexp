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
from trialexp.process.ephys.utils import compare_fr_with_random, get_pvalue_random_events, plot_firing_rate
from trialexp.process.group_analysis.plot_utils import style_plot
from joblib import Parallel, delayed

from workflow.scripts import settings
from scipy.special import kl_div
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

# find out the location of each cluster
# the total shank length of 1.0 NXP probe is 10mm
for probe_name in xr_spikes_trials.probe_name.data:
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


#%% Align the photometry time to the firing rate time

xr_session = xr.load_dataset(Path(sinput.xr_session))
xr_session = xr_session.interp(time=xr_fr_coord.time)

#%% Firing rate map 
sns.set_context('paper')

for probe_name in np.unique(xr_fr_coord.probe_name.data):
    xr_fr_coord_probe = xr_fr_coord.sel(cluID=(xr_fr_coord.probe_name==probe_name))
    fig = plot_firing_rate(xr_fr_coord_probe, xr_session, df_pycontrol, ['hold_for_water', 'spout','bar_off','aborted']);
    fig.savefig(figures_path/f'firing_map_{probe_name}.png',dpi=200)

    # a zoomed in version
    fig = plot_firing_rate(xr_fr_coord_probe, xr_session, df_pycontrol,
                        ['hold_for_water', 'spout','bar_off','aborted'],
                        xlim=[180*1000, 240*1000]);

    fig.savefig(figures_path/f'firing_map_{probe_name}_1min.png',dpi=200)

#%%
var2plot = [x for x in xr_spikes_trials if x.startswith('spikes_FR')]
bin_duration = xr_fr.attrs['bin_duration']
trial_window = xr_spikes_trials.attrs['trial_window']
style_plot()

def draw_response_curve(var_name):
    print(f'Drawing the response curve for {var_name}')
    da = xr_spikes_trials[var_name]

    da_rand, pvalues, pvalue_ratio = get_pvalue_random_events(da, xr_fr, trial_window, bin_duration)

    # sort the cluID according to the pvalue_ratio descendingly
    pvalue_ratio[(pvalue_ratio<0.2)|(pvalue_ratio>0.8)] = 0 #only focus on ratio between 0.2 and 0.8
    sortIdx = np.argsort(pvalue_ratio)[::-1]
    pvalue_ratio_sorted = pvalue_ratio[sortIdx]
    cluID_sorted = da.cluID[sortIdx]
    pvalues_sorted = pvalues[sortIdx,:]


    fig, ax = plt.subplots(4,4,dpi=200, figsize=(4*3,4*3))

    for cellIdx2plot in range(len(ax.flat)):
        compare_fr_with_random(da, da_rand, 
                            cluID_sorted[cellIdx2plot], pvalues_sorted[cellIdx2plot,:],
                            ax=ax.flat[cellIdx2plot])

    fig.tight_layout()
    fig.savefig(figures_path/f'event_response_{var_name}.png',dpi=200)
    
    #TODO also draw the heatmaps
    
# use joblib to speed up the processing
# generator expression
Parallel(n_jobs=len(var2plot))(delayed(draw_response_curve)(var_name) for var_name in var2plot)



# %%
