'''
Script to perform correlations between spikes and photometry for each event type

'''
#%%
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import xarray as xr
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from snakehelper.SnakeIOHelper import getSnake

from trialexp.process.ephys.utils import calculate_pearson_lags
from trialexp.process.group_analysis.plot_utils import style_plot
from trialexp.process.pyphotometry.utils import *
from trialexp import config
import itertools
from tqdm.auto import tqdm
from loguru import logger
import seaborn as sns
from trialexp.process.ephys.photom_correlation import plot_extrem_corr, get_corr_spatial_distribution, analyze_correlation, plot_extrem_corr_timewarp

#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [config.debug_folder + r'/processed/xr_corr.nc'],
  'session_correlations')

#%% Path definitions
verbose = True
root_path = Path(config.SESSION_ROOT_DIR)

#%% Loading files
xr_spike_trial = xr.open_dataset(sinput.xr_spikes_trials) #file is huge, use lazy loading
session_root_path = Path(sinput.xr_spike_fr).parent
xr_session = xr.load_dataset(session_root_path/'xr_session.nc')
xr_spike_fr_interp = xr_spike_trial.interp(spk_event_time=xr_session.event_time)
df_metrics = pd.read_pickle(sinput.df_quality_metrics)


#%% Calculate cross-correlation between unit activity and photometry signals
'''
Some Neurpixels only recoding do not have photometry data
'''
photom_vars = ['_zscored_df_over_f', '_zscored_df_over_f_analog_2']
var = [v.replace(photom_vars[0],'') for v in xr_session.data_vars.keys() if v.endswith(photom_vars[0])]
trial_outcomes = np.unique(xr_spike_fr_interp.trial_outcome)
var2analyze = list(itertools.product(var, photom_vars, trial_outcomes))
evt_time = xr_spike_fr_interp.spk_event_time
evt_time_step = np.mean(np.diff(evt_time))


#%% Calculate the time step for each unit of lag
evt_time = xr_spike_fr_interp.spk_event_time
evt_time_step = np.mean(np.diff(evt_time))
# by default both the firing rate and photom data are averaged before calculating the correlation
# need to make sure photometry and neuropixels are using the new lick detection method
results = Parallel(n_jobs=20, verbose=5)(delayed(analyze_correlation)(xr_spike_fr_interp,
                                                           xr_session,
                                                           evt_name,
                                                           sig_name,
                                                           evt_time_step,
                                                           xr_spike_trial.cluID,
                                                           trial_outcome=trial_outcome,
                                                           average_trial=True) for evt_name, sig_name, trial_outcome in var2analyze)

#%% Save
xr_corr = xr.merge(results)
xr_corr.to_netcdf(soutput.xr_corr, engine='h5netcdf')
xr_spike_trial.close()
xr_session.close()

#%% Plot the correlation figures
for evt_name, sig_name,outcome in itertools.product(var, photom_vars, ['success','aborted']):
    # only plot successful trials
    idx = xr_session.isel(session_id=0).trial_outcome ==outcome
    if sum(idx)>0:
        xr_corr2plot = xr_corr.sel(trial_outcome=outcome)
        
        xr_spike2plot = xr_spike_fr_interp.sel(trial_nb = idx)
        xr_session2plot = xr_session.isel(session_id=0).sel(trial_nb = idx)
        
        fig = plot_extrem_corr(xr_corr2plot, xr_spike2plot, xr_session2plot, evt_name, sig_name)
        fig.savefig(Path(soutput.corr_plots)/f'corr_{evt_name}_{sig_name}_{outcome}.png',dpi=200)

# %% plot the overall distribution
sig_names = ['_zscored_df_over_f','_zscored_df_over_f_analog_2']

if len(xr_corr)>0:
    for outcome in trial_outcomes:
        fig,axes = plt.subplots(1,2,figsize=(10,10),dpi=200)

        for i, sn in enumerate(sig_names):
            df_meancorr = get_corr_spatial_distribution(xr_corr.sel(trial_outcome=outcome), df_metrics, sn)
            
            sns.heatmap(df_meancorr,cmap='vlag',ax=axes[i])
            axes[i].invert_yaxis()
            axes[i].set_title(sn)
            axes[i].set_ylabel('Depth um')

        fig.tight_layout()
        fig.savefig(Path(soutput.corr_plots)/f'corr_dist_{outcome}.png', dpi=200)

# %%
