'''
Do time warping on spike sorted data

'''

#%%
from snakehelper.SnakeIOHelper import getSnake
from trialexp.process.pyphotometry.plotting_utils import annotate_trial_number, plot_and_handler_error, plot_pyphoto_heatmap
from trialexp.process.pyphotometry.utils import *
import xarray as xr
import pandas as pd 
import seaborn as sns 
from matplotlib import pyplot as plt 
import numpy as np
from trialexp import config
import trialexp.process.pyphotometry.linear_modelling as lm
from pathlib import Path
import itertools
from joblib import Parallel, delayed
import trialexp.process.ephys.photom_correlation as pc
#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [config.debug_folder + '/processed/figures/ephys/timewarp'],
  'spike_timewarp_figures')

# %% Load data
xr_corr = xr.load_dataset(sinput.xr_corr, engine = 'h5netcdf')
xr_photom_timewarpped = xr.load_dataset(sinput.xr_photom_timewarp, engine = 'h5netcdf')
xr_session = xr.load_dataset(sinput.xr_session, engine = 'h5netcdf')
xr_warped = xr.load_dataset(sinput.xr_spike_timewarp)
# %% Parameters
signal2analyze = ['spikes_FR_session', 'spikes_zFR_session']
signal2analyze = [s for s in signal2analyze if s in xr_warped.data_vars]

# trigger, extraction_specs, outcome2plot = lm.get_warping_specs(df_events_cond, df_conditions, 'params/timewarp_spec.json')

trigger = xr_warped.attrs['triggers']
extraction_specs = eval(xr_warped.attrs['extraction_specs'])
outcome2plot = eval(xr_warped.attrs['outcome2plot'])


# #%% Plot presentative figures from each events

# Get the names of the signal and event to plot
photom_vars = ['zscored_df_over_f', 'zscored_df_over_f_analog_2']
event2plot = [v.replace('_'+photom_vars[0],'') for v in xr_session.data_vars.keys() if v.endswith(photom_vars[0]) and not (v in photom_vars)]

#plot
def plot_timewarp(photom_signal_name, evt_name, figure_dir):
    print(f'Plotting time warp for {photom_signal_name} at {evt_name}')
    fig = pc.plot_extrem_corr_timewarp(extraction_specs, trigger, 
                                       xr_corr, xr_warped, xr_photom_timewarpped, photom_signal_name, 
                            'spikes_FR_session', 
                            evt_name, mode='abs')
    fig.savefig(Path(figure_dir)/f'timewarp_{evt_name}_{photom_signal_name}.png',dpi=200)
    plt.close(fig)

Parallel(n_jobs=4)(
    delayed(plot_timewarp)(photom_signal_name, evt_name, str(soutput.figure_dir))
    for photom_signal_name, evt_name in itertools.product(photom_vars, event2plot)
)
