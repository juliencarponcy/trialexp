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
from loguru import logger
#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [config.debug_folder + '/processed/xr_spikes_timewarped.nc'],
  'spike_timewarp')

# %% Load data
df_events_cond = pd.read_pickle(sinput.event_dataframe)
df_conditions = pd.read_pickle(sinput.condition_dataframe)
xr_spike_fr = xr.open_dataset(sinput.xr_spikes_fr, engine = 'h5netcdf')

# %% Parameters
signal2analyze = ['spikes_FR_session', 'spikes_zFR_session']
signal2analyze = [s for s in signal2analyze if s in xr_spike_fr.data_vars]

trigger, extraction_specs, outcome2plot = lm.get_warping_specs(df_events_cond, df_conditions, 'params/timewarp_spec.json')


#%% Time warping to align events

xr_conditions = xr.Dataset.from_dataframe(df_conditions)

xa_list = []
for signal_var in signal2analyze:
    xa, interp_results_list = lm.time_warp_data(df_events_cond, 
                           xr_spike_fr[signal_var], 
                           extraction_specs, 
                           trigger,
                           1000/xr_spike_fr.attrs['bin_duration'],
                           verbose=False)
                        
    
    xa_list.append(xa)
    
#add the interp result for later plotting
df_interp_res = pd.DataFrame(interp_results_list)
df_interp_res['trial_nb'] = xa.trial_nb
df_interp_res = df_interp_res.set_index('trial_nb')
xr_interp_res = df_interp_res.to_xarray()


#%% Save data
xr_warped = xr.merge([xr_conditions, xr_interp_res, *xa_list])

# add additional info to xr_warped
xr_warped.attrs['extraction_specs'] = str(extraction_specs) #work around save filetype limitation
xr_warped.attrs['outcome2plot'] = str(outcome2plot)
xr_warped.attrs.update(xr_spike_fr.attrs)
xr_warped.attrs.update(df_events_cond.attrs)

xr_warped.to_netcdf(soutput.xr_timewarpped, engine='h5netcdf')

#%% check for valid trials
xr_success  = xr_warped.sel(trial_nb=(xr_warped.trial_outcome=='success'))
valid_trials = np.all(~np.isnan(xr_success['spikes_FR_session'].data[:,:,0]),axis=1)
logger.info('Ratio of valid trials:', np.sum(valid_trials)/len(valid_trials))


# %%
xr_spike_fr.close()
# %%
