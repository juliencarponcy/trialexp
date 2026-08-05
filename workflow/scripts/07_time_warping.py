'''
Perform time warping to align events across trials
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
import json
#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflow/pycontrol.smk',
  [config.debug_folder + '/processed/xr_photom_timewarped.nc'],
  'time_warping')

# %% Load data
df_events_cond = pd.read_pickle(sinput.event_dataframe)
df_conditions = pd.read_pickle(sinput.condition_dataframe)
xr_photometry = xr.load_dataset(sinput.xr_photometry, engine = 'h5netcdf')


# %% Parameters
signal2analyze = ['zscored_df_over_f', 'zscored_df_over_f_analog_2','zscored_df_over_f_analog_3']
signal2analyze = [s for s in signal2analyze if s in xr_photometry.data_vars]

# the warp spec loading has moved to get_warping_specs, please update the spec matching there
trigger, extraction_specs, outcome2plot = lm.get_warping_specs(df_events_cond, df_conditions, 'params/timewarp_spec.json')


#%% Time warping to align events

xr_conditions = xr.Dataset.from_dataframe(df_conditions)

xa_list = []
interp_results_list = None
for signal_var in signal2analyze:
    # TODO: find a way to warp aborted trial differently
    xa, interp_results_list = lm.time_warp_data(df_events_cond, 
                           xr_photometry[signal_var], 
                           extraction_specs, 
                           trigger,
                           xr_photometry.attrs['sampling_rate'],
                           verbose=False)
                        
                        
    
    xa_list.append(xa)


if interp_results_list:
    #add the interp result for later plotting
    df_interp_res = pd.DataFrame(interp_results_list)
    df_interp_res['trial_nb'] = xa.trial_nb
    df_interp_res = df_interp_res.set_index('trial_nb')
    xr_interp_res = df_interp_res.to_xarray()
else:
    xr_interp_res = xr.Dataset()




# %% Also do time wrapping on the licking rate
lick_on = df_events_cond[df_events_cond.content=='lick'].time

lick_rate,_ = np.histogram(lick_on, xr_photometry.time)

# calculate the rolling mean lick rate
lick_bin_size = 0.2 # in seconds
win_size = int(lick_bin_size*xr_photometry.attrs['sampling_rate'])
lick_rate = np.convolve(lick_rate, np.ones(win_size)/win_size, mode='same')*xr_photometry.attrs['sampling_rate']


xa_lick_rate = xr.DataArray(lick_rate, name='lick_rate',
                            coords={'time':xr_photometry.time[:-1]},dims=['time'])

xa, interp_results_list = lm.time_warp_data(df_events_cond, 
                        xa_lick_rate, 
                        extraction_specs, 
                        trigger,
                        xr_photometry.attrs['sampling_rate'],
                        verbose=False)

xa_list.append(xa)


#%% Save data
xr_warped = xr.merge([xr_conditions, xr_interp_res, *xa_list])
xr_warped.attrs['extraction_specs'] = json.dumps(extraction_specs) # also store the spec for reference
xr_warped.to_netcdf(soutput.xr_timewarpped, engine='h5netcdf')

#%% check for valid trials
xr_success  = xr_warped.sel(trial_nb=(xr_warped.trial_outcome=='success'))
valid_trials = np.all(~np.isnan(xr_success['zscored_df_over_f'].data),axis=1)
if len(valid_trials)>0:
    ratio = np.sum(valid_trials)/len(valid_trials)
else:
    ratio = 0
logger.info('Ratio of valid successful trials:', ratio)

lm.print_time_warping_summary(xr_warped, signal2analyze[0])

#%%
# Debug plot for failed trials (geometry view in original trial time)
failed_mask = np.isnan(xr_warped[signal2analyze[0]].data).all(axis=1)
trial_outcome_mask = (xr_warped.trial_outcome == 'success')
failed_trial_nbs = xr_warped.trial_nb.values[failed_mask&trial_outcome_mask].tolist()
if failed_trial_nbs:
    fig, ax = lm.plot_trial_warp_debug(
        df_events_cond, extraction_specs, trigger,
        trial_nbs=failed_trial_nbs,
        max_trials=10,
        output_file=Path(soutput.figure_dir) / 'timewarp_debug.png'
    )

#%% Plot the time wrapped data
for var in signal2analyze:
    unique_outcome = np.unique(xr_warped.trial_outcome)
    fig, axes = plt.subplots(len(outcome2plot),1,figsize=(10,4*len(outcome2plot)))
    
    if type(axes) is not np.ndarray:
        axes =[axes]
        
    for outcome, ax in zip(outcome2plot, axes):
        xr2plot = xr_warped.sel(trial_nb = xr_warped.trial_outcome.isin(outcome))
        lm.plot_warpped_data(xr2plot, var, extraction_specs, trigger, ax=ax)
        
    fig.tight_layout()
    fig.savefig(Path(soutput.figure_dir)/f'{var}_timewarp.png', bbox_inches='tight', dpi=200)


# %%
var='lick_rate'
unique_outcome = np.unique(xr_warped.trial_outcome)
fig, axes = plt.subplots(len(outcome2plot),1,figsize=(10,4*len(outcome2plot)))

if type(axes) is not np.ndarray:
    axes =[axes]
    
for outcome, ax in zip(outcome2plot, axes):
    xr2plot = xr_warped.sel(trial_nb = xr_warped.trial_outcome.isin(outcome))
    lm.plot_warpped_data(xr2plot, var, extraction_specs, trigger, ax=ax, ylabel='Licking rate (per sec)', ylim=[0,15])

fig.tight_layout()
fig.savefig(Path(soutput.figure_dir)/f'{var}_timewarp.png', bbox_inches='tight', dpi=200)

xr_warped.close()
# %%
