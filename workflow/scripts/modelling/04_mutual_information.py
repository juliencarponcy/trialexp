'''
Calculating mutual information and do shuffling analysis
'''

#%%
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt
import trialexp.process.ephys.utils as ephys_utils
from sklearn.preprocessing import normalize
import torch
import pickle
import trialexp.process.model.pytorch_decomposition as decomp
from snakehelper.SnakeIOHelper import getSnake
from trialexp import config
import xarray as xr
from scipy.signal import savgol_filter
from loguru import logger
from pathlib import Path
import numpy as np
from trialexp.process.pyphotometry.linear_modelling import compute_ticks
import trialexp.process.model.mutual_info as mi 
import seaborn as sns
import itertools
#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflow/modelling.smk',
  [config.debug_folder + r'/processed/xr_mi.nc'],
  'compute_mutual_information')

#%% load data
xr_spikes_all = xr.open_dataset(sinput.xr_timewarpped)
xr_photom_warp = xr.load_dataset(sinput.xr_photom_timewarped)

#%% Merge Neuropixels with photometry
sampling_rate = 1000/np.diff(xr_photom_warp['time'])[0]
print(f'Photometry sample rate: {sampling_rate} Hz')

xr_spikes_all = xr_spikes_all.interp(time = xr_photom_warp.time) # make sure the time axis is the same

xr_all_trials = xr_spikes_all.merge(xr_photom_warp)


xr_session = xr_all_trials
xr_session = xr_session.sel(trial_nb = (xr_session.trial_outcome!='nan'))
signal2analyze_list = ['zscored_df_over_f', 'zscored_df_over_f_analog_2','lick_rate']

#%%
xr_mi = xr.Dataset()
for sig_var in signal2analyze_list:
    logger.info(f'Calculating MI for {sig_var}')
    # calculate MI on specific event window
    extraction_specs = eval(xr_session.attrs['extraction_specs'])
    event_win = mi.extract_event_windows(extraction_specs, xr_session)
    xa_mi = mi.calculate_mi_per_event(xr_session,event_win, photom_var=sig_var)
    xa_mi_shuffle = mi.calculate_mi_per_event_shuffled(xr_session,event_win, photom_var=sig_var, n_shuffles=100)
    xa_comp = mi.compare_mi_significance(xa_mi, xa_mi_shuffle)
    
    xr_mi[f'mi_{sig_var}'] = xa_mi
    xr_mi[f'mi_{sig_var}_shuffle'] = xa_mi_shuffle
    xr_mi[f'mi_{sig_var}_comp_pvalue'] = xa_comp['p_value']

#%%

#%% plot figures
for sig_var, evt in itertools.product(signal2analyze_list, xr_mi.event.data):
    logger.info(f'Processing {evt} of {sig_var}')
    fig = mi.plot_top_mi_cells(xa_mi, xr_session,sig_var, extraction_specs, event=evt, n_cells=5)
    
    fig.savefig(Path(soutput.figures_dir)/f'mi_{evt}_{sig_var}.png',dpi=200)
    
# %% save
xr_mi.to_netcdf(soutput.xr_mi)