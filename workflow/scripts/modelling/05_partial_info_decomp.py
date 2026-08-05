'''
Do partial informationa decomposition to study influence of DA/ACh on neural firing rate
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
from sklearn.feature_selection import mutual_info_regression
from tqdm.auto import tqdm
import pandas as pd
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
fr = xr_session['spikes_FR_session'].data
photom_da = xr_session['zscored_df_over_f_analog_2'].data  # trial x time
photom_ach = xr_session['zscored_df_over_f'].data 
fr = fr.transpose([2, 0, 1]) # cluID x trial x time

# filter invalid trials
mask_idx = (~np.isnan(fr[0, :, 0])) & (~np.isnan(photom_da[:,0]))
# Filter valid trials
fr = fr[:, mask_idx, :]
photom_da = photom_da[mask_idx,:]
photom_ach = photom_ach[mask_idx,:]

#%%
from idtxl.bivariate_pid import BivariatePID
from idtxl.data import Data
from idtxl.idtxl_utils import discretise  

alph = 10

x1 = discretise(photom_ach.ravel(), numBins=alph)
x1 = x1.reshape( photom_ach.shape[0],-1)

x2 = discretise(photom_da.ravel(), numBins=alph)
x2 = x2.reshape(photom_da.shape[0],-1)

y = discretise(fr.reshape(fr.shape[0],-1).T, numBins=alph) # expect replication x variable
y = y.T.reshape(fr.shape)
#%%
# plt.plot(fr[0,0,:])
# plt.plot(y[0,0,:])
#%%
# Note: calculation of PD is very computationally expensive and grow expontential according to the size of the alphabet
# too slow to do any permutation test
all_signals = np.array([x1.T, x2.T, y[0,:,:].T]).astype(int) # should be process x time x trial

data = Data(all_signals, dim_order='psr', normalise=False) # have to disable normalized

settings = {
    # 'nm_bins': alph,            # Number of bins for discretization (e.g., Low, Med, High)
    # 'discretisation': 'equiwidth', # Or 'equal_count' (quantiles)
    # 'n_perm_pid': 100,       # Number of permutations for statistical significance
    'pid_estimator': 'SydneyPID',
    'alph_s1': alph, #alphabet size
    'alph_s2': alph,
    'alph_t': alph,
    'alpha':0.05,
    'num_reps': 63,
    'max_iters': 1000,
    'max_unsuc_swaps_row_parm': 60,
}
pid_estimator = BivariatePID()
results = pid_estimator.analyse_single_target(data=data, target=2, sources=[0, 1], settings=settings)

target_info = results.get_single_target(2)
unique_ach = target_info['unq_s1']
unique_da = target_info['unq_s2']
redundancy = target_info['shd_s1_s2']
synergy = target_info['syn_s1_s2']

print(f"Unique DA: {unique_da:.4f} bits")
print(f"Unique ACh: {unique_ach:.4f} bits")
print(f"Redundancy: {redundancy:.4f} bits")
print(f"Synergy: {synergy:.4f} bits")

#%% Use another library to try

import numpy as np
from entropy_estimators import continuous

#%%
results_list = []

for i in tqdm(range(10)):
    (synergy, unique_ach, unique_da, redundancy) = continuous.get_pid(photom_ach.ravel(),
                                                                    photom_da.ravel(),
                                                                    fr[i].ravel(), 
                                                                    k=3 )
    # print(f'{synergy=:.2f} {unique_ach=:.2f} {unique_da=:.2f} {redundancy=:.2f}')

    results  = {
        'synergy': synergy,
        'unique_ach': unique_ach,
        'unique_da': unique_da,
        'redundancy': redundancy,
        'idx': i
    }
    results_list.append(results)
    

df_results = pd.DataFrame(results_list)
