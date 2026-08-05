'''
Train pytorch model for sparse encoding with time shift
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
#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflow/modelling.smk',
  [config.debug_folder + r'/processed/da_sparse_encode.pkl'],
  'train_sparse_model')

#%% load data
xr_spikes_all = xr.open_dataset(sinput.xr_timewarpped)
xr_photom_warp = xr.load_dataset(sinput.xr_photom_timewarped)

#%% Merge Neuropixels with photometry
sampling_rate = 1000/np.diff(xr_photom_warp['time'])[0]
print(f'Photometry sample rate: {sampling_rate} Hz')

xr_spikes_all = xr_spikes_all.interp(time = xr_photom_warp.time) # make sure the time axis is the same

xr_all_trials = xr_spikes_all.merge(xr_photom_warp)

#%% add in coefficient of variance information
cv = ephys_utils.get_cell_mean_cv(xr_all_trials['spikes_FR_session'], axis=1)
fr = ephys_utils.get_cell_mean_fr(xr_all_trials['spikes_FR_session'], axis=1)

xr_all_trials['cv'] = xr.DataArray(cv, dims=['cluID'], coords={'cluID': xr_all_trials.cluID})
xr_all_trials['fr'] = xr.DataArray(fr, dims=['cluID'], coords={'cluID': xr_all_trials.cluID})

# Filter cells based on firing rate and coefficient of variation criteria
mask = (xr_all_trials['fr'] > 1) #doesn't quite make sense to use cv as criteira here, as cell response differently for different trials
xr_all_filtered = xr_all_trials.sel(cluID=mask)

print(f'Original number of cells: {len(xr_all_trials.cluID)}')
print(f'Filtered number of cells: {len(xr_all_filtered.cluID)}')

#%%
# # the combined dataset contains all possible trial_nb in the combined data, that's why some trial_nb are nan
# we need to filter it down to the valid data only, otherwise there will be too many nan

xr_session = xr_all_filtered
xr_session = xr_session.sel(trial_nb = (xr_session.trial_outcome!='nan'))

save_files = [soutput.ach_model, soutput.da_model]
signal2analyze_list = ['zscored_df_over_f', 'zscored_df_over_f_analog_2']
# signal2analyze_list = ['zscored_df_over_f_analog_2']


# for target_sensor, signal2analyze in zip(target_sensor, signal2analyze_list):
# Extract atoms and target for all trials
# extract the appropriate signal for each trial outcome 


for save_file, signal2analyze in zip(save_files, signal2analyze_list):
    
    # Use the function
    prepared_data = decomp.prepare_data_for_encoding(xr_session, signal2analyze)

    dict_atoms_smooth = prepared_data['dict_atoms_smooth']
    target_stack_smooth = prepared_data['target_stack_smooth']
    target_stack = prepared_data['target_stack']
    atoms = prepared_data['atoms']
    target = prepared_data['target']
    event_time = prepared_data['event_time']
    cluID_atom = prepared_data['cluID_atom']
    trial_outcome = prepared_data['trial_outcome']
    trial_nb = prepared_data['trial_nb']
    mask_idx = prepared_data['mask_idx'] #which trial is actually used to train the model
    lick_rate = prepared_data['lick_rate'] # currently not used


    # Check if CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    init_shifts = np.random.normal(100, 1, dict_atoms_smooth.shape[0])

    code_final, info, reconstruction, model = decomp.sparse_encode_pytorch_with_shift(
        target=target_stack_smooth,
        dictionary=dict_atoms_smooth,  # unshifted patterns
        max_shift_ms=[-200, 200],
        init_shift_ms= init_shifts,
        sampling_rate=sampling_rate,
        n_iterations=10,
        sparsity_weight=1e-3,
        n_steps_code=5000,
        n_steps_shift= 1000,
        sparsity_type='elastic_net',
        max_lr_shift=0.1,
        device='cuda',
        shift_print_step= 400,
        code_print_step=1000,
        early_stop_patience = 100,
        use_mlflow=True  # Track experiments
    )

    with open(save_file,'wb') as f:
        pickle.dump({
            'code_final': code_final,
            'target_stack_smooth': target_stack_smooth,
            'dict_atoms_smooth': dict_atoms_smooth,
            'reconstruction': reconstruction,
            'trial_outcome':  trial_outcome,
            'event_time': event_time,
            'atoms': atoms,
            'target': target,
            'info': info,
            'model': model,
            'lick_rate': lick_rate,
            'cluID': cluID_atom,
            'trial_nb': trial_nb,
            'mask_idx': mask_idx
        }, f)
        
        
    # plot figures for quick check
    fig, ax = decomp.plot_reconstruction_comparison(target_stack_smooth, reconstruction)
    fig.savefig(Path(soutput.figures_dir)/f'reconstruction_{signal2analyze}.png', dpi=200)