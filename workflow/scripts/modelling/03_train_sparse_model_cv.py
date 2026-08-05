'''
Train pytorch model for sparse encoding with 10-fold cross-validation
'''

#%%
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt
import trialexp.process.ephys.utils as ephys_utils
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold, train_test_split
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
import matplotlib.pyplot as plt

#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflow/modelling.smk',
  [config.debug_folder + r'/processed/ach_sparse_encode_cv.pkl'],
  'train_sparse_model_cv')

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

# CV configuration
n_folds = 10
random_state = 42
val_split = 0.1  # Validation split ratio (20% of training data in each fold)
val_early_stop_patience = 3  # Early stopping patience

# Shuffle configuration for null distribution
n_shuffles = 1000
shuffle_random_seed = 42

#%%
for save_file, signal2analyze in zip(save_files, signal2analyze_list):
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing {signal2analyze} with {n_folds}-fold cross-validation (with validation split)")
    logger.info(f"{'='*80}\n")

    # ===== STAGE 1: Prepare full dataset (filters NaN trials internally) =====
    logger.info("Stage 1: Preparing full dataset (NEW trial-stacked format)...")
    full_data = decomp.prepare_data_for_encoding_trials(xr_session, signal2analyze)

    # Extract arrays (already filtered, includes baseline atom)
    atoms = full_data['atoms_norm_smooth']  # (time × n_neurons+1 × n_valid_trials)
    target = full_data['target_norm_smooth']  # (time × n_valid_trials)
    lick_rate = full_data['lick_rate']
    trial_outcome = full_data['trial_outcome']
    trial_nb = full_data['trial_nb']

    n_valid_trials = atoms.shape[2]
    T = atoms.shape[0]  # timepoints per trial

    logger.info(f"Valid trials after filtering: {n_valid_trials}")
    logger.info(f"Timepoints per trial: {T}")
    logger.info(f"Number of neurons (including baseline): {atoms.shape[1]}")

    # Transpose to model input format: (n_neurons+1, time, n_trials)
    dict_atoms = atoms.transpose([1, 0, 2])

    # Initialize CV splitter
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Storage for OOF predictions
    oof_predictions = np.full((n_valid_trials, T), np.nan)
    oof_ground_truth = np.full((n_valid_trials, T), np.nan)
    fold_results = []

    # Storage for shuffled ground truth (for null distribution)
    shuffle_ground_truth = {}
    for shuffle_idx in range(n_shuffles):
        shuffle_ground_truth[shuffle_idx] = np.full((n_valid_trials, T), np.nan)

    # ===== STAGE 2: CV loop =====
    logger.info(f"\nStage 2: Running {n_folds}-fold cross-validation...")

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(np.arange(n_valid_trials))):
        logger.info(f"\n{'-'*60}")
        logger.info(f"Fold {fold_idx+1}/{n_folds}")
        logger.info(f"Train trials: {len(train_idx)}, Test trials: {len(test_idx)}")
        logger.info(f"{'-'*60}")

        # ===== Further split train into train/validation =====
        train_inner_idx, val_idx = train_test_split(
            train_idx,
            test_size=val_split,
            random_state=random_state,
            shuffle=True
        )

        logger.info(f"Split training fold: {len(train_inner_idx)} train, {len(val_idx)} validation")

        # ===== Prepare train, validation, and test data =====
        dict_train = dict_atoms[:, :, train_inner_idx]  # (n_neurons+1, time, n_train)
        target_train = target[:, train_inner_idx]        # (time, n_train)

        dict_val = dict_atoms[:, :, val_idx]    # (n_neurons+1, time, n_val)
        target_val = target[:, val_idx]          # (time, n_val)

        dict_test = dict_atoms[:, :, test_idx]    # (n_neurons+1, time, n_test)
        target_test = target[:, test_idx]          # (time, n_test)

        # ===== Train model with validation =====
        logger.info("Training model with validation monitoring...")
        init_shifts = np.random.normal(100, 1, dict_train.shape[0])

        code_train, info_train, reconstruction_train, model_train = \
            decomp.sparse_encode_pytorch_with_shift_trials(
                target=target_train,
                dictionary=dict_train,
                validation_target=target_val,
                validation_dictionary=dict_val,
                val_early_stop_patience=val_early_stop_patience,
                val_early_stop_metric='r2',
                save_best_model=True,
                verbose_validation=True,
                max_shift_ms=[-200, 200],
                init_shift_ms=init_shifts,
                sampling_rate=sampling_rate,
                n_iterations=10,
                sparsity_weight=5e-3,
                n_steps_code=5000,
                n_steps_shift=1000,
                sparsity_type='elastic_net',
                max_lr_shift=0.1,
                device='cuda',
                shift_print_step=400,
                code_print_step=1000,
                early_stop_patience=100,
                use_mlflow=False
            )

        train_r2 = info_train['variance_explained']
        logger.info(f"\nTrain R²: {train_r2:.4f}")

        # Log validation results
        if info_train['val_r2_history']:
            best_val_r2 = info_train['best_val_metric']
            final_val_r2 = info_train['val_r2_history'][-1]
            logger.info(f"Best Val R²: {best_val_r2:.4f} at iteration {info_train['best_val_iteration']+1}")
            logger.info(f"Final Val R²: {final_val_r2:.4f}")
            if info_train['val_early_stopped']:
                logger.info(f"Training stopped early after {info_train['n_iterations_completed']} iterations")

        # ===== Predict on test fold =====
        logger.info("\nPredicting on test fold...")
        test_results = decomp.evaluate_sparse_model_on_test_data(
            model_train=model_train,
            dict_test=dict_test,
            target_test=target_test,
            sampling_rate=sampling_rate,
            max_shift_ms=[-200, 200],
            device='cuda'
        )

        # Extract results
        reconstruction_test = test_results['reconstruction']
        test_r2 = test_results['test_r2']
        test_mse = test_results['test_mse']

        logger.info(f"Test R²: {test_r2:.4f}")
        logger.info(f"Train-Test gap: {train_r2 - test_r2:.4f}")
        if info_train['val_r2_history']:
            logger.info(f"Train-Val gap: {train_r2 - info_train['val_r2_history'][-1]:.4f}")

        # ===== Store OOF predictions =====
        # Reshape predictions back to (n_test_trials × T)
        n_test_trials = len(test_idx)

        test_pred_trials = reconstruction_test.T  # (time, trials) -> (trials, time)
        test_true_trials = target_test.T

        # Store in OOF arrays (using original trial indices)
        for i, orig_idx in enumerate(test_idx):
            oof_predictions[orig_idx] = test_pred_trials[i]
            oof_ground_truth[orig_idx] = test_true_trials[i]

        # ===== Generate shuffled ground truth for null distribution =====
        logger.info(f"Generating {n_shuffles} shuffled ground truth samples...")
        for shuffle_idx in range(n_shuffles):
            # Create reproducible permutation
            shuffle_seed = shuffle_random_seed + fold_idx * n_shuffles + shuffle_idx
            rng = np.random.RandomState(shuffle_seed)
            permutation = rng.permutation(len(test_idx))

            # Shuffle test ground truth (predictions stay fixed)
            shuffled_true_trials = test_true_trials[permutation]

            # Store in global array using original trial indices
            for i, orig_idx in enumerate(test_idx):
                shuffle_ground_truth[shuffle_idx][orig_idx] = shuffled_true_trials[i]

        # Store fold metadata
        fold_results.append({
            'fold_idx': fold_idx,
            'train_inner_idx': train_inner_idx,
            'val_idx': val_idx,
            'test_idx': test_idx,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'val_r2_history': info_train['val_r2_history'],
            'best_val_r2': info_train['best_val_metric'],
            'best_val_iteration': info_train['best_val_iteration'],
            'val_early_stopped': info_train['val_early_stopped'],
            'shifts_ms': info_train['shifts_ms'],
            'model_state_dict': model_train.state_dict()
        })

        # Clear GPU memory
        del model_train
        torch.cuda.empty_cache()

    # ===== STAGE 3: Calculate Global OOF R² =====
    logger.info(f"\n{'='*80}")
    logger.info("Stage 3: Computing global out-of-fold metrics...")
    logger.info(f"{'='*80}")

    # Verify all trials were filled
    assert not np.any(np.isnan(oof_predictions)), "Not all trials were filled in OOF predictions!"
    assert not np.any(np.isnan(oof_ground_truth)), "Not all trials were filled in OOF ground truth!"

    oof_pred_flat = oof_predictions.ravel()
    oof_true_flat = oof_ground_truth.ravel()

    oof_mse = np.mean((oof_pred_flat - oof_true_flat) ** 2)
    oof_var = np.var(oof_true_flat)
    oof_r2 = 1 - (oof_mse / oof_var)

    # Calculate per-fold statistics
    fold_train_r2_mean = np.mean([f['train_r2'] for f in fold_results])
    fold_train_r2_std = np.std([f['train_r2'] for f in fold_results])
    fold_test_r2_mean = np.mean([f['test_r2'] for f in fold_results])
    fold_test_r2_std = np.std([f['test_r2'] for f in fold_results])

    # Validation statistics
    fold_best_val_r2_mean = np.mean([f['best_val_r2'] for f in fold_results])
    fold_best_val_r2_std = np.std([f['best_val_r2'] for f in fold_results])
    n_early_stopped = sum([f['val_early_stopped'] for f in fold_results])

    logger.info(f"\n{'='*60}")
    logger.info("CROSS-VALIDATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Global OOF R²: {oof_r2:.4f}")
    logger.info(f"Mean fold train R²: {fold_train_r2_mean:.4f} ± {fold_train_r2_std:.4f}")
    logger.info(f"Mean best validation R²: {fold_best_val_r2_mean:.4f} ± {fold_best_val_r2_std:.4f}")
    logger.info(f"Mean fold test R²: {fold_test_r2_mean:.4f} ± {fold_test_r2_std:.4f}")
    logger.info(f"Train-test gap: {fold_train_r2_mean - fold_test_r2_mean:.4f}")
    logger.info(f"Folds stopped early: {n_early_stopped}/{n_folds}")
    logger.info(f"{'='*60}\n")

    # ===== STAGE 3.5: Calculate Shuffle R² scores (null distribution) =====
    logger.info(f"\n{'='*80}")
    logger.info("Stage 3.5: Computing shuffle R² scores (null distribution)...")
    logger.info(f"{'='*80}")

    # Verify all shuffle arrays are filled
    for shuffle_idx in range(n_shuffles):
        assert not np.any(np.isnan(shuffle_ground_truth[shuffle_idx])), \
            f"Shuffle {shuffle_idx} has unfilled trials!"

    # Calculate R² for each shuffle
    shuffle_r2_scores = np.zeros(n_shuffles)
    for shuffle_idx in range(n_shuffles):
        shuffle_true_flat = shuffle_ground_truth[shuffle_idx].ravel()

        # Calculate R² comparing OOF predictions to shuffled ground truth
        shuffle_mse = np.mean((oof_pred_flat - shuffle_true_flat) ** 2)
        shuffle_var = np.var(shuffle_true_flat)
        shuffle_r2_scores[shuffle_idx] = 1 - (shuffle_mse / shuffle_var)

    # Calculate statistics
    shuffle_r2_mean = np.mean(shuffle_r2_scores)
    shuffle_r2_std = np.std(shuffle_r2_scores)
    shuffle_r2_min = np.min(shuffle_r2_scores)
    shuffle_r2_max = np.max(shuffle_r2_scores)
    shuffle_r2_percentiles = np.percentile(shuffle_r2_scores, [5, 25, 50, 75, 95])

    # Calculate empirical p-value
    p_value = np.mean(shuffle_r2_scores >= oof_r2)

    logger.info(f"\n{'='*60}")
    logger.info("SHUFFLE NULL DISTRIBUTION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"True OOF R²: {oof_r2:.4f}")
    logger.info(f"Shuffle R² (mean ± std): {shuffle_r2_mean:.4f} ± {shuffle_r2_std:.4f}")
    logger.info(f"Shuffle R² range: [{shuffle_r2_min:.4f}, {shuffle_r2_max:.4f}]")
    logger.info(f"Shuffle R² percentiles [5,25,50,75,95]: {shuffle_r2_percentiles}")
    logger.info(f"Empirical p-value: {p_value:.4f} ({int(p_value * n_shuffles)}/{n_shuffles} shuffles >= true R²)")
    logger.info(f"{'='*60}\n")

    # ===== STAGE 4: Save results =====
    logger.info(f"Saving results to {save_file}")

    with open(save_file, 'wb') as f:
        pickle.dump({
            'cv_type': 'kfold',
            'n_folds': n_folds,
            'random_state': random_state,
            'val_split': val_split,
            'oof_predictions': oof_predictions,
            'oof_ground_truth': oof_ground_truth,
            'oof_r2': oof_r2,
            'oof_mse': oof_mse,
            'fold_results': fold_results,
            'fold_train_r2_mean': fold_train_r2_mean,
            'fold_train_r2_std': fold_train_r2_std,
            'fold_test_r2_mean': fold_test_r2_mean,
            'fold_test_r2_std': fold_test_r2_std,
            'fold_best_val_r2_mean': fold_best_val_r2_mean,
            'fold_best_val_r2_std': fold_best_val_r2_std,
            'n_folds_early_stopped': n_early_stopped,
            # Shuffle/null distribution results
            'shuffle_r2_scores': shuffle_r2_scores,
            'n_shuffles': n_shuffles,
            'shuffle_random_seed': shuffle_random_seed,
            'shuffle_r2_mean': shuffle_r2_mean,
            'shuffle_r2_std': shuffle_r2_std,
            'shuffle_r2_percentiles': shuffle_r2_percentiles,
            'empirical_p_value': p_value,
            # Metadata
            'signal2analyze': signal2analyze,
            'trial_outcome': trial_outcome,
            'trial_nb': trial_nb,
            'n_valid_trials': n_valid_trials,
            'timepoints_per_trial': T,
            'n_neurons': atoms.shape[1],
            'sampling_rate': sampling_rate,
            'cluID': xr_all_filtered.cluID.data,
        }, f)

    logger.info(f"Completed processing {signal2analyze}\n")

logger.info("All signals processed successfully!")
# %%
