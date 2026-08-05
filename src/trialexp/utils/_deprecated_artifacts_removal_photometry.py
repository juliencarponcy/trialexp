# DEPRECATED, possibly better done by Continuous_Dataset() methods
if remove_artifacts == True:
    
    # dbscan_anomaly_detection(photo_array)

    # Fit exponential to red channel
    red_exp = fit_exp_func(photometry_dict['analog_2'], fs = fs, medfilt_size = 3)
    # substract exponential from red channel
    red_minus_exp = photometry_dict['analog_2'] - red_exp

    if verbose:
        try:
            rig_nb = int(self.files['mp4'][0].split('Rig_')[1][0])
        except:
            rig_nb = 'unknown'

        time = np.linspace(1/fs, len(photometry_dict['analog_2'])/fs, len(photometry_dict['analog_2']))

        fig, ax = plt.subplots()

        plt.plot(time, photometry_dict['analog_1'], 'g', label='dLight')
        plt.plot(time, photometry_dict['analog_2'], 'r', label='red channel')
        # plt.plot(time, photometry_dict['analog_1_expfit'], 'k')
        plt.plot(time, red_exp, 'k')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Signal (volts)')
        plt.title('Raw signals')
        ax.text(0.02, 0.98, f'subject: {self.subject_ID}, date: {self.datetime}, rig: {rig_nb}',
                ha='left', va='top', transform=ax.transAxes)
        # plt.xlim(xlim)
        # plt.ylim(ylim)
        plt.legend()
        plt.show()


    # red_data = photo_array[:,:,col_names_numpy['analog_2_filt']]

    # find how many gaussian in red channel
    nb_gauss_in_red_chan = find_n_gaussians(
        data = red_minus_exp,
        plot_results = verbose,
        max_nb_gaussians = 4
    )


    if nb_gauss_in_red_chan == 1:
        # session look "clean", no artifacts
        if verbose:
            print('signal looks clean, no trial removed')

    else:
        # session look like there is different levels of baseline fluorescence,
        # likely indicating human interventions

        # HARD CODED, z-score value to exclude trials based on variance of the
        # filtered red-channnel
        z_var_median_thresh = 0.02
        z_mean_thresh = 1
        var_mean_thresh = 0.0005
        # compute variance of red channel
        var_trials = photo_array[:,:,col_names_numpy['analog_2_filt']].var(1)
        # compute mean of red channel
        mean_trials = photo_array[:,:,col_names_numpy['analog_2_filt']].mean(1)
        # z-score the variance of all trials
        zscore_var = zscore(var_trials)
        zscore_var_med = np.median(zscore_var)



        # First step: determine which trials looks artifacted based on variance (red chan)
        trials_to_exclude = [idx for idx, v in enumerate(zscore_var) if
            (v > zscore_var_med + z_var_median_thresh)]# or (np.abs(zscore_mean[idx]) > 2)]
        # As trial_nb starts at 1, adding +1 to use as filter in the df_meta_photo dataframe
        trials_to_include = [idx for idx, v in enumerate(zscore_var) if
            (v < zscore_var_med + z_var_median_thresh)]# and (np.abs(zscore_mean[idx]) < 5)]

        # Second step: perform check on the mean (red chan) for remaining trials
        zscore_mean = zscore(mean_trials[trials_to_include])                  
        var_mean = mean_trials[trials_to_include].var() 

        print(f'median: {np.median(var_trials)}, var_mean: {var_mean}, zscore_mean: {zscore_mean}')                
        if var_mean > var_mean_thresh:
            
            
            excluded_on_mean = [idx for idx, v in enumerate(zscore_mean) if
                (np.abs(v) > z_mean_thresh)]

            trials_to_exclude = trials_to_exclude + excluded_on_mean # or (np.abs(zscore_mean[idx]) > 2)]

            print(f'{len(excluded_on_mean)} exclusion(s) based on abnormal mean of red channel')


        all_trials = set(range(len(mean_trials)))
        trials_to_include = list(all_trials - set(trials_to_exclude))

            
        
    if verbose:
        if nb_gauss_in_red_chan == 1:
            trials_to_include = range(photo_array.shape[0])
            trials_to_exclude = []

        print(f'{len(trials_to_exclude)} trials with artifacts were removed')

        # retransfrom trials_to_include in zero-based indexing to plot from
        # the numpy array

        fig, axs = plt.subplots(nrows=2, ncols=2, sharey='row')
        timevec_trial = np.linspace(self.trial_window[0], self.trial_window[1], photo_array.shape[1])

        if nb_gauss_in_red_chan != 1:
            _ = axs[0,0].plot(timevec_trial, photo_array[trials_to_exclude,:,col_names_numpy['analog_2_filt']].T, alpha=0.3)
            _ = axs[0,0].plot(timevec_trial, photo_array[trials_to_exclude,:,col_names_numpy['analog_2_filt']].mean(0), c='k', alpha=1)

        _ = axs[0,1].plot(timevec_trial, photo_array[trials_to_include,:,col_names_numpy['analog_2_filt']].T, alpha=0.3)
        _ = axs[0,1].plot(timevec_trial, photo_array[trials_to_include,:,col_names_numpy['analog_2_filt']].mean(0), c='k', alpha=1)

        if nb_gauss_in_red_chan != 1:
            _ = axs[1,0].plot(timevec_trial, photo_array[trials_to_exclude,:,col_names_numpy['analog_1_filt']].T, alpha=0.3)
            _ = axs[1,0].plot(timevec_trial, photo_array[trials_to_exclude,:,col_names_numpy['analog_1_filt']].mean(0), c='k', alpha=1)

        _ = axs[1,1].plot(timevec_trial, photo_array[trials_to_include,:,col_names_numpy['analog_1_filt']].T, alpha=0.3)
        _ = axs[1,1].plot(timevec_trial, photo_array[trials_to_include,:,col_names_numpy['analog_1_filt']].mean(0), c='k', alpha=1)

        axs[0,0].set_title('red channel excluded trials')
        axs[0,1].set_title('red channel included trials')

        axs[1,0].set_title('green channel excluded trials')
        axs[1,1].set_title('green channel included trials')
        plt.show()
    
    # retransfrom trials_to_include in zero-based indexing to plot from
    # the numpy array
    trials_to_include = np.array(trials_to_include)
    trials_to_include = trials_to_include-1

    # delete trials from the numpy array
    np.delete(photo_array, trials_to_exclude, 0)
    # delete trials from df_meta_photo metadata DataFrame
    df_meta_photo = df_meta_photo[df_meta_photo['trial_nb'].isin(trials_to_include)]
