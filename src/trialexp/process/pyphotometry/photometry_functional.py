import pandas as pd
from scipy.signal import butter, filtfilt, decimate

from trialexp.process.pyphotometry.utils import *
from trialexp.utils.pycontrol_utilities import *
# from trialexp.utils.rsync import *



def get_photometry_trials(
        session, #: Session, To be integrated with functional trial-processed Session Outputs
        conditions_list: list = None,
        cond_aliases: list = None,
        trial_window: list = None,
        trig_on_ev: str = None,
        last_before: str = None,
        baseline_low_pass: float = None, 
        low_pass: int = None, 
        median_filt: int = None, 
        motion_corr: bool = False, 
        df_over_f: bool = False,
        z_score: bool = False,
        downsampling_factor: int = None,
        return_full_session: bool = False, 
        export_vars: list = ['analog_1','analog_2'],
        # remove_artifacts: bool = False,
        verbose: bool = False):
        
    # TODO write docstrings
    
    if not isinstance(conditions_list, list):
        conditions_list= [conditions_list] 
    
    if not isinstance(export_vars, list):
        export_vars= [export_vars] 

    if not hasattr(session, 'photometry_rsync'):
        raise Exception('The session has not been matched with a .ppd file, \
            please run experiment.match_to_photometry_files(kvargs)')
    elif session.photometry_rsync == None:
        raise Exception('The session has no matching .ppd file, or no alignment \
            could be performed between rsync pulses')
    
    if motion_corr == True and baseline_low_pass == None and low_pass == None and median_filt == None:
        raise Exception('You need to high_pass and/or low_pass and/or median_filt the signal for motion correction')
    if df_over_f == True and motion_corr == False:
        raise Exception('You need motion correction to compute dF/F')

    # import of raw and filtered data from full photometry session

    photometry_dict = import_ppd(session.files['ppd'][0])

    #----------------------------------------------------------------------------------
    # Filtering / Motion correction / resampling block below
    #----------------------------------------------------------------------------------
    if low_pass:
            # Filter signals with specified high and low pass frequencies (Hz).
            b, a = get_filt_coefs(low_pass=low_pass, high_pass=None, sampling_rate=photometry_dict['sampling_rate'])
            
            if median_filt:
                analog_1_medfilt = median_filtering(photometry_dict['analog_1'], medfilt_size = median_filt)
                analog_2_medfilt = median_filtering(photometry_dict['analog_2'], medfilt_size = median_filt)
                photometry_dict['analog_1_filt'] = filtfilt(b, a, analog_1_medfilt)
                photometry_dict['analog_2_filt'] = filtfilt(b, a, analog_2_medfilt)

            else:
                photometry_dict['analog_1_filt'] = filtfilt(b, a, photometry_dict['analog_1'])
                photometry_dict['analog_2_filt'] = filtfilt(b, a, photometry_dict['analog_2'])
    else:
        if median_filt:
            photometry_dict['analog_1_filt'] = median_filtering(photometry_dict['analog_1'], medfilt_size = median_filt)
            photometry_dict['analog_2_filt'] = median_filtering(photometry_dict['analog_2'], medfilt_size = median_filt)  
        else:
            photometry_dict['analog_1_filt'] = photometry_dict['analog_2_filt'] = None
    # TODO: verify/improve/complement the implementation of the following:
    # https://github.com/ThomasAkam/photometry_preprocessing/blob/master/Photometry%20data%20preprocessing.ipynb
    
    if motion_corr == True:

        slope, intercept, r_value, p_value, std_err = linregress(x=photometry_dict['analog_2_filt'], y=photometry_dict['analog_1_filt'])
        photometry_dict['analog_1_est_motion'] = intercept + slope * photometry_dict['analog_2_filt']
        photometry_dict['analog_1_corrected'] = photometry_dict['analog_1_filt'] - photometry_dict['analog_1_est_motion']
        
        if df_over_f == False:
            export_vars.append('analog_1_corrected')
            # signal = photometry_dict['analog_1_corrected']
        elif df_over_f == True:
            # fror 
            b,a = butter(2, baseline_low_pass, btype='low', fs=photometry_dict['sampling_rate'])
            photometry_dict['analog_1_baseline_fluo'] = filtfilt(b,a, photometry_dict['analog_1_filt'], padtype='even')

            # Now calculate the dF/F by dividing the motion corrected signal by the time varying baseline fluorescence.
            photometry_dict['analog_1_df_over_f'] = photometry_dict['analog_1_corrected'] / photometry_dict['analog_1_baseline_fluo'] 
            export_vars.append('analog_1_df_over_f')
            # signal = photometry_dict['analog_1_df_over_f']
    if z_score:
        # z-score the signal
        photometry_dict['zscored_df_over_f'] = zscore(photometry_dict['analog_1_df_over_f'])
        export_vars.append('zscored_df_over_f')
    elif baseline_low_pass or low_pass:
        # signal = photometry_dict['analog_1_filt']']
        export_vars.append('analog_1_filt')

        # control = photometry_dict['analog_2_filt']']
    else:
        export_vars.append('analog_1')
    # signal = photometry_dict['analog_1']']

    # only keep unique items (keys for the photometry_dict)
    export_vars = list(set(export_vars))

    if downsampling_factor:
        # downsample
        for k in export_vars:
            photometry_dict[k] = decimate(photometry_dict[k], downsampling_factor)
        # adjust sampling rate accordingly (maybe unnecessary)
        photometry_dict['sampling_rate'] = photometry_dict['sampling_rate'] / downsampling_factor

    fs = photometry_dict['sampling_rate']

    df_meta_photo = pd.DataFrame(columns=['subject_ID', 'datetime', 'task_name', 'condition_ID', 'trial_nb'])
    
    # Prepare dictionary output with keys are variable names and values are columns index
    col_names_numpy = {var: var_idx for var_idx, var in enumerate(export_vars)}
    
    # TODO: must be more effective to do that process a single time and then sort by / attribute conditions
    for condition_ID, conditions_dict in enumerate(conditions_list):
        # TEST the option of triggering on the first event of a trial

        trials_idx, timestamps_pycontrol = session.get_trials_times_from_conditions(conditions_dict, trig_on_ev=trig_on_ev, last_before=last_before)
        
        if len(trials_idx) == 0 :
            continue

        timestamps_photometry = session.photometry_rsync.A_to_B(timestamps_pycontrol)
        photometry_idx = (timestamps_photometry / (1000/photometry_dict['sampling_rate'])).round().astype(int)
        

        # print(f'photometry {photometry_idx[0:10]} \r\n pycontrol {timestamps_pycontrol[0:10]}')
        
        # Compute mean time difference between A and B and replace NaN values at extremity of files
        # (rsync_aligner timestamps are only computed BETWEEN rsync pulses)
        # NOTE: Can be uncommented but likely to reintroduce timestamps outside of photometry timestamps span
        # nan_idx = isnan(timestamps_photometry)
        # mean_diff = nanmean(timestamps_photometry - timestamps_pycontrol.astype(float))
        # timestamps_photometry[nan_idx] = timestamps_pycontrol[nan_idx] + mean_diff
        # timestamps_photometry = timestamps_photometry.astype(int)

        # retain only trials with enough values left and right
        complete_mask = (photometry_idx + trial_window[0]/(1000/photometry_dict['sampling_rate']) >= 0) & (
            photometry_idx + trial_window[1] < len(photometry_dict[export_vars[0]])) 

        # complete_idx = np.where(complete_mask)
        trials_idx = np.array(trials_idx)
        photometry_idx = np.array(photometry_idx)

        trials_idx = trials_idx[complete_mask]           
        photometry_idx = photometry_idx[complete_mask]
        
        if verbose:
            print(f'condition {condition_ID} trials: {len(trials_idx)}')
        
        if len(trials_idx) == 0 :
            continue

        # Construct ranges of idx to get chunks (trials) of photometry data with np.take method 
        photometry_idx = [range(idx + int(trial_window[0]/(1000/photometry_dict['sampling_rate'])) ,
            idx + int(trial_window[1]/(1000/photometry_dict['sampling_rate']))) for idx in photometry_idx]

        if condition_ID == 0:
            # initialization of 3D numpy arrays (for all trials)
            # Dimensions are M (number of trials) x N (nb of samples by trial) x P (number of variables,
            # e.g.: photometry_dict['analog_1_filt'], analog_1_df_over_f, etc)

            photo_array = np.ndarray((len(trials_idx), len(photometry_idx[0]),len(export_vars)))

            for var_idx, photo_var in enumerate(export_vars):
                # print(f'condition {condition_ID} var: {var_idx} shape {np.take(photometry_dict[photo_var], photometry_idx).shape}')
                photo_array[:,:,var_idx] = np.take(photometry_dict[photo_var], photometry_idx)


            df_meta_photo['trial_nb'] = trials_idx
            df_meta_photo['subject_ID'] = session.subject_ID
            df_meta_photo['datetime'] = session.datetime
            df_meta_photo['task_name'] = session.task_name
            df_meta_photo['condition_ID'] = condition_ID

        else:
            # initialization of temp 3D numpy arrays (for subset of trials by conditions)
            # Dimensions are M (number of trials) x N (nb of samples by trial) x P (number of variables,
            # e.g.: photometry_dict['analog_1_filt'], analog_1_df_over_f, etc)

            photo_array_temp = np.ndarray((len(trials_idx), len(photometry_idx[0]),len(export_vars)))

            for var_idx, photo_var in enumerate(export_vars):
                # print(f'condition {condition_ID} var: {var_idx} shape {np.take(photometry_dict[photo_var], photometry_idx).shape}')
                photo_array_temp[:,:,var_idx]  = np.take(photometry_dict[photo_var], photometry_idx)


            df_meta_photo_temp = pd.DataFrame(columns=['subject_ID', 'datetime', 'task_name', 'condition_ID', 'trial_nb'])

            df_meta_photo_temp['trial_nb'] = trials_idx
            df_meta_photo_temp['subject_ID'] = session.subject_ID
            df_meta_photo_temp['datetime'] = session.datetime
            df_meta_photo_temp['task_name'] = session.task_name
            df_meta_photo_temp['condition_ID'] = condition_ID      
            # concatenate with previous dataframe
            df_meta_photo = pd.concat([df_meta_photo, df_meta_photo_temp], axis=0, ignore_index=True)
            # concatenate with previous numpy array
            
            # Take into account when there is no trial in first condition
            if 'photo_array' in locals():
                photo_array = np.concatenate((photo_array, photo_array_temp), axis=0)
            else:
                photo_array = photo_array_temp



    if 'photo_array' in locals():
        photo_array = photo_array.swapaxes(2,1)
    else:
        # This occurs when no photometry data is recored at all for the session
        # would occur anyway without the previous check, 
        # avoid it happening spontaneously on return.
        # useless but could be use to convey extra information to calling method
        
        if verbose:
            print(f'No photometry data to collect for subject ID:{session.subject_ID}\
                \nsession: {session.datetime}')

        raise UnboundLocalError()

        # Trying to implement empty arrays and dataframe when nothing to return
        # df_meta_photo = pd.DataFrame(columns=['subject_ID', 'datetime', 'task_name', 'condition_ID', 'trial_nb'])
        # ra
        # photo_array = np.ndarray((len(trials_idx), len(photometry_idx),len(export_vars)))

    if return_full_session == False:
        return df_meta_photo, col_names_numpy, photo_array, fs
    else:
        return df_meta_photo, col_names_numpy, photo_array, photometry_dict


def get_trials_times_from_conditions(
        session,
        conditions_dict: dict = None, 
        trig_on_ev: str = None,
        last_before: str = None):
    '''
    Get the indices and timestamps of the trials matching a set dict of conditions,
    offsetted (or not) by the first (or last before) occurence of a particular event
    '''
    if conditions_dict is not None:
        cond_pairs = list(conditions_dict.items())
        idx_rows = []

        for idx_cond, cond in enumerate(cond_pairs):
            # create a list of set of index values for each conditions
            idx_rows.append(set(session.df_conditions.index[session.df_conditions[cond_pairs[idx_cond][0]] == cond_pairs[idx_cond][1]].values))

        # compute the intersection of the indices of all conditions requested
        idx_joint = list(set.intersection(*idx_rows))
        idx_joint.sort()
    else:
        idx_joint = session.df_conditions.index.values

    # if output_first_ev and not isinstance(trig_on_ev, str):
    #     raise NotPermittedError('get_trials_times_from_conditions',
    #         ['trig_on_ev','output_first_ev'], [trig_on_ev,output_first_ev])

    # if no trig_on_ev specified, return the original trigger timestamp
    if trig_on_ev == None:
        trials_times = session.df_events.loc[(idx_joint),'timestamp'].values
    elif trig_on_ev not in session.events_to_process:
        raise Exception('trig_on_ev not in events_to_process')
    # Otherwise offset trigger timestamp by occurence of first event (TODO: or last before)
    else:    
        trials_times = session.df_events.loc[(idx_joint), 'timestamp'].copy()
        
        df_ev_copy = session.df_events.copy()
        # TODO: continue implementation
        if last_before is not None and last_before in set(session.events_to_process):

            if len(idx_joint) == 0: # Added because find_last_time_before_list did not do well with empty Df
                ev_times = pd.DataFrame()
            else:
                ev_col = trig_on_ev + '_trial_time'
                before_col = last_before + '_trial_time'

                ev_times = df_ev_copy.loc[(idx_joint), [ev_col, before_col]].apply(
                    lambda x: find_last_time_before_list(x[ev_col], x[before_col]), axis=1)               
            
        # If last_before is not requested
        else:

            ev_times = df_ev_copy.loc[(idx_joint), trig_on_ev + '_trial_time'].apply(
                lambda x: find_min_time_list(x))
            

        #keep ony trial_times with a first event
        ev_times_nona = ev_times.dropna(inplace=False)
        
        # Add time of the first event to trial onset
        trials_times = trials_times.loc[ev_times_nona.index] + ev_times_nona
        # trials_times.dropna(inplace=True)
        # Keep only the index where events (trig_on_ev) were found
        idx_joint = trials_times.index.values.astype(int)
        # retransmorm trial_times as an array
        trials_times = trials_times.values.astype(int)
        # ev_times = ev_times_nona.astype(int).values
    
        #print(idx_joint.shape,first_ev_times_nona.shape, first_ev_times.shape)
    # if output_first_ev:
    #     return idx_joint, trials_times
    # else:    
    return idx_joint, trials_times
