# Utility functions for pycontrol and pyphotometry files processing

import itertools
import json
import logging

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt, medfilt
from scipy.stats import linregress, zscore
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from trialexp.process.pycontrol.event_filters import extract_event_time
from trialexp.utils.rsync import *

'''
Most of the photometry data processing functions are based on the intial design
of the pyPhotometry package. They are stored in a dictionary containing both
metadata and the data. The dictionary is returned by the import_ppd function.

Assumptions:
    - Analog 1 is the isosbestic control
    - Analog 2 is the signal of interest
'''


#----------------------------------------------------------------------------------
# Plotting
#----------------------------------------------------------------------------------


#----------------------------------------------------------------------------------
# Get photometry data by trials
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# Motion correction / Normalization 
#----------------------------------------------------------------------------------

# Note that there is a dependency in the workflow between these filtering 
# and normalization functions. The normalization functions assume that the
# data has already been filtered.

def denoise_filter(photometry_dict:dict, lowpass_freq = 20) -> dict:
    # apply a low-pass filter to remove high frequency noise
    b,a = get_filt_coefs(low_pass=lowpass_freq, sampling_rate=photometry_dict['sampling_rate'])
    analog_1_filt = filtfilt(b, a, photometry_dict['analog_1'], padtype='even')
    analog_2_filt = filtfilt(b, a, photometry_dict['analog_2'], padtype='even')
    
    photometry_dict['analog_1_filt'] = analog_1_filt
    photometry_dict['analog_2_filt'] = analog_2_filt
    
    return photometry_dict
    

def motion_correction(photometry_dict: dict) -> dict:
    
    if any(['analog_1_filt' not in photometry_dict, 'analog_2_filt' not in photometry_dict]):
        raise Exception('Analog 1 and Analog 2 must be filtered before motion correction')
    
    try:
        slope, intercept, r_value, p_value, std_err = linregress(x=photometry_dict['analog_2_filt'], y=photometry_dict['analog_1_filt'])
        photometry_dict['analog_1_est_motion'] = intercept + slope * photometry_dict['analog_2_filt']
        photometry_dict['analog_1_corrected'] = photometry_dict['analog_1_filt'] - photometry_dict['analog_1_est_motion']
    except ValueError:
        print('Motion correction failed. Skipping motion correction')
        # probably due to saturation , do not do motion correction
        photometry_dict['analog_1_corrected'] = photometry_dict['analog_1_filt']


    return photometry_dict

def compute_df_over_f(photometry_dict: dict, low_pass_cutoff: float = 0.001) -> dict:
    
    if 'analog_1_corrected' not in photometry_dict:
        raise Exception('Analog 1 must be motion corrected before computing dF/F')
    
    b,a = butter(2, low_pass_cutoff, btype='low', fs=photometry_dict['sampling_rate'])
    photometry_dict['analog_1_baseline_fluo'] = filtfilt(b,a, photometry_dict['analog_1_filt'], padtype='even')

    # Now calculate the dF/F by dividing the motion corrected signal by the time varying baseline fluorescence.
    photometry_dict['analog_1_df_over_f'] = photometry_dict['analog_1_corrected'] / photometry_dict['analog_1_baseline_fluo'] 
    
    return photometry_dict

#----------------------------------------------------------------------------------
# Filtering 
#----------------------------------------------------------------------------------

def median_filtering(data, medfilt_size: int = 3) -> np.ndarray:
    
    if medfilt_size % 2 == 0:
        raise Exception('medfilt_size must be an odd number') 
    
    data = medfilt(data,medfilt_size)
    
    return data

def get_filt_coefs(low_pass: int = None, high_pass: int = None, sampling_rate = 1000):
    if low_pass and high_pass:
        b, a = butter(2, np.array([high_pass, low_pass])/(0.5*sampling_rate), 'bandpass')
    elif low_pass:
        b, a = butter(2, low_pass/(0.5*sampling_rate), 'low')
    elif high_pass:
        b, a = butter(2, high_pass/(0.5*sampling_rate), 'high')
    
    return b,a

#----------------------------------------------------------------------------------
# Exponential fitting currently not in use in our pipelines
#----------------------------------------------------------------------------------

# The exponential curve we are going to fit.
def exp_func(x, a, b, c):
   return a*np.exp(-b*x) + c


def fit_exp_func(data, fs: int = 100, medfilt_size: int = 3) -> np.ndarray:
    '''
    compute the exponential fitted to data. This unused in current filtering because
    unsuitable when behavioural box openings / closing provoked transitory changes
    in baseline fluorescence.
    '''
    if medfilt_size % 2 == 0:
        raise Exception('medfilt_size must be an odd number') 
    
    time = np.linspace(1/fs, len(data)/fs, len(data))

    fit_params, parm_cov = curve_fit(
        exp_func, time, medfilt(data,medfilt_size),
        p0=[1,1e-3,1],bounds=([0,0,0],[4,0.1,4]), maxfev=1000)

    fitted_data = exp_func(time, * fit_params)


    return fitted_data


#----------------------------------------------------------------------------------
# Load analog data
#----------------------------------------------------------------------------------

def import_ppd(file_path):
    '''Function to import pyPhotometry binary data files into Python. The high_pass 
    and low_pass arguments determine the frequency in Hz of highpass and lowpass 
    filtering applied to the filtered analog signals. To disable highpass or lowpass
    filtering set the respective argument to None.  Returns a dictionary with the 
    following items:
        'subject_ID'    - Subject ID
        'date_time'     - Recording start date and time (ISO 8601 format string)
        'mode'          - Acquisition mode
        'sampling_rate' - Sampling rate (Hz)
        'LED_current'   - Current for LEDs 1 and 2 (mA)
        'version'       - Version number of pyPhotometry
        'analog_1'      - Raw analog signal 1 (volts)
        'analog_2'      - Raw analog signal 2 (volts)
        'analog_1_filt' - Filtered analog signal 1 (volts)
        'analog_2_filt' - Filtered analog signal 2 (volts)
        'digital_1'     - Digital signal 1
        'digital_2'     - Digital signal 2
        'pulse_inds_1'  - Locations of rising edges on digital input 1 (samples).
        'pulse_inds_2'  - Locations of rising edges on digital input 2 (samples).
        'pulse_times_1' - Times of rising edges on digital input 1 (ms).
        'pulse_times_2' - Times of rising edges on digital input 2 (ms).
        'time'          - Time of each sample relative to start of recording (ms)
    '''
    with open(file_path, 'rb') as f:
        header_size = int.from_bytes(f.read(2), 'little')
        data_header = f.read(header_size)
        data = np.frombuffer(f.read(), dtype=np.dtype('<u2'))
    # Extract header information
    header_dict = json.loads(data_header)
    volts_per_division = header_dict['volts_per_division']
    sampling_rate = header_dict['sampling_rate']
    # Extract signals.
    analog  = data >> 1                     # Analog signal is most significant 15 bits.
    digital = ((data & 1) == 1).astype(int) # Digital signal is least significant bit.
    # Alternating samples are signals 1 and 2.
    analog_1 = analog[ ::2] * volts_per_division[0]
    analog_2 = analog[1::2] * volts_per_division[1]
    digital_1 = digital[ ::2]
    digital_2 = digital[1::2]
    # Time relative to start of recording (ms).
    time = np.arange(analog_1.shape[0]).astype(np.int64)*1000/sampling_rate #warning: default data type np.int32 will lead to overflow
    # time = np.arange(analog_1.shape[0])*1000/sampling_rate #warning: default data type np.int32 will lead to overflow

   
    # Extract rising edges for digital inputs.
    pulse_inds_1 = 1+np.where(np.diff(digital_1) == 1)[0]
    pulse_inds_2 = 1+np.where(np.diff(digital_2) == 1)[0]
    pulse_times_1 = pulse_inds_1*1000/sampling_rate
    pulse_times_2 = pulse_inds_2*1000/sampling_rate
    # Return signals + header information as a dictionary.
    data_dict = {'analog_1'      : analog_1,
                 'analog_2'      : analog_2,
                 'digital_1'     : digital_1,
                 'digital_2'     : digital_2,
                 'pulse_inds_1'  : pulse_inds_1,
                 'pulse_inds_2'  : pulse_inds_2,
                 'pulse_times_1' : pulse_times_1,
                 'pulse_times_2' : pulse_times_2,
                 'time'          : time}
    
    # Add metadata to dictionary.
    data_dict.update(header_dict)
    
    return data_dict

#----------------------------------------------------------------------------------
# Rsync functions
#----------------------------------------------------------------------------------


def sync_photometry_file(
        session_file: str,
        photometry_file: str = None, 
        rsync_chan: int = 2,
        delete_unsynced: bool = True, 
        verbose: bool = False):
    """
    This function create a rsync aligment object into the corresponding
    session if the rsync pulses match betwwen pycontrol and pyphotometry files.

        Parameters:
            session_file (str): PyControl txt file path 
            photometry_file (str): PyPhotometry ppd file path 
            rsync_chan (int): Channel on which pulses have been
                recorded on the py_photometry device.
            delete_unsynced (bool): Delete the photometry file path in
                session.files['ppd'] if rsync does not match
            verbose (bool): display match/no match messages for each file

        Returns:
            None

        The warning:
            KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads...
        
        is due to rsync function.

        https://stackoverflow.com/questions/69596239/how-to-avoid-memory-leak-when-dealing-with-kmeans-for-example-in-this-code-i-am
        Follow the answer and set the einvironment variable OMP_NUM_THREADS to supress the warning.
                
    """
    
    session = Session(session_file, int_subject_IDs=True, verbose=False) 

    session.files = dict()
    session.files['ppd']  = [photometry_file] # list to make it backward compatible (implemented to allow for multiple matches [eg cameras])
    if photometry_file:
        # try to align times with rsync
        try:
            # Gives KeyError exception if no rsync pulses on pycontrol file
            pycontrol_rsync_times = session.times['rsync']
        
            photometry_dict = import_ppd(photometry_file)
            
            photometry_rsync_times = photometry_dict['pulse_times_' + str(rsync_chan)]

            photometry_rsync = Rsync_aligner(pulse_times_A= pycontrol_rsync_times, 
                pulse_times_B= photometry_rsync_times, plot=False)
            
            if verbose:
                print('pycontrol: ', session.subject_ID, session.datetime,
                '/ pyphotometry: ', photometry_file, ' : rsync does match')
            


        # if rsync aligner fails    
        except (RsyncError, ValueError, KeyError):
            photometry_rsync = None

            if verbose:
                print('pycontrol: ', session.subject_ID, session.datetime,
                '/ pyphotometry: ', photometry_file, ' : rsync does not match')

            if delete_unsynced:
                session.files['ppd'] = []

    # if there is no subject + date match in .ppd files
    else: 
        photometry_rsync = None

        if verbose:
            print('pycontrol: ', session.subject_ID, session.datetime,
            '/ pyphotometry: no file matching both subject and date')

    # for now return a session with embedded rsync object. 
    # Ouput will change when getting closer to fully functional implementation
    return photometry_rsync

#----------------------------------------------------------------------------------
# From here, legacy methods which will be probably deprecated in the future
#----------------------------------------------------------------------------------


def find_n_gaussians(
        data: np.ndarray,
        plot_results: bool = True,
        max_nb_gaussians: int = 4
    ) -> int: 
    '''
    Function to detect how many gaussians are needed to
    decribe a dataset.
    Re-use the original part to find M_best when there is
    more than 2-3 gaussians, and adjust the N range.
    Uncomment first original part to create artificial dataset

    Original author: Jake VanderPlas
    License: BSD
        The figure produced by this code is published in the textbook
        "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
        For more information, see http://astroML.github.com
        To report a bug or issue, use the following forum:
        https://groups.google.com/forum/#!forum/astroml-general
    
    '''
    #------------------------------------------------------------
    # Set up the dataset. 

    # median filter of raw red channel to remove small electric
    # data = medfilt(data,3)


    X = data.reshape(-1, 1)

    # (original code: We'll create our dataset by drawing samples from Gaussians)
    # random_state = np.random.RandomState(seed=1)

    # X = np.concatenate([random_state.normal(-1, 1.5, 350),
    #                     random_state.normal(0, 1, 500),
    #                     random_state.normal(3, 0.5, 150)]).reshape(-1, 1)

    #------------------------------------------------------------
    # Learn the best-fit GaussianMixture models
    #  Here we'll use scikit-learn's GaussianMixture model. The fit() method
    #  uses an Expectation-Maximization approach to find the best
    #  mixture of Gaussians for the data

    # fit models with 1-10 components
    N = np.arange(1, max_nb_gaussians)
    models = [None for i in range(len(N))]

    for i in range(len(N)):
        models[i] = GaussianMixture(N[i]).fit(X)

    # compute the AIC and the BIC
    AIC = [m.aic(X) for m in models]
    # BIC = [m.bic(X) for m in models]

    #------------------------------------------------------------
    # Plot the results
    #  We'll use three panels:
    #   1) data + best-fit mixture
    #   2) AIC and BIC vs number of components
    #   3) probability that a point came from each component

    # Original part, take the min of AIC to determine how many gaussians
    # M_best = models[np.argmin(AIC)]

    # Customized part tweaked to reduce the nb of gaussian used to the minimum
    diff_AIC = np.diff(np.diff(AIC))
    print(diff_AIC)  
    if diff_AIC[0] < 0:
        n_best = 0
    else:
        n_best = np.where(diff_AIC == min(diff_AIC))[0][0]+2
    

    M_best = models[n_best]
    # end of customized part

    p, bins = np.histogram(X, bins=np.arange(min(X),max(X),0.0002), density=True)
    print(len(bins))
    x = bins
    logprob = M_best.score_samples(x.reshape(-1, 1))
    # logprob = M_best.score_samples(x)

    responsibilities = M_best.predict_proba(x.reshape(-1, 1))
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    if plot_results:
        fig = plt.figure(figsize=(15, 5))
        # fig.subplots_adjust(left=0.12, right=0.97,
        #                     bottom=0.21, top=0.9, wspace=0.5)


        # plot 1: data + best-fit mixture
        ax = fig.add_subplot(121)

        ax.plot(x[:-1], p, 'r') # approximation
        ax.plot(x, pdf, '-k')
        ax.plot(x, pdf_individual, '--k')
        ax.text(0.04, 0.96, f'Best-fit Mixture n={n_best+1}',
                ha='left', va='top', transform=ax.transAxes)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$p(x)$')


        # plot 2: AIC and BIC
        ax = fig.add_subplot(122)
        ax.plot(N, AIC, '-k', label='AIC')
        # ax.plot(N, BIC, '--k', label='BIC')
        ax.set_xlabel('n. components')
        ax.set_ylabel('information criterion')
        ax.legend(loc=2)

        plt.show()

    return n_best+1

#----------------------------------------------------------------------------------
# Processing helper remaining from legacy
#----------------------------------------------------------------------------------

def compute_PCA(
        data: np.ndarray
    ):
    
    scaler = StandardScaler()
    pca = PCA(0.7, random_state=33)
    pca.fit(scaler.fit_transform(X.iloc[past_id]))
    
    Xt = pca.inverse_transform(
        pca.transform(
            scaler.transform(X.iloc[future_id])
        ))


def dbscan_anomaly_detection(data):

    ### DBSCAN ANOMALY DETECTION ###

    network_ano = {}
    dbscan = DBSCAN(eps=0.6, min_samples=1, metric="precomputed")

    for trial_idx in range(data.shape[0]+1):
    
        trial = data[trial_idx, :,:].squeeze()
        preds = dbscan.fit_predict(
            pairwise_distances(trial, metric='correlation')
        )
        if (preds > 0).any():
            ano_features = list(X.columns[np.where(preds > 0)[0]])
            network_ano[past_id[-1]] = ano_features
        else:
            network_ano[past_id[-1]] = None

        

def photometry2xarray(data_photometry, skip_var=None):
    """
    Converts a pyphotometry dictionary into an xarray dataset. 
    
    Parameters
    ----------
    data_photometry : dict
        A pyphotometry dictionary containing data and associated time stamps.
    skip_var: list
        name of keyword in the data_photometry dict that you want to skip, mainly use to skip intermeidate variables
        
    Returns
    -------
    dataset : xarray.Dataset
        An xarray Dataset containing data and attributes associated with the 
        pyphotometry dictionary. 
    """ 
    
    data_list = {}
    attr_list = {}
    time = data_photometry['time'].astype(np.int64)
    
    if skip_var is None:
        skip_var = []

    for k, data in data_photometry.items():
        if not k in skip_var:
            if isinstance(data, (list,np.ndarray)) and len(data) == len(time):
                    array = xr.DataArray(data, coords={'time':time}, dims=['time'])
                    data_list[k] = array
            else:
                attr_list[k] = data

    dataset = xr.Dataset(data_list)
    dataset.attrs.update(attr_list)
    
    return dataset

def resample_event(pyphoto_aligner, ref_time, event_time, event_value, fill_value=-1):
    """
    Resample an event to a reference time.

    Parameters
    ----------
    pyphoto_aligner : object
        An instance of the Rsync_aligner class.
    ref_time : array-like
        Reference time points.
    event_time : array-like
        Event time points.
    event_value : array-like
        Event values corresponding to the event time points.
    fill_value : float, optional
        Value used to fill in for requested points outside of the range of event_time. The default is -1.

    Returns
    -------
    array-like
        Resampled event values corresponding to the reference time points.
        can contain NaN value if there is no overlap data
    """
    
    new_time = pyphoto_aligner.A_to_B(event_time)
    f = interp1d(new_time, event_value, kind = 'previous', 
                bounds_error=False, fill_value=fill_value)
    
    return f(ref_time)


def extract_event_data(trigger_timestamp, window, aligner, dataArray, sampling_rate, data_len =None, time_tolerance=5):
    '''
    Extract continous data around a timestamp. The original timestamp will be
    aligned to the coordinate of the dataArray with aligner
    
    Parameters:
        trigger_timestamp : float or int
            Timestamp around which data has to be extracted, in ms
        window : tuple
            Tuple containing minimum and maximum value of time window for which data is to be extracted
        aligner : object
            Object containing A_to_B() method for alignment of timestamp
        dataArray : array-like
            Array containing data from which required data is to be extracted. It it assumed to have a time coordinate in ms
        data_len : int, optional
            If provided, checks for length of output data
        time_tolerance: int, default=5
            the minimum time difference in ms that must be matched between the trigger stampstamp and the time coordinate of the dataArray
    
    Returns:
        data : numpy.ndarray
            Array of continuous data around the timestamp
        event_found : list
            List of boolean values indicating if any event was found around the given timestamp

    '''

    
    ts = aligner.A_to_B(trigger_timestamp)
    ref_time = dataArray.time
    data = []
    event_found = []
    pre_time_shift = int(window[0]/1000*sampling_rate)
    post_time_shift = int(window[1]/1000*sampling_rate)
    
    for t in ts:
        d = abs((ref_time-t).data)
        #Find the most close matched time stamp and extend it both ends 
        min_idx = np.argmin(d)
        min_time = d[min_idx]
        
        start_idx = min_idx + pre_time_shift
        end_idx = min_idx + post_time_shift
        
        # make sure we have enough data to extract
        if min_time < time_tolerance and (start_idx>0) and (end_idx < len(dataArray.data)):

            data.append(dataArray.data[start_idx:end_idx])
            event_found.append(True)
        else:
            x = np.zeros((int((window[1]-window[0])/1000*sampling_rate),))*np.NaN
            data.append([x])
            event_found.append(False)
            
    # align to the longest element
    # data =  np.vstack(list(itertools.zip_longest(*data)))
    data  = np.vstack(data)
    
    # if data_len is provide, perform additional check or correct the data length
    if data_len is not None:
        if not data.shape[0]==data_len:
            data = data[:data_len,:]
    
    return data.astype(float),event_found #only float support NA

#%% Calulate the relative time
def get_rel_time(trigger_timestamp, window, aligner, ref_time):
    # Calculate the time relative to a trigger timestamp)
    ts = aligner.A_to_B(trigger_timestamp)
    time_relative = np.ones_like(ref_time)*np.NaN
    
    for t in ts: 
        d = ref_time-t
        idx = (d>window[0]) & (d<window[1])
        time_relative[idx] = d[idx]
        
    return time_relative


def bin_rel_time(xr_dataset, bin_size):
    """Bins relative time in the input Xarray dataset to the given bin size.
    
    
    we need to do some special treatment to the relative time because the time stamp for that may not fall in the 
    same time bin, and hence the mean value of them will be different for different trial
    this will create problem with plotting and analysis later, so we need to fix it now

    Args:
        xr_dataset (xarray.core.dataset.Dataset): The input Xarray dataset.
        bin_size (float): The size of each bin.

    Returns:
        xarray.core.dataset.Dataset: The binned Xarray dataset.
    """
    for k in xr_dataset.data_vars.keys():
        if 'rel_time' in k:
            xr_dataset[k] = np.round(xr_dataset[k]/bin_size)*bin_size
            
    return xr_dataset


def bin_dataset(xr_dataset, bin_size, sampling_fs=1000):
    """
    Bin the input xarray dataset by grouping data within specified time intervals.
    
    Args:
    xr_dataset (xarray.Dataset): Input xarray dataset to be binned
    time_bin (float): Width of each time bin for grouping data, in ms
    
    Returns:
    dataset_binned (xarray.Dataset): Binned xarray dataset
    """
    
    ds_factor = int((bin_size/1000)*sampling_fs)
    logging.debug(f'Downsampling testby {ds_factor}')

    dataset_binned = xr_dataset.coarsen(time=ds_factor, boundary='trim').mean()

    dataset_binned = bin_rel_time(dataset_binned, bin_size)
    dataset_binned.attrs.update(xr_dataset.attrs)
    
    return dataset_binned


def make_condition_xarray(df_condition, dataset_binned):
    """
    Merge condition for each trial to the xarray
    
    Args:
    -----------------------------
    df_condition : pd.DataFrame
        Dataframe containing trial specific variables (e.g. condition to be analyzed)
    dataset_binned : xr.dataset
        Xarray dataset that has already been binned
    
    Returns:
    -----------------------------
    xr_condition : xr.dataset
        Dataset with the condition merged in as a new dimension
    
    """
    
    df_trial_nb = dataset_binned.trial_nb.to_dataframe()
    df_trial_nb['trial_nb'] = df_trial_nb['trial_nb'].astype(np.int16)
    df_trial_condition = df_trial_nb.merge(df_condition, on='trial_nb')
    df_trial_condition['time'] = dataset_binned.time
    df_trial_condition = df_trial_condition.set_index('time')
    xr_condition = df_trial_condition.to_xarray()
    
    return xr_condition



def make_rel_time_xr(event_time, windows, pyphoto_aligner, ref_time):
    time_rel = get_rel_time(event_time, windows, pyphoto_aligner, ref_time)

    rel_time = xr.DataArray(
        time_rel, coords={'time':ref_time}, dims=('time')
    )
    
    return rel_time

def make_event_xr(event_time, trial_window, pyphoto_aligner,
                  event_time_coordinate,  dataArray, sampling_rate):
    '''
    Create xarray.DataArray object for the continuous data around provided timestamp. 

    Parameters:
        event_time : array_like
            List of timestamps around which continuous data is to be extracted
            it is assumed to have a index corresponds to the trial number
        trial_window : tuple
            Tuple containing minimum and maximum value of time window for which data is to be extracted
        pyphoto_aligner : Object
            Object containing A_to_B() method for alignment of timestamp
        event_time_coordinate : array_like
            List of trial numbers corresponding to each event_time
        dataArray : array_like
            Array containing time and data information
            
    Returns:
        da : xarray.DataArray
            DataArray object containing continuous data around the provided timestamp
    
    Note:
        It returns only data from extract_event_data() function ignoring the event_found.
    '''
    if len(event_time)==0:
        return None
    
    assert event_time.index.name =='trial_nb', 'event_time should have a trial_nb index'
    data, _ = extract_event_data(event_time, trial_window, pyphoto_aligner,
                                dataArray, sampling_rate)
    da = xr.DataArray(
        data, coords={'event_time':event_time_coordinate, 
                      'trial_nb':event_time.index.values},
                        dims=('trial_nb','event_time'))
        
    return da

def add_event_data(df_event, filter_func, trial_window, aligner,
                   dataset, event_time_coordinate, data_var_name, 
                   event_name, sampling_rate, filter_func_kwargs={}):
    '''
    Add continuous data around provided timestamp to a dataset.

    Parameters:
        df_event : pandas.DataFrame
            DataFrame containing event and timestamp information
        filter_func : function
            Function to filter particular events from the dataframe
        trial_window : tuple
            Tuple containing minimum and maximum value of time window for which data is to be extracted.
        aligner : Object
            Object containing A_to_B() method for alignment of timestamp
        dataset : xarray.Dataset
            Dataset in which data is to be added
        event_time_coordinate : array_like
            relative time coordinate in ms of the continous data
        data_var_name : str
            The variable name of the data in `dataset`.
        event_name : str
            name of the event
        sampling_rate: int
            sampling rate of the continous signal
        filter_func_args: dict
            argument that should be passed to the filter function
        
    Returns:
        None

    Note:
        It updates the given `dataset` with a new variable that contains continuous data around 
        each timestamp.  It requires two supporting functions make_event_xr() & extract_event_time().
    '''

    event_time = extract_event_time(df_event, filter_func, filter_func_kwargs)
    xr_event_data = make_event_xr(event_time, trial_window, aligner,
                                            event_time_coordinate,
                                            dataset[data_var_name], sampling_rate)
    
    if xr_event_data is not None:
        dataset[f'{event_name}_{data_var_name}'] = xr_event_data
# %%
