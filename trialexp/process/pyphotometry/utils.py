# Utility functions for pycontrol and pyphotometry files processing

import json

from matplotlib import pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt, medfilt
from scipy.stats import linregress, zscore

from trialexp.utils.rsync import *
from trialexp.process.data_import import Session

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

def motion_correction(photometry_dict: dict) -> dict:
    
    if any(['analog_1_filt' not in photometry_dict, 'analog_2_filt' not in photometry_dict]):
        raise Exception('Analog 1 and Analog 2 must be filtered before motion correction')
    
    slope, intercept, r_value, p_value, std_err = linregress(x=photometry_dict['analog_2_filt'], y=photometry_dict['analog_1_filt'])
    photometry_dict['analog_1_est_motion'] = intercept + slope * photometry_dict['analog_2_filt']
    photometry_dict['analog_1_corrected'] = photometry_dict['analog_1_filt'] - photometry_dict['analog_1_est_motion']
    
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
    time = np.arange(analog_1.shape[0])*1000/sampling_rate # Time relative to start of recording (ms).

   
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
        rsync_chan: int = 2,
        delete_unsynced: bool = True, 
        verbose: bool = False):
    """
    This function create a rsync aligment object into the corresponding
    session if the rsync pulses match betwwen pycontrol and pyphotometry files.

        Parameters:
            session_file (str): PyControl txt file path 
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

    
    if session.files['ppd'] != []:
        # try to align times with rsync
        try:
            # Gives KeyError exception if no rsync pulses on pycontrol file
            pycontrol_rsync_times = session.times['rsync']
        
            photometry_dict = import_ppd(session.files['ppd'][0])
            
            photometry_rsync_times = photometry_dict['pulse_times_' + str(rsync_chan)]

            pyphoto_aligner = Rsync_aligner(pulse_times_A= pycontrol_rsync_times, 
                pulse_times_B= photometry_rsync_times, plot=False)
            
            if verbose:
                print('pycontrol: ', session.subject_ID, session.datetime,
                '/ pyphotometry: ', session.files['ppd'][0], ' : rsync does match')
            
            session.photometry_rsync = pyphoto_aligner

        # if rsync aligner fails    
        except (RsyncError, ValueError, KeyError):
            session.photometry_rsync = None

            if verbose:
                print('pycontrol: ', session.subject_ID, session.datetime,
                '/ pyphotometry: ', session.files['ppd'][0], ' : rsync does not match')

            if delete_unsynced:
                session.files['ppd'] = []

    # if there is no subject + date match in .ppd files
    else: 
        session.photometry_rsync = None

        if verbose:
            print('pycontrol: ', session.subject_ID, session.datetime,
            '/ pyphotometry: no file matching both subject and date')



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
# Processing helper
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

        
