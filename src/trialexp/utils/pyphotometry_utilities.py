# Utility functions for pycontrol and pyphotometry files processing
from re import search
from datetime import datetime, timedelta
import warnings

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt, medfilt
from trialexp.utils.rsync import *

from trialexp.process.pycontrol.data_import import session_dataframe
from trialexp.process.pyphotometry.utils import import_ppd
from trialexp.utils.rsync import Rsync_aligner, RsyncError
from loguru import logger

def create_photo_sync(data_pycontrol, photometry_dict):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        photo_rsync = photometry_dict['pulse_times_2']
        
        if 'framework_version' in data_pycontrol.attrs:
            framework_ver = data_pycontrol.attrs['framework_version']
        else:
            # search it in the raw dataframe
            framework_ver = data_pycontrol[data_pycontrol.subtype=='framework_version'].content.iloc[0]
            
        if framework_ver in ['1.8.1','1.8']:
            pycontrol_rsync = data_pycontrol[data_pycontrol.content=='rsync'].time
        else:
            pycontrol_rsync = data_pycontrol[data_pycontrol.subtype=='sync'].time

        
        try:
            return Rsync_aligner(pulse_times_A= photo_rsync, 
                pulse_times_B= pycontrol_rsync, plot=False) #align pycontrol time to pyphotometry time
            
        except (RsyncError, ValueError, ZeroDivisionError) as e:
            logger.info('Cannot align sync pulse. Trying another signal')
            # First try fail, let's try the next one
            if 'pulse_times_3' in photometry_dict:
                photo_rsync = photometry_dict['pulse_times_3']
                try:
                    return Rsync_aligner(pulse_times_A= photo_rsync, 
                        pulse_times_B= pycontrol_rsync, plot=False) #align pycontrol time to pyphotometry time
                    
                except (RsyncError, ValueError,ZeroDivisionError) as e:
                    logger.error('Failed to align sync pulse')
                    return None

def parse_pyhoto_fn(fn):
    pattern = r'(\w+)-(.*)\.ppd'
    m = search(pattern, fn.name)
    if m:
        try:
            subject_id = m.group(1)
            date_string = m.group(2)
            expt_datetime = datetime.strptime(date_string, "%Y-%m-%d-%H%M%S")
            
            return {
                    'subject_id': subject_id, 
                    'path':fn, 
                    'filename':fn.stem, 
                    'timestamp':expt_datetime}    
        except Exception as e:
            print(e)
            return {
                'subject_id': '00', 
                'path':fn, 
                'filename':fn.stem, 
                'timestamp':0} 

#----------------------------------------------------------------------------------
# Plotting
#----------------------------------------------------------------------------------


#----------------------------------------------------------------------------------
# Helpers
#----------------------------------------------------------------------------------

# The exponential curve we are going to fit.
def exp_func(x, a, b, c):
   return a*np.exp(-b*x) + c

# compute the exponential fitted to data
def fit_exp_func(data, fs: int = 100, medfilt_size: int = 3) -> np.ndarray:
    if medfilt_size % 2 == 0:
        raise Exception('medfilt_size must be an odd number') 
    
    time = np.linspace(1/fs, len(data)/fs, len(data))

    fit_params, parm_cov = curve_fit(
        exp_func, time, medfilt(data,medfilt_size),
        p0=[1,1e-3,1],bounds=([0,0,0],[4,0.1,4]), maxfev=1000)

    fitted_data = exp_func(time, * fit_params)

    return fitted_data
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



def match_photometry(df_pycontrol, df_pyphoto):
    matched_photo_path = []
    matched_photo_fn  = []
    
    for _, row in df_pycontrol.iterrows():
    
        # Photometry matching
        # will only compute time diff on matching subject_id
        # First identify the same animal
        if not df_pyphoto.empty:
            df_pyphoto_subject = df_pyphoto[df_pyphoto.subject_id == row.subject_id]
        else:
            matched_photo_path.append(None)
            matched_photo_fn.append(None)
            
        # find the closet match in time
        if not df_pyphoto_subject.empty:
            min_td = np.min(abs(row.timestamp - df_pyphoto_subject.timestamp))
            idx = np.argmin(abs(row.timestamp - df_pyphoto_subject.timestamp))

            if min_td < timedelta(minutes=15):
                matched_photo_path.append(df_pyphoto_subject.iloc[idx].path)
                matched_photo_fn.append(df_pyphoto_subject.iloc[idx].filename)
            else:
                matched_photo_path.append(None)
                matched_photo_fn.append(None)
        
        elif not df_pyphoto.empty and df_pyphoto_subject.empty:
            matched_photo_path.append(None)
            matched_photo_fn.append(None)
            
    df_pycontrol['pyphoto_path'] = matched_photo_path
    df_pycontrol['pyphoto_filename'] = matched_photo_fn