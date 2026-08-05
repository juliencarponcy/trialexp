
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd   
import os
from tqdm.auto import tqdm
import xarray as xr
from trialexp.process.ephys.spikes_preprocessing import build_evt_fr_xarray
from elephant.conversion import BinnedSpikeTrain
import quantities as pq
from trialexp.process.figures.plot_utils import create_plot_grid
from trialexp.process.group_analysis.plot_utils import style_plot
import seaborn as sns
import neo 
from scipy.stats import ttest_ind, wilcoxon, ranksums, permutation_test, variation
from statsmodels.stats.multitest import multipletests
from sklearn import decomposition, manifold, preprocessing, cluster
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from scipy import signal, spatial
from fastdtw import fastdtw
from tslearn.barycenters import softdtw_barycenter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pingouin as pg
from trialexp.process.ephys.spikes_preprocessing import get_spike_trains
from tslearn.backend import instantiate_backend
from tslearn.utils import to_time_series
from tslearn.metrics import dtw_variants
from tslearn.metrics.dtw_variants import *
from tslearn.barycenters import softdtw_barycenter
import scipy
from scipy.stats import pearsonr



def denest_string_cell(cell):
        if len(cell) == 0: 
            return 'ND'
        else:
            return str(cell[0])


def analyzer2dataframe(analyzer, extension2skip=['waveforms']):
    # Convert the sorting analyzer object in spikeinterface to dataframe
    units_ids = analyzer.unit_ids
    metrics = {}
    df2join=[]
    other_metrics = {}
    for extension in analyzer.get_loaded_extension_names():
        if extension in extension2skip:
            continue
        
        wv = analyzer.get_extension(
            extension_name=extension
        )
        data = wv.get_data()
        # print(extension, type(data))
        if type(data) is np.ndarray and data.shape[0] == len(units_ids):
            metrics[extension] = data.tolist()
        elif type(data) is pd.core.frame.DataFrame:
            df2join.append(data)
        else:
            if extension != 'waveforms':
                other_metrics[extension] = data
            

    df_metrics = pd.DataFrame(metrics)
    df_metrics['unit_id'] = units_ids
    for df in df2join:
        df['unit_id'] = units_ids
        df_metrics = df_metrics.merge(df, on='unit_id')
        
    #Special processing for correlogram
    # only extract the auto-correlogram
    if 'autocorrelograms' in other_metrics.keys():
        df_metrics['acg'] = [other_metrics['autocorrelograms'][0][i] for i in range(len(df_metrics))]
        df_metrics.attrs['acg_bin_ms'] = other_metrics['autocorrelograms'][1]
    df_metrics.attrs.update(other_metrics)
    
    return df_metrics

def session_and_probe_specific_uid(session_ID: str, probe_name: str, uid: int):
    '''
    Build unique cluster identifier string of cluster (UID),
    session and probe specific
    '''
    
    return session_ID + '_' + probe_name + '_' + str(uid)

def np2xrarray(x, cluID, new_dim_prefix:str):
    #Convert a numpy ndarray to xr.DataArray, taking into account the data dimension
    
    data = np.stack(x)
            
    var_new_dims = [f'{new_dim_prefix}_d{i+1}' for i in range(data.ndim-1)]
    extra_coords = {var_new_dims[i]:np.arange(data.shape[i+1]) for i in range(data.ndim-1)} # skip the first UID cooordinates
    extra_coords['cluID'] = cluID
    
    # print(name, k, data.shape, var_new_dims, extra_coords.keys())
    
    da = xr.DataArray(
        data,
        coords=extra_coords,
        dims = ['cluID',*var_new_dims]
    )
    
    
    return da

def flatten_dict(d, prefix):
    return {f'{prefix}_{k}':v for k,v in d.items()}
    

def cellmat2xarray(cell_metrics, cluID_prefix=''):
    df = pd.DataFrame()
    #convert the cell matrics struct from MATLAB to dataframe
    cell_var_names = cell_metrics.keys()
    n_row = cell_metrics['UID'].size
    
    # Reformat the cluID to be unique
    cluID = [f'{cluID_prefix}{id}' for id in cell_metrics['cluID']]
    cell_metrics.pop('UID')
    cell_metrics.pop('cluID')
    
    da_list = {}
    attrs_list = {}
    dims_dict = {}
      
    for name in cell_var_names:
        metrics = cell_metrics[name]
        if type(metrics) is np.ndarray and metrics.shape == (n_row,):
            try:
                da = np2xrarray(metrics, cluID, name)
                da_list[name] = da
            except ValueError:
                #TODO: fix the incompatibility of some object type in the attrs, preventing saving to netCDF file
                # attrs_list[name] = metrics.tolist()
                pass

        elif type(metrics) is dict:
            # More complex nested metrics, in higher dimension (e.g. 1D)
            # expand as new variable
            for k in metrics.keys():   
                if (type(metrics[k]) is np.ndarray and 
                    metrics[k].ndim==2 and 
                    metrics[k].shape[1] == n_row):
                    # 2D data
                    
                    var_new_dim = f'{k}_idx'
                    da = xr.DataArray(
                        metrics[k],
                        coords={var_new_dim:np.arange(metrics[k].shape[0]), 'cluID':cluID},
                        dims = [var_new_dim,'cluID']
                    )
                    
                    da_list[f'{name}_{k}'] = da
                    
                elif (type(metrics[k]) is np.ndarray and 
                      metrics[k].ndim==1 and 
                      metrics[k].shape[0] == n_row):
                    # more complex data, e.g. for waveforms, 3 or more dimensions
                    
                    try:
                        data = np.stack(metrics[k])
                    except ValueError:
                        # variable data format, save in attrs
                        # attrs_list[f'{name}_{k}'] = metrics[k].tolist()
                        continue
                        
                    var_new_dim = f'{k}_idx'
                    
                    var_new_dims = [f'{name}_{k}_d{i+1}' for i in range(data.ndim-1)]
                    extra_coords = {var_new_dims[i]:np.arange(data.shape[i+1]) for i in range(data.ndim-1)} # skip the first UID cooordinates
                    extra_coords['cluID'] = cluID
                    
                    # print(name, k, data.shape, var_new_dims, extra_coords.keys())
                    
                    da = xr.DataArray(
                        data,
                        coords=extra_coords,
                        dims = ['cluID',*var_new_dims]
                    )
                    da_list[f'{name}_{k}'] = da
                            
    dataset = xr.Dataset(da_list)
    dataset.attrs.update(attrs_list)

    if 'general' in cell_metrics.keys():
        # only extract some useful field
        chan_coords = flatten_dict(cell_metrics['general']['chanCoords'], 'chanCoords')
        dataset.attrs.update(chan_coords)  
        
    if 'putativeConnections' in cell_metrics.keys():
        connections = flatten_dict(cell_metrics['putativeConnections'], 'putativeConnections')
        dataset.attrs.update(connections)
        
        
    # do a check to make sure all attribute can be exported
    for k in dataset.attrs.keys():
        assert type(dataset.attrs[k]) is not dict, f'Error, dict type detectec in attribute {k}'
            
    return dataset




def cellmat2dataframe(cell_metrics):
    df = pd.DataFrame()
    #convert the cell matrics struct from MATLAB to dataframe
    cell_var_names = cell_metrics.keys()
    n_row = cell_metrics['UID'].size
    
      
    for name in cell_var_names:
        metrics = cell_metrics[name]
        if type(metrics) is np.ndarray and metrics.shape == (n_row,):
            # Save single value metrics for each cluster
            df[name] = metrics
        elif type(metrics) is dict:
            # More complex nested metrics, in higher dimension (e.g. 1D)
            # expand as new variable
            for k in metrics.keys():   
                # print(name,k)             
                if (type(metrics[k]) is np.ndarray and 
                    metrics[k].ndim==2 and 
                    metrics[k].shape[1] == n_row):
                    #1D data
                    df[f'{name}_{k}'] = metrics[k].T.tolist()
                    
                elif (type(metrics[k]) is np.ndarray and 
                      metrics[k].ndim==1 and 
                      metrics[k].shape[0] == n_row):
                    # more complex data, e.g. for waveforms
                    df[f'{name}_{k}'] = metrics[k]
                    
            
    # also save the generate properties
    if 'general' in cell_metrics.keys():
        df.attrs.update(cell_metrics['general'])  
        
    if 'putativeConnections' in cell_metrics.keys():
        df.attrs['putativeConnections'] = cell_metrics['putativeConnections']
            
    return df
    
    
def plot_firing_rate(xr_fr_coord, xr_session, df_pycontrol, events2plot, xlim=None):
    # xlim should be in milisecond
    # the xr_fr_coord should already be sorted in pos_y
    
    style_plot()
    assert all(np.diff(xr_fr_coord.ks_chan_pos_y)>=0), 'Error! Datset must be first sorted by pos_y'
    bin_duration = xr_fr_coord.attrs['bin_duration']

    
    spike_rates = xr_fr_coord.spikes_zFR_session.data
    
    fig,ax = plt.subplots(3,1,figsize=(20,15),dpi=200, sharex=True)
    
    ax_photo, ax_fr, ax_event = ax
        
    # photometry
    if 'zscored_df_over_f' in xr_session:
        ax_photo.plot(xr_session.zscored_df_over_f.data.ravel())
    
    # firing rate map
    image = ax_fr.imshow(spike_rates.T, vmax=2, vmin=-2,cmap='icefire')
    ax_fr.set_aspect('auto')
    
    yticks = np.arange(0, spike_rates.shape[1],50 ) #where we want to show the
    
    
    ax_fr.set_yticks(yticks)
    ax_fr.set_yticklabels(xr_fr_coord.ks_chan_pos_y.data[yticks]); #the cooresponding label for the tick
    ax_fr.invert_yaxis()
    
    
    xticks = np.linspace(0,spike_rates.shape[0]-10,10).astype(int)
    
    ax_fr.set_xticks(xticks)
    xticklabels = (xr_fr_coord.time[xticks].data/1000).astype(int)
    ax_fr.set_xticklabels(xticklabels)

    
    ax_fr.set_ylabel('Distance from tip (um)')
    ax_fr.set_xlabel('Time (s)')

    # also plot the important pycontrol events
    
    events2plot = df_pycontrol[df_pycontrol.content.isin(events2plot)]

    ## Event
    evt_colours =['r','g','b','w']
    # Note: the time coordinate of the firing map corresponds to the time bins
    for i, event in enumerate(events2plot.content.unique()):
        evt_time = events2plot[events2plot.content==event].time
        evt_time_idx = [np.searchsorted(xr_fr_coord.time, t) for t in evt_time]
        # evt_time = evt_time/bin_duration
        ax_event.eventplot(evt_time_idx, lineoffsets=80+20*i, linelengths=20,label=event, color=evt_colours[i])
    
    ax_event.legend(loc='upper left', prop = { "size": 12 }, ncol=4)

    
    cbar_ax = fig.add_axes([0.95, 0.55, 0.02, 0.35]) 
    fig.colorbar(image, cax=cbar_ax)
    
    if xlim is not None:
        ax_photo.set_xlim(np.array(xlim)/bin_duration)
    
    return fig

def get_max_sig_region_size(pvalues, p_threshold=0.05):
    #return the size of the maximum consecutive region where pvalues<p_threshold
    # pvalues is assign to be of shape (cell x time)
    
    cond = (pvalues<p_threshold).astype(int)
    pad = np.zeros((cond.shape[0],1))
    cond = np.hstack([pad, cond, pad]) #take care of the edge
    
    d = np.diff(cond,axis=1)
    max_region_size = np.zeros((cond.shape[0],))
    for i in range(cond.shape[0]):
        #detect the state change
        start = np.where(d[i,:]==1)[0]
        end = np.where(d[i,:]==-1)[0]
        region_size = end-start
        if len(region_size)>0:
            max_region_size[i]=np.max(region_size) # maximum consecutive region size
            
    return max_region_size


def compare_fr_with_random(da, da_rand, cluID, pvalues=None, random_n=1000, ax=None):
    # xr_fr: the dataArray with the continuuous firing rate of the cell
    
    style_plot()
    df2plot = da.sel(cluID=cluID).to_dataframe()
    df2plot['type'] = 'event-triggered'
    df2plotR = da_rand.sel(cluID=cluID).to_dataframe()
    df2plotR['type'] = 'random'

    df2plot = pd.concat([df2plot, df2plotR]).reset_index()
    ax = sns.lineplot(df2plot, y=da.name, x='spk_event_time', hue='type', n_boot=100, ax=ax)
    ax.legend(loc='upper left', prop = { "size": 8 }, ncol=4)
    ax.set(xlabel='Time around event (ms)')
    
    s = str(cluID.data)
    cluLabel = '_'.join(s.split('_')[1:])
    ax.set_title(cluLabel)

    if pvalues is not None:
        # also indiciate where the difference is significant
        idx = np.where(pvalues<0.05)[0]
        yloc = ax.get_ylim()[0]
        ax.plot(da.spk_event_time[idx], [yloc]*len(idx),'r*')
        
    return ax        
    

def binned_firing_rate(spiketrains, bin_size, t_start=None, t_stop=None,
                   output='counts'):
    # modified from time_histogram from elephant because the original function collapses
    # the spike train of all cells
    
    bs = BinnedSpikeTrain(spiketrains, t_start=t_start, t_stop=t_stop,
                          bin_size=bin_size)
    
    bs_hist = bs.to_array().T
    # Renormalise the histogram
    if output == 'counts':
        # Raw
        bin_hist = pq.Quantity(bs_hist, units=pq.dimensionless, copy=False)
    elif output == 'mean':
        # Divide by number of input spike trains
        bin_hist = pq.Quantity(bs_hist / len(spiketrains),
                               units=pq.dimensionless, copy=False)
    elif output == 'rate':
        # Divide by number of input spike trains and bin width
        bin_hist = bs_hist / (len(spiketrains) * bin_size)
    else:
        raise ValueError(f'Parameter output ({output}) is not valid.')

    return neo.AnalogSignal(signal=bin_hist,
                            sampling_period=bin_size, units=1/pq.s,
                            t_start=bs.t_start, normalization=output,
                            copy=False)
    
def diff_permutation_test(x,y):
    # use permutation test to compare the difference of the mean of two populations
    # permutation test is non-parametric
    
    def statistic(x, y, axis):
        return np.nanmean(x, axis=axis) - np.nanmean(y, axis=axis)

    res = permutation_test((y,x), statistic, n_resamples=100, vectorized=True)
    
    return res.pvalue

def get_pvalue_random_events(da, xr_fr, trial_window, bin_duration,  num_sample=1000):
    # Compare with random event and return the corrected p values
    
    # choose some random event
    timestamps = sorted(np.random.choice(xr_fr.time, size=num_sample, replace=False))
    trial_nb = np.arange(len(timestamps))

    da_rand = build_evt_fr_xarray(xr_fr.spikes_FR_session, timestamps, trial_nb, f'{da.name}', 
                                            trial_window, bin_duration)
    
    
    # Compare the event response with the random events
    pvalue_ratio = np.zeros((len(da.cluID),))
    pvalues = np.zeros((len(da.cluID),len(da.spk_event_time) ))
    
    for i, cluID in enumerate(da.cluID):
        x = da_rand.sel(cluID=cluID).data
        y = da.sel(cluID=cluID).data
                
        # firing rate may not be normally distributed
        # pvalues[i,:] = ttest_ind(x,y,axis=0, nan_policy='omit').pvalue #Note: can be nan in the data if the event cannot be found
       
        # Note: rank sum test although is non-parametric, it is testing the shape of the distribution
        #   rather than the mean, so it results can be confusing when compared with the confidence interval plot
        # pvalues[i,:] = ranksums(x,y,axis=0, nan_policy='omit').pvalue #wilcoxon two samples
        
        # permutation test is non-parametric
        pvalues[i,:] = diff_permutation_test(x,y)
        
        
        # adjust for multiple comparison
        # Disable for now because the test will become too stringent
        # rejected,pvalues[i,:],_,_ = multipletests(pvalues[i,:],0.05)
        # pvalue_ratio[i] = np.mean(rejected)
        
        pvalue_ratio[i] = np.mean(pvalues[i,:]<0.05)
        
    return da_rand, pvalues, pvalue_ratio


# Cross-corr with lags from: https://towardsdatascience.com/computing-cross-correlation-between-geophysical-time-series-488642be7bf0
def crosscorr(datax: pd.Series, datay: pd.Series, lag:int =0):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag))

def crosscorr_lag_range(datax: pd.Series, datay: pd.Series, lags:list):
    cross_corr = np.ndarray(shape=(len(lags)))
    for lag_idx, lag in enumerate(lags):
        cross_corr[lag_idx] = crosscorr(datax,datay,lag)

    return cross_corr


def calculate_pearson_lags(x, y, max_lag, lag_step=1, trial_average=True):
    """
    Calculate Pearson correlation coefficients and lags between two signals.
    
    Improve speed by creating a lag matrix.
    Each row of the lag matrix corresponds to the original signal shifted by some lag.
    The shifted signals from all trials are flattened into one row.
    
    At negative lag, y will be shifted left, positive lag vice versa

    Args:
        x (ndarray): The first signal.
        y (ndarray): The second signal.
        max_lag (int): The maximum lag to consider.
        trial_average (bool): whether to compute the correlation only on trial average

    Returns:
        tuple: A tuple containing:
            - lags (ndarray): The array of lag values.
            - correlations (ndarray): The correlation matrix containing the auto and cross correlations.
            - corr (ndarray): The cross-correlation values.
     
    """
    lags = np.arange(-max_lag, max_lag + 1, lag_step)
    correlations = np.zeros(len(lags))
    
    if trial_average:
        xm = np.zeros((len(lags),x.shape[1]))
        ym = np.zeros_like(xm)
    else:
        xm = np.zeros((len(lags),x.shape[0]*x.shape[1]))
        ym = np.zeros_like(xm)
        
    for i, lag in enumerate(lags):
        if lag < 0:
            shifted_x = x[:,:lag]
            shifted_y = y[:,-lag:]
        elif lag > 0:
            shifted_x = x[:,lag:]
            shifted_y = y[:,:-lag]
        else:
            shifted_x = x
            shifted_y = y
        
        # remove NAN data
        valid_idx = ~np.isnan(shifted_y.mean(axis=1))
        if trial_average:
            shifted_x = np.nanmean(shifted_x[valid_idx,:], axis=0)
            shifted_y = np.nanmean(shifted_y[valid_idx,:], axis=0)
        else:
            shifted_x = shifted_x[valid_idx,:].ravel()
            shifted_y = shifted_y[valid_idx,:].ravel() 
            
        assert len(shifted_x) == len(shifted_y), f'Length mismatch {len(shifted_x)} vs {len(shifted_y)}'
        xm[i, :len(shifted_x)] = shifted_x
        ym[i, :len(shifted_y)] = shifted_y
    
    correlations= np.corrcoef(xm, ym) #contains the auto and then cross correlation between variables
    halfw = len(correlations)//2
    corr = np.diag(correlations[halfw:, :halfw]) # the bottom quandrant is the cross-correlation
    return lags,correlations,corr



def plot_correlated_neurons(cross_corr, xr_spike_session, lags, n_fig = 5):
    UIDs = xr_spike_session.cluID.data
    
    max_corr = cross_corr.max(axis=1)
    max_corr_lag = cross_corr.argmax(axis=1)
    cell_idx_sorted = np.argsort(max_corr)[::-1]
    max_corr_lag_sorted = max_corr_lag[cell_idx_sorted]
    uid_sorted = UIDs[cell_idx_sorted]
    max_corr_sorted = max_corr[cell_idx_sorted]

    start_time = 180
    stop_time = start_time + 30
    start_time_idx = np.searchsorted(xr_spike_session.time, start_time*1000)
    stop_time_idx = np.searchsorted(xr_spike_session.time, stop_time*1000)
    time2plot = xr_spike_session.time[start_time_idx:stop_time_idx]

    style_plot()
    fig, ax = plt.subplots(n_fig,1,figsize=(10,n_fig*2), sharex=True)
    photom = xr_spike_session.analog_1_df_over_f.sel(time=time2plot).data.ravel()
    ax[0].plot(time2plot/1000, photom, label='df/f',color='g')
    ax[0].set_title('dF/F')

    for i, ax in enumerate(ax.flat[1:]):

        shift = int(max_corr_lag_sorted[i]-25)
        time2plot_maxlag = xr_spike_session.time[(start_time_idx+shift):(stop_time_idx+shift)]

        fr = xr_spike_session.spikes_zFR_session.sel(cluID=uid_sorted[i], time=time2plot_maxlag).data

        title = '_'.join(str(uid_sorted[i]).split('_')[1:])
        ax.plot(time2plot/1000,fr,label='Z-scored Firing rate')
        ax.set_title(f'{title}, R2={max_corr_sorted[i]:.2f}, lag={lags[max_corr_lag_sorted[i]]}')
        ax.set_xlabel('Time (s)')
        
    fig.tight_layout()
    
    return fig

def compute_tuning_prop(xr_spikes_trials, xr_fr, trial_window, bin_duration, var2plot):
    # compute the tuninng propertie of each cell
    
    def calculate_tuning_prop(var_name):
        da = xr_spikes_trials[var_name]
        da_rand, pvalues, pvalue_ratio = get_pvalue_random_events(da, xr_fr, trial_window, bin_duration)
        max_region_size = get_max_sig_region_size(pvalues, p_threshold=0.05)
        return {var_name:{
            'pvalues': pvalues.tolist(),
            'pvalue_ratio': pvalue_ratio,
            'max_region_size':max_region_size
        }}

    result = Parallel(n_jobs=10)(delayed(calculate_tuning_prop)(var_name) for var_name in var2plot)

    # combin into one dict
    tuning_dict = result[0]
    for r in result[1:]:
        tuning_dict.update(r)
    
    pvalues_dict = {}
    for var_name, prop_dict in tuning_dict.items():
        for k,v in prop_dict.items():
            pvalues_dict[f'{var_name}:{k}'] = v
    
    # convert into dataframe    
    pvalues_dict['cluID'] = xr_fr.cluID.data
    df_tuning = pd.DataFrame(pvalues_dict)
    
    return df_tuning

def combine2dataframe(result):
    # combine a list of dictionary into dataframe
    tuning_dict = result[0]
    for r in result[1:]:
        tuning_dict.update(r)
    
    pvalues_dict = {}
    for var_name, prop_dict in tuning_dict.items():
        for k,v in prop_dict.items():
            pvalues_dict[f'{var_name}:{k}'] = v
    
    # convert into dataframe    
    df_tuning = pd.DataFrame(pvalues_dict)
    
    return df_tuning

def get_cell_mean_cv(x, coarsen_factor = None, axis=0):
    # Get the mean coefficient of variation for a cell
    cv_list = []
    if coarsen_factor is not None and 'spk_event_time' in x.dims:
        x = x.coarsen(spk_event_time=coarsen_factor,boundary="trim").mean()
    for id in tqdm(x.cluID):
        x_cell = x.sel(cluID=id)
        mean_cv = np.mean(variation(x_cell,axis=axis,nan_policy='omit'))
        cv_list.append(mean_cv)

    return cv_list

def get_cell_mean_fr(x, coarsen_factor = None, axis=0):
    # Get the mean firing rate of th cell
    cv_list = []
    if coarsen_factor is not None and 'spk_event_time' in x.dims:
        x = x.coarsen(spk_event_time=coarsen_factor,boundary="trim").mean()
    for id in tqdm(x.cluID):
        x_cell = x.sel(cluID=id)
        mean_cv = np.mean(x_cell)
        cv_list.append(mean_cv)

    return cv_list

def get_cell_mean_fano(x, coarsen_factor = 5, axis=0):
    # Get the mean fano factor for a cell
    cv_list = []
    x = x.coarsen(spk_event_time=coarsen_factor,boundary="trim").mean()
    for id in tqdm(x.cluID):
        x_cell = x.sel(cluID=id).data
        v = np.nanvar(x_cell, axis=axis)
        m = np.nanmean(x_cell, axis=axis)
        mean_cv = np.mean(v/m)
        cv_list.append(mean_cv)

    return cv_list


def std_score(w,axis):
    return np.max(np.std(w,axis=1),axis=1)


def dtw_path_unilateral(
    s1,
    s2,
    be=None,
):
    #only allow  s2 to expand
    be = instantiate_backend(be, s1, s2)
    s1 = to_time_series(s1, remove_nans=True, be=be)
    s2 = to_time_series(s2, remove_nans=True, be=be)

    if len(s1) == 0 or len(s2) == 0:
        raise ValueError(
            "One of the input time series contains only nans or has zero length."
        )

    if be.shape(s1)[1] != be.shape(s2)[1]:
        raise ValueError("All input time series must have the same feature size.")

    # use a mask to force only s2 to strech
    mask = np.zeros((len(s1), len(s2)))
    indx = np.triu_indices(n=len(s1), m=len(s2),k=1) #the diagnoal must not be inf

    mask[indx] = np.inf
                    
    acc_cost_mat = njit_accumulated_matrix(s1, s2, mask=mask)
    path = dtw_variants._njit_return_path(acc_cost_mat)

    return path, be.sqrt(acc_cost_mat[-1, -1])


def expand_curve(s, factor):
    idx = np.arange(len(s))
    idx = np.repeat(idx, factor)
    return s[idx]

def match2center(s_center, s, k=10):
    '''
    Perform a unilateral match of the curve in s to the center s_center using DTW
    the s_center will be expanded by k times first, then a special mask
    will be used in DTW to constrain only s to expand, but not the reference s_center curve
    Finally, the curves will be downsampled back to the original size
    s should be in the form of time x n
    '''
    s_center_s = expand_curve(s_center, k)
    s_matched = np.zeros_like(s)    
    for i in range(s.shape[1]):
        d1 = s[:,i]
        path, sim = dtw_path_unilateral(s_center_s, d1)
        p1,p2 = list(zip(*path))
    
        #downsample back to the original shape
        s_d = signal.decimate(d1[list(p2)],k)
        s_matched[:,i] = s_d[:s.shape[0]] # discard extra data
        
    return s_matched    

def plot_clusters(data_norm, labels, spk_event_time, var_name, ncol=4, 
                  use_dtw_average=False, align_to_average=False, 
                  alpha=0.7, aspect=1, error_band=None, return_centre=False):
    # align_to_average: whether to align the individual curves to the average by DTW first

    labels_unique = np.unique(labels)
    labels_unique = [x for x in labels_unique if x!=-1] #remove the outliner cluster
    nrow = (len(labels_unique)-1)//ncol+1
    fig = plt.figure(figsize=(ncol*3,nrow*3*aspect), dpi=100)
    colors = plt.cm.tab20.colors
    
    if use_dtw_average:
        # need to precompute the mean curve with joblib because it takes time
        print('I will now compute the barycenters of clusters')
        def compute_mean_curve(curves):
            return softdtw_barycenter(curves, gamma=1, max_iter=20, tol=1e-3)  
        
        mean_curves = Parallel(n_jobs=10,verbose=10)(delayed(compute_mean_curve)(data_norm[labels==lbl,:]) for lbl in labels_unique)
    
    mean_curve_list =[]
    for idx,lbl in enumerate(labels_unique):
        ax = fig.add_subplot(nrow, ncol, idx+1)
        ax.set_title(f'Cluster {lbl} (n={np.sum(labels==lbl)} units)')
        curves = data_norm[labels==lbl,:].T
        x_coord = spk_event_time
        if not use_dtw_average:
            mean_curve = curves.mean(axis=1)
        else:
            mean_curve = mean_curves[idx].ravel()
            mean_curve_list.append(mean_curve)
            # use soft-DTW Barycenter for the average, it is smoother than original Barycenter
        
        if align_to_average:
            # align individual curves with DTW to the average first
            curves = match2center(mean_curve, curves)
        
        if error_band=='ci':
            df2plot = pd.DataFrame({
                'time': np.tile(x_coord, (1,curves.shape[1])).ravel(),
                'fr': curves.T.ravel()
            })
            sns.lineplot(df2plot, x='time', y='fr',n_boot=1000, color=colors[idx%len(colors)])
        elif error_band == 'sem':
            sem = scipy.stats.sem(curves,axis=1)
            ax.fill_between(x_coord, mean_curve+sem, mean_curve-sem,
                            alpha=alpha,
                            color=colors[idx%len(colors)])
            ax.plot(x_coord, mean_curve, color='k');
        elif error_band == 'std':
            std = np.std(curves,axis=1)
            ax.fill_between(x_coord, mean_curve+std, mean_curve-std,
                            alpha=alpha,
                            color=colors[idx%len(colors)])
            ax.plot(x_coord, mean_curve, color='k');


        else:
            ax.plot(x_coord, curves, color=colors[idx%len(colors)],alpha=alpha);
            ax.plot(x_coord, mean_curve, color='k');

            
        
        ax.axvline(x=0,ls='--', color='gray')
    
    
    if return_centre:
        fig.suptitle(var_name)
        fig.tight_layout()
        return (fig, np.stack(mean_curve_list), np.array(labels_unique))
    else:
        return fig

def smooth_response(da_sel, window_length=51):
    #smooth the resopnse curve
    data = da_sel.data.T
    data_smooth = signal.savgol_filter(data,window_length,1)
    data_norm = preprocessing.minmax_scale(data_smooth,axis=1) 
    return data_norm

def cluster_cell(data_norm, min_sample = 10, verbose=False):
    # search for the best eps
    metrics = []
    nlabels_list = []
    eps_list = np.linspace(0.003, 0.02, 10)
    for eps in eps_list:
        clustering = cluster.DBSCAN(min_samples=min_sample, eps=eps, metric='correlation').fit(data_norm)
        nlabels = len(np.unique(clustering.labels_))
        outliner_perc = np.mean(clustering.labels_==-1)
        if verbose:
            print(f'{eps}: {nlabels} clusters, outliner percent: {outliner_perc}')
            
        metrics.append(nlabels/(outliner_perc*5+1))
        nlabels_list.append(nlabels)

    best_idx = np.argmax(metrics)
    best_eps = eps_list[best_idx]
    if verbose:
        print(f'best eps is {best_eps} with {nlabels_list[best_idx]} clusters')
        # plt.plot(metrics)
        
    clustering = cluster.DBSCAN(min_samples=min_sample, eps=best_eps, metric='correlation').fit(data_norm)
        
    labels = clustering.labels_

    return labels

def parse_cluID(s):
    ss = s.split('_')
    session_id = ss[0]
    probe = ss[1]
    id = ss[2]
    animal_id = session_id.split('-')[0]
    date = '-'.join(session_id.split('-')[1:5])

    return {'animal_id':animal_id,
            'probe':probe,
            'id': id,
            'date': date}
    
def make_symmetric(results):
    #
    max_len = max(map(len,results))
    out = np.zeros((max_len,max_len))
    
    for i in range(len(results)):
        out[i,:len(results[i])] = results[i]

    out = np.vstack(out)
    
    #make diagonal matrix
    out = out + out.T-np.diag(out.diagonal())
    return out

def cal_dtw(da_norm, i):
    dist = np.zeros((i+1,))
    x = da_norm.copy()
    for j in range(i):
        dist[j],_= fastdtw(x[i,:], x[j,:])

    return dist

def plot_cell_waveforms(xa_waveforms, cluIDs, ncol=3, metrics=None, figsize_subplot=(3,3)):
    # plot the spike waveform of specified IDs
    # xa_waveforms is assumed to have the dimension ['cluID', 'channels','waveform_time_idx']

    fig, axes = create_plot_grid(len(cluIDs), ncol, dpi=100, figsize_plot=figsize_subplot)

    for i in range(len(cluIDs)):
        ax = axes[i]
        cell2plot = xa_waveforms.sel(cluID=cluIDs[i])
        cell2plot.plot(ax=ax)
        
        ax_insert = inset_axes(ax, width='40%', height='40%', loc='lower left')
        ax_insert.plot(cell2plot.data.T)
        
        if metrics is not None:
            ax.set_title(f'{metrics[i]:.2f}')
        else:
            title = '_'.join(cluIDs[i].split('_')[-2:])
            ax.set_title(title)

    fig.tight_layout()
    
def prominence_score(w,axis):
    # ratio of the peak to peak to the standard deviation
    return np.log(np.max(np.ptp(w,axis=1)/(np.median(w,axis=1)+0.001),axis=1))



def get_random_evt_data(xr_fr, da, trial_window, num_sample=1000):
    timestamps = sorted(np.random.choice(xr_fr.time, size=num_sample, replace=False))
    trial_nb = np.arange(len(timestamps))
    bin_duration = xr_fr.attrs['bin_duration']
    da_rand = build_evt_fr_xarray(xr_fr.spikes_FR_session, timestamps, trial_nb, f'{da.name}', 
                                                trial_window, bin_duration)

    return da_rand

def create_comparison_dataframe(da_rand, da, cluID, dpvar_name, coarsen_factor=5):
    """
    Create a comparison dataframe by combining two input DataArrays and adding group labels.

    Parameters:
    - da_rand (xarray.DataArray): Random DataArray.
    - da (xarray.DataArray): Event-triggered DataArray.
    - cluID (str): Cluster ID.
    - dpvar_name (str): Name of the dependent variable.
    - coarsen_factor (int, optional): Coarsening factor for spk_event_time. Default is 5.

    Returns:
    - data2test (pandas.DataFrame): Comparison dataframe with group labels.

    """
    da1 = da_rand.sel(cluID=cluID).coarsen(spk_event_time=coarsen_factor, boundary='trim').mean()
    df_da1 = da1.to_dataframe().reset_index()
    df_da1['group'] = 'random'
    df_da1 = df_da1.dropna()

    da2 = da.sel(cluID=cluID).coarsen(spk_event_time=coarsen_factor, boundary='trim').mean()
    df_da2 = da2.to_dataframe().reset_index()
    df_da2['group'] = 'event-triggered'
    
    # Do it separately for data without a trial structure
    if 'trial_nb' not in da2.dims:
        # rename the column to make a dummy trial_nb
        df_da2 = df_da2.rename(columns={da2.dims[0]:'trial_nb'})

    df_da2['trial_nb'] += df_da1.trial_nb.max()+1
    data2test = pd.concat([df_da1, df_da2])
    
    data2test = data2test.rename(columns={dpvar_name:'dv'})
    data2test = data2test.dropna()
    return data2test

def do_mix_anova_analysis(data2test):

    # First do mixed anova test
    if data2test['group'].nunique()<2:
        # error, early return
        comparison_result = {
            'cluID': data2test.iloc[0].cluID,
            'group_p': np.NaN,
            'spk_event_time_p': np.NaN,
            'interaction_p': np.NaN,
            'sig_interaction_time': np.NaN,
            'sig_interaction_time_p': np.NaN,
            'interaction_padjust': np.NaN
        }

        return comparison_result

    anova_result = pg.mixed_anova(dv='dv', within='spk_event_time',between='group', 
                   subject='trial_nb', 
                   data=data2test)
    anova_result = anova_result.set_index('Source')
    
    comparison_result = {
        'cluID': data2test.iloc[0].cluID,
        'group_p': anova_result.loc['group','p-unc'],
        'spk_event_time_p': anova_result.loc['spk_event_time','p-unc'],
        'interaction_p': anova_result.loc['Interaction','p-unc']
    }

    # then do postdoc test with multiple comparison correction
    paired_test_result = pg.pairwise_tests(data2test, dv='dv', 
                within='spk_event_time',between='group', 
            subject='trial_nb', padjust='bonf')


    #only focus on the time period where there is siginficant interaction between group and time
    time_sig = paired_test_result[(paired_test_result['p-corr']<0.05) & (paired_test_result.Paired==False)]

    comparison_result.update(
        {
            'sig_interaction_time': time_sig.spk_event_time.values,
            'sig_interaction_time_p': time_sig['p-corr'].values,
            'interaction_padjust': time_sig.iloc[0]['p-adjust'] if len(time_sig)>0 else None
        })

    return comparison_result

def get_chan_coords(xr_spikes_trials):
    # return a dataframe of the coordinations of each unit based on its maxWaveformCh
    waveform_chan = xr_spikes_trials.maxWaveformCh.to_dataframe()
    chanCoords_x = xr_spikes_trials.attrs['chanCoords_x']
    chanCoords_y = xr_spikes_trials.attrs['chanCoords_y']
    waveform_chan['pos_x'] = chanCoords_x[waveform_chan.maxWaveformCh.astype(int)]
    waveform_chan['pos_y'] = chanCoords_y[waveform_chan.maxWaveformCh.astype(int)]
    return waveform_chan.reset_index()


def get_pss(data, axis):
    # calculate the post-spike supression
    #assumption is that post-spike suppression ends after 100ms
    fr_exceed = data[500:1000,:] > np.mean(data[600:900,:],axis=0)
    
    # find out when the firing rate return to mean after suppression
    # i.e. the first True value after the comparison
    return list(map(lambda x: np.where(x)[0][0], fr_exceed.T))

def long_isi_ratio_(spiketimes, total_time):
    isi = np.diff(spiketimes)
    return np.sum(isi[isi>2000])/total_time
    
def cal_long_isi_ratio(sorting_path):
    synced_timestamp_files = list(sorting_path.glob('*/sorter_output/rsync_corrected_spike_times.npy'))
    spike_clusters_files = list(sorting_path.glob('*/sorter_output/spike_clusters.npy'))
    spike_trains, all_clusters_UIDs = get_spike_trains(synced_timestamp_files, spike_clusters_files)

    # calculate the long isi ratio
    long_isi_ratio = np.zeros((len(spike_trains),))
    for i, spktrain in enumerate(spike_trains):
        long_isi_ratio[i] =  long_isi_ratio_(spktrain.times, spktrain.t_stop)

    df = pd.DataFrame({'cluID': all_clusters_UIDs, 'long_isi_ratio': long_isi_ratio})
    df = df.set_index('cluID')
    
    return df

def classify_cell_type(cell):
    '''
    According to Andrew Peters et. al (Nature 2021)
    '''
    if cell['troughToPeak'] <0.4:
            
        if cell['long_isi_ratio']>0.1:
            return 'UIN'
        else:
            return 'FSI'
    else:
        if cell['acg_pss'] > 40: #assume 1 bin is 1ms
            return 'TAN'
        else:
            return 'SPN'
        
def get_leaves(node):
    # return all the leaves from this node
    if not node.is_leaf():
        return get_leaves(node.left) + get_leaves(node.right)
    else:
        return [node.id]

def get_clus_at_level(node, lvl):
    clus_list =[]
    def get_clusters_at_level_(node, lvl):
        #return all the cluster at specified level
        if lvl>0:
            if node.left is not None:
                get_clusters_at_level_(node.left, lvl-1)
            if node.right is not None:
                get_clusters_at_level_(node.right, lvl-1)
        else:
            clus_list.append(get_leaves(node))

    get_clusters_at_level_(node,lvl)
            
    return clus_list

def extract_valid_subarray(x):
    # extract a subarray without NaN valid
    # x is assumed to be symmetric
    valid_idx = ~np.isnan(np.diag(x))
    x = x[~np.isnan(x)]
    n = np.sum(valid_idx)
    x = x.reshape((n,n))
    return x, valid_idx

def select_cluster_cells(clus_centroid, clus_lbls, da_norm, labels, df_clus, 
                         cluster_label, time_tolerance=3, plot=True):
    """
    Select cells from a cluster based on temporal alignment of peak responses.
    
    Parameters
    ----------
    clus_centroid : ndarray
        Array of cluster centroids with shape (n_clusters, n_timepoints)
    clus_lbls : ndarray
        Cluster labels corresponding to centroids
    da_norm : ndarray
        Normalized firing rate curves with shape (n_cells, n_timepoints)
    labels : ndarray
        Cluster assignment for each cell
    df_clus : DataFrame
        DataFrame with 'cluID' and 'label' columns mapping cells to clusters
    cluster_label : int
        The cluster label to analyze
    time_tolerance : int, optional
        Maximum time offset (in bins) from centroid peak, by default 3
    plot : bool, optional
        Whether to plot the results, by default True
    
    Returns
    -------
    clus_sel : ndarray
        Array of selected cell IDs (cluID values)
    """
    # Find the centroid for this cluster
    idx = np.flatnonzero(clus_lbls == cluster_label)
    
    if plot:
        plt.plot(clus_centroid[idx[0], :])
        plt.title(f'Cluster {cluster_label} centroid')
        plt.show()
    
    # Find critical time point (peak) in the centroid
    c = clus_centroid[idx[0], :].argmax()
    
    # Get all cells in this cluster
    clus_idx = np.flatnonzero(labels == cluster_label)
    
    # Find peak time for each cell
    min_pt = np.argmax(da_norm[clus_idx, :], axis=1)
    
    # Select cells where peak is within tolerance of centroid peak
    ingroup = (np.abs(c - min_pt) < time_tolerance)
    
    print(f"Selected {sum(ingroup)} cells out of {len(ingroup)} in cluster {cluster_label}")
    
    if plot:
        plt.plot(da_norm[clus_idx[ingroup], :].T)
        plt.title(f'Selected cells from cluster {cluster_label}')
        plt.xlabel('Time bin')
        plt.ylabel('Normalized firing rate')
        plt.show()
    
    # Get the cluID values for selected cells
    cluID_idx = np.flatnonzero(labels == cluster_label)[ingroup]
    clus_sel = df_clus.iloc[cluID_idx]['cluID'].values
    
    return clus_sel
