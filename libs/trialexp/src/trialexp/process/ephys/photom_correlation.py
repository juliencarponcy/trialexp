import numpy as np 
import matplotlib.pylab as plt
import pandas as pd
from trialexp.process.ephys.utils import calculate_pearson_lags
import xarray as xr
import seaborn as sns
import trialexp.process.pyphotometry.linear_modelling as lm


def plot_extrem_corr(xr_corr, xr_spike_fr_interp, xr_session, evt_name, sig_name, mode='abs'):
    '''
    Plot the average photometry signal together with the average firing rate for the largest correlations.

    Parameters:
    - xr_corr (xarray.DataArray): Xarray data array containing the correlation coefficients.
    - xr_spike_fr_interp (xarray.Dataset): Xarray dataset containing the interpolated spike firing rate data.
    - xr_session (xarray.Dataset): Xarray dataset containing the session data.
    - evt_name (str): Name of the event.
    - sig_name (str): Name of the signal.
    - mode (str, optional): How to sort the coefficient. Can be 'desc', 'asc', or 'abs'. Defaults to 'abs'.
        for 'abs', the absolute value will be sorted descendingly

    Returns:
    - None

    '''
    
    var_name = evt_name+sig_name

    # find the largest correlation
    photom_data = xr_session[var_name].values
    fr = xr_spike_fr_interp[f'spikes_FR.{evt_name}'].data
    c = xr_corr[var_name].data
    if mode == 'abs':
        idx = np.argmax(np.abs(c).data,axis=1) # the lag with the large correlation
        extrema_corr = c[np.arange(c.shape[0]),idx] #advance indexing do not work on xarray directly
        sorted_idx = np.argsort(np.abs(extrema_corr))[::-1]
    elif mode == 'asc':
        idx = np.argmin(c.data,axis=1)
        extrema_corr = c[np.arange(c.shape[0]),idx] 
        sorted_idx = np.argsort(extrema_corr)
    else:
        idx = np.argmax(c.data,axis=1)
        extrema_corr = c[np.arange(c.shape[0]),idx] 
        sorted_idx = np.argsort(extrema_corr)[::-1]
        
    lag = xr_corr.lag.data
    max_corr_loc = lag[idx]
    
    # plot average photometry signal together with average firing rate
    fig, axes = plt.subplots(3,3,figsize=(3*4,3*3))
    label1 = None
    label2 = None
    for i,ax in enumerate(axes.flat):
        ax2 = ax.twinx()
    
        if i == len(axes.flat)-1:
            label1='unit firing'
            label2='photmetry'
        
        x = fr[:,:,sorted_idx[i]].mean(axis=0)
        y = np.nanmean(photom_data,axis=0)
        shift = int(max_corr_loc[sorted_idx[i]]/(1000/50))
        ax.plot(xr_spike_fr_interp.event_time, x, label=label1)
        ax.set_ylabel('Firing rate (Hz)')
        ax2.plot(xr_spike_fr_interp.event_time, y,'r', label=label2)
        ax2.set_ylabel('dF/F')
        ax.set_title(f'Largest corr = {extrema_corr[sorted_idx[i]]:.2f} at {max_corr_loc[sorted_idx[i]]:.1f}ms\n {xr_corr.cluID.data[sorted_idx[i]]}')
    
    fig.tight_layout()
    fig.legend()
    
    return fig


def get_corr_spatial_distribution(xr_corr, df_metrics, signal_name):
    corr_dict={}
    for varname in xr_corr.data_vars:
        c = xr_corr[varname].data
        idx = np.argmax(np.abs(c).data,axis=1)
        extrema_corr = c[np.arange(c.shape[0]),idx] #advance indexing do not work on xarray directly
        corr_dict[varname] = extrema_corr
    
    df_corr = pd.DataFrame(corr_dict)
    df_corr['cluID'] = xr_corr.cluID;
    
    idvars = [c for c in df_corr.columns if c.endswith(signal_name)]
    
    df_corr = df_corr.merge(df_metrics,on='cluID')
    
    df_meancorr = df_corr[idvars+['ks_chan_pos_y']]
    df_meancorr = df_meancorr.groupby('ks_chan_pos_y').mean()
    
    # simplfy the column for plotting
    df_meancorr.columns = [c.replace(signal_name,'') for c in df_meancorr.columns]
    return df_meancorr


def analyze_correlation(xr_spike_fr_interp, xr_session, evt_name, sig_name, evt_time_step, cluID, average_trial, trial_outcome=None):
    
    fr_data= xr_spike_fr_interp[f'spikes_FR.{evt_name}']
    xr_session  = xr_session.isel(session_id=0) # reduce the dimension for easier processing
    photom_data=xr_session[f'{evt_name}{sig_name}']
    
    if trial_outcome is not None and 'trial_nb' in photom_data.coords:
        photom_data = photom_data.sel(trial_nb = (xr_session.trial_outcome==trial_outcome)).data
        fr_data = fr_data.sel(trial_nb = (xr_spike_fr_interp.trial_outcome == trial_outcome)).data
    else:
        # Calculate the correlation of each units with the photometry signal
        # for all trials
        photom_data = photom_data
        fr_data = fr_data.data
        trial_outcome = 'success'
    
    print(f'Processing {evt_name+sig_name} for {trial_outcome}')
    max_lags = 5 # corresponds to arround 100ms at 50Hz, larger shift may risk mixing with other events
    lag_step = 1
    nlags = max_lags//lag_step
    
    corr = np.zeros((fr_data.shape[2],nlags*2+1))
    for i in range(fr_data.shape[2]):
        # Negative lag means the photometry signal will be shifted left
        lags,_, corr[i,:] = calculate_pearson_lags(fr_data[:,:,i], photom_data,max_lags, lag_step, average_trial)
    
    xr_data = xr.DataArray(corr,
                           name = evt_name+sig_name,
                           dims=['cluID', 'lag'],
                           coords={'cluID':cluID, 'lag':lags*evt_time_step})
    
    xr_data = xr_data.expand_dims({'trial_outcome':[trial_outcome]})
    return xr_data


def plot_extrem_corr_timewarp(extraction_specs, trigger, xr_corr, xr_spike_timewarp, xr_photom_timewarp, 
                              photom_signal_name,  spike_signal_name,
                                event_name,trial_outcome='success', mode='abs'):
    '''
    Plot the average photometry signal together with the average firing rate for the largest correlations.

    Parameters:
    - xr_corr (xarray.DataArray): Xarray data array containing the correlation coefficients.
    - xr_spike_fr_interp (xarray.Dataset): Xarray dataset containing the interpolated spike firing rate data.
    - xr_session (xarray.Dataset): Xarray dataset containing the session data.
    - evt_name (str): Name of the event.
    - sig_name (str): Name of the signal.
    - mode (str, optional): How to sort the coefficient. Can be 'desc', 'asc', or 'abs'. Defaults to 'abs'.
        for 'abs', the absolute value will be sorted descendingly

    Returns:
    - None

    '''

    # find the largest correlation
    photom_data = xr_photom_timewarp[photom_signal_name]
    fr = xr_spike_timewarp[spike_signal_name]

    # only select a specific trial outcome
    sel_idx = (xr_photom_timewarp.trial_outcome.data == trial_outcome)
    photom_data = photom_data.sel(trial_nb=sel_idx)
    fr = fr.sel(trial_nb=sel_idx)

    c = xr_corr[event_name+'_'+photom_signal_name].sel(trial_outcome=trial_outcome).data # event x 

    if mode == 'abs':
        idx = np.argmax(np.abs(c).data,axis=1) # the lag with the large correlation
        extrema_corr = c[np.arange(c.shape[0]),idx] #advance indexing do not work on xarray directly
        sorted_idx = np.argsort(np.abs(extrema_corr))[::-1]
    elif mode == 'asc':
        idx = np.argmin(c.data,axis=1)
        extrema_corr = c[np.arange(c.shape[0]),idx] 
        sorted_idx = np.argsort(extrema_corr)
    else:
        idx = np.argmax(c.data,axis=1)
        extrema_corr = c[np.arange(c.shape[0]),idx] 
        sorted_idx = np.argsort(extrema_corr)[::-1]
        
    lag = xr_corr.lag.data
    max_corr_loc = lag[idx]

    # plot average photometry signal together with time warped firing rate
    fig, axes = plt.subplots(5,1,figsize=(8,5*3))
    label1 = None
    label2 = None
    for i,ax in enumerate(axes.flat):
        ax2 = ax.twinx()
    
        if i == len(axes.flat)-1:
            label1='unit firing'
            label2='photmetry'
        
        df_fr = fr[:,:,sorted_idx[i]].to_dataframe()
        df_photom = photom_data.to_dataframe()

        sns.lineplot(data=df_fr, x='time', y=spike_signal_name, ax=ax, label=label1, n_boot=10)
        sns.lineplot(data=df_photom, x='time', y=photom_signal_name, ax=ax2, color='r', label=label2,n_boot=10)

        # add in the warp information
        lm.add_warp_info(ax, extraction_specs, trigger)
        # sns.move_legend(ax, 'upper left', bbox_to_anchor=[1,1], title=None, frameon=False)
        ticks, ticks_labels = lm.compute_ticks(extraction_specs)
        ax.set_xticks(ticks, labels =ticks_labels, rotation=30) # duplicated tick will be overrided
        ax.set_title(f'{extrema_corr[sorted_idx[i]]:.2f} at {max_corr_loc[sorted_idx[i]]:.2f} \n {fr.cluID.data[sorted_idx[i]]}')
        ax.spines['top'].set_visible(False)

    fig.tight_layout()
    fig.legend()
    
    return fig