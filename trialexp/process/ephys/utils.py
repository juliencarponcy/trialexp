
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd   
import os
from tqdm.auto import tqdm
import xarray as xr
from trialexp.process.ephys.spikes_preprocessing import build_evt_fr_xarray
from elephant.conversion import BinnedSpikeTrain
import quantities as pq
from trialexp.process.group_analysis.plot_utils import style_plot
import seaborn as sns
import neo 
from scipy.stats import ttest_ind, wilcoxon, ranksums
from statsmodels.stats.multitest import multipletests

def denest_string_cell(cell):
        if len(cell) == 0: 
            return 'ND'
        else:
            return str(cell[0])


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



def prepare_mathlab_path(eng):
    # Adding Path to Matlab from Environment variables defined in .env file.
    s = eng.genpath(os.environ['CORTEX_LAB_SPIKES_PATH']) # maybe unnecessary, just open all ks3 results
    n = eng.genpath(os.environ['NPY_MATLAB_PATH'])
    c = eng.genpath(os.environ['CELL_EXPLORER_PATH'])

    eng.addpath(s, nargout=0)
    eng.addpath(n, nargout=0)
    eng.addpath(c, nargout=0)
    
    
def plot_firing_rate(xr_fr_coord, xr_session, df_pycontrol, events2plot, xlim=None):
    # xlim should be in milisecond
    
    style_plot()
    bin_duration = xr_fr_coord.attrs['bin_duration']

    
    spike_rates = xr_fr_coord.spikes_zFR_session.data
    
    fig,ax = plt.subplots(3,1,figsize=(20,15),dpi=200, sharex=True)
    
    ax_photo, ax_fr, ax_event = ax
        
    # photometry
    ax_photo.plot(xr_session.zscored_df_over_f.data.ravel())
    
    # firing rate map
    image = ax_fr.imshow(spike_rates.T, vmax=2, vmin=-2,cmap='icefire')
    ax_fr.set_aspect('auto')
    
    yticks = np.arange(0, spike_rates.shape[1],50 ) #where we want to show the
    
    
    ax_fr.set_yticks(yticks)
    ax_fr.set_yticklabels(xr_fr_coord.pos_y.data[yticks]); #the cooresponding label for the tick
    ax_fr.invert_yaxis()
    
    
    xticks = np.linspace(0,spike_rates.shape[0]-10,10).astype(int)
    
    ax_fr.set_xticks(xticks)
    xticklabels = (xr_fr_coord.time[xticks].data/1000).astype(int)
    ax_fr.set_xticklabels(xticklabels)

    
    ax_fr.set_ylabel('Distance from tip (um)')
    ax_fr.set_xlabel('Time (s)')

    # also plot the important pycontrol events
    
    events2plot = df_pycontrol[df_pycontrol.name.isin(events2plot)]

    ## Event
    evt_colours =['r','g','b','w']
    # Note: the time coordinate of the firing map corresponds to the time bins
    for i, event in enumerate(events2plot.name.unique()):
        evt_time = events2plot[events2plot.name==event].time
        evt_time_idx = [np.searchsorted(xr_fr_coord.time, t) for t in evt_time]
        # evt_time = evt_time/bin_duration
        ax_event.eventplot(evt_time_idx, lineoffsets=80+20*i, linelengths=20,label=event, color=evt_colours[i])
    
    ax_event.legend(loc='upper left', prop = { "size": 12 }, ncol=4)

    
    cbar_ax = fig.add_axes([0.95, 0.55, 0.02, 0.35]) 
    fig.colorbar(image, cax=cbar_ax)
    
    if xlim is not None:
        ax_photo.set_xlim(np.array(xlim)/bin_duration)
    
    return fig



def compare_fr_with_random(da, da_rand, cluID, pvalues=None, random_n=1000, ax=None):
    # xr_fr: the dataArray with the continuuous firing rate of the cell
    # timestamps = sorted(np.random.choice(xr_fr.time, size=random_n, replace=False))
    # trial_nb = np.arange(len(timestamps))
    # da_rand = build_evt_fr_xarray(xr_fr, timestamps, trial_nb, f'{da.name}', 
    #                                         trial_window, bin_duration)

    df2plot = da.sel(cluID=cluID).to_dataframe()
    df2plot['type'] = 'event-triggered'
    df2plotR = da_rand.sel(cluID=cluID).to_dataframe()
    df2plotR['type'] = 'random'

    df2plot = pd.concat([df2plot, df2plotR]).reset_index()
    ax = sns.lineplot(df2plot, y=da.name, x='spk_event_time', hue='type', n_boot=100, ax=ax)
    ax.legend(loc='upper left', prop = { "size": 8 }, ncol=4)
    ax.set(xlabel='Time around event (ms)')

    if pvalues is not None:
        # also indiciate where the difference is significant
        idx = np.where(pvalues<0.05)[0]
        yloc = ax.get_ylim()[0]
        ax.plot(da.spk_event_time[idx], [yloc]*len(idx),'r*')
        
    

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
    
    for i, cluID in tqdm(enumerate(da.cluID), total=len(da.cluID)):
        x = da_rand.sel(cluID=cluID).data
        y = da.sel(cluID=cluID).data
        
        # firing rate may not be normally distributed
        # pvalues[i,:] = ttest_ind(x,y,axis=0, nan_policy='omit').pvalue #Note: can be nan in the data if the event cannot be found
        pvalues[i,:] = ranksums(x,y,axis=0, nan_policy='omit').pvalue #wilcoxon two samples
        
        # adjust for multiple comparison
        rejected,pvalues[i,:],_,_ = multipletests(pvalues[i,:],0.05)
        pvalue_ratio[i] = np.mean(rejected)
        
    return da_rand, pvalues, pvalue_ratio