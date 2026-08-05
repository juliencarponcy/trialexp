import xarray as xr
import numpy as np
from sklearn import linear_model
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from collections import Counter
from matplotlib.patches import Polygon
from matplotlib.path import Path
import matplotlib.patches as patches
from sklearn import preprocessing
import json
from loguru import logger

def extract_event(df_events, event, order, dependent_event=None, alternative=None):
    # extract the required event according to order, which can be one of 'first','last','last_before_first'
    # if order is 'last_before', you need to specify the depedent_event as well, it will always be
    # result is a pandas series
    # optionally, you can match multiple events using the alterantive argument

    
    if alternative is None:
        events = df_events[(df_events.content==event) & (df_events.trial_time>0)]
    else:
        events = df_events[(df_events.content.isin([event, *alternative])) & (df_events.trial_time>0)]

    
    if len(events) == 0:
        return None

    if order =='first':
        return events.iloc[0]
    elif order == 'last':
        return events.iloc[-1]
    elif order == 'last_before_first':
        assert dependent_event is not None, 'You must supply the dependent_event'
        if (dep_event := extract_event(df_events, dependent_event, 'first')) is not None:
            df_filter = df_events[df_events.time<=dep_event.time]
            if len(events := df_filter[df_filter.content == event])>0:
                return events.iloc[-1]
        return None
    elif order == 'first_after_last':
        assert dependent_event is not None, 'You must supply the dependent_event'
        if (dep_event := extract_event(df_events, dependent_event, 'last')) is not None:
            df_filter = df_events[df_events.time>=dep_event.time]
            if len(events := df_filter[df_filter.name == event])>0:
                return events.iloc[0]
        
        return None
    else:
        raise NotImplementedError('The specified order is not supported')



def interp_data(trial_data, df_trial, trigger, extraction_specs, sampling_rate):
    # extract siganl around event by event_window in ms
    # all the time between events wall be warpped according to padding in ms
    # the trigger event must be present in df_trial

    event_specs = dict(filter(lambda k: k[0]!=trigger, extraction_specs.items()))
    trigger_specs = extraction_specs[trigger]

    # Construct the interpolation time stamp, need to be consistent with how the window length is calculated later
    event_window_len = np.sum([int((v['event_window'][1]-v['event_window'][0])/1000*sampling_rate) for k,v in extraction_specs.items()])
    total_padding_len = np.sum([int(v['padding']/1000*sampling_rate) for k,v in extraction_specs.items()])
    total_len = total_padding_len+event_window_len
    # print(total_padding_len, event_window_len)

    if len(trial_data) < total_len:
        raise ValueError(f'There is not enough data for interpolation required: {total_len}, available: {len(trial_data)}')
        
    t = np.zeros((total_len),)
    cur_idx = 0

    # find the trigger
    t_trigger_event = df_trial[df_trial.content == trigger]
    if len(t_trigger_event) == 0:
        raise ValueError(f"Error: the trigger {trigger} is not found")

    #copy the signal around the trigger
    t_trigger = t_trigger_event.iloc[0].time
    event_window_len = int((trigger_specs['event_window'][1] - trigger_specs['event_window'][0])/1000*sampling_rate)
    t[cur_idx:(cur_idx+event_window_len)] = np.linspace(t_trigger+trigger_specs['event_window'][0], 
                                                      t_trigger+trigger_specs['event_window'][1], event_window_len, endpoint=False)
    cur_idx += event_window_len
    cur_time = t_trigger+trigger_specs['event_window'][1] #the trial time corresponding to cur_idx
    padding = trigger_specs['padding']
    padding_len = int(padding/1000*sampling_rate)
    
    interp_result = {'interp_'+trigger:True} #the trigger is always found
    last_event = trigger
    last_event_time = t_trigger
    # print(f'cur_idx: {cur_idx}, padding_len: {padding_len}, event_window_len: {event_window_len} total:{total_len}')

    # process the other events one by one
    # TODO  Figure out what to do if one event is missing but not the next
    for evt, specs in event_specs.items():
        dependent_event = specs.get('dependent_event', None)
        # if we can find the event, then warp from the event, if not, just start after padding
        if (event := extract_event(df_trial, evt, specs['order'], dependent_event, specs.get('alternative',None) )) is not None:
            t_event = event.time
            interp_result['interp_'+evt] = True
        else:
            t_event = cur_time+padding-specs['event_window'][0]
            interp_result['interp_'+evt] = False

            
        # find a way to warp between two events
        # Note: note there will be nan when the animal touch the spout too close to the start of next trial
        # e.g. in aborted trial
        
        '''
        Warp the inter-event period
        Raise error if the padding is too long
        cur_time is pointing at the timestamp of last event + post-event window
        Note: there will be error when animal put its right paw on the holding bar. This will somtimes result
        in the spout happening before bar_off, thus searching for the last bar_off before the first spout will fail
        '''

        # TODO: handle this gracefully
        if (cur_time) > (t_event+specs['event_window'][0]):
            raise ValueError(
                f'Not enough time to warp to next event.\n'
                f'  Previous event: {last_event} at {last_event_time:.1f} ms\n'
                f'  Current event: {evt} at {t_event:.1f} ms\n'
                f'  Time between events: {t_event-last_event_time:.1f} ms\n'
                f'  Available time for warping: {t_event-cur_time:.1f} ms\n'
                f'  Required pre-event window for {evt}: {-specs["event_window"][0]:.1f} ms\n'
                f'  Trial outcome: {df_trial.iloc[0].trial_outcome}'
            )

        # warp the signal in the padding_len region
        t[cur_idx:(cur_idx+padding_len)] = np.linspace(cur_time, t_event+specs['event_window'][0], padding_len)
        cur_idx += padding_len
        # cur_time = cur_time + padding # the original idea is only to compress the signal, never expand it
        cur_time = t_event+specs['event_window'][0]+1

        # copy the data around event
        event_window_time = specs['event_window'][1] - specs['event_window'][0]
        event_window_len = int(event_window_time/1000*sampling_rate)
        t[cur_idx:(cur_idx+event_window_len)] = np.linspace(t_event+specs['event_window'][0], t_event+specs['event_window'][1], event_window_len)

        cur_idx += event_window_len
        cur_time = cur_time + event_window_time
        padding = specs['padding']
        padding_len = int(specs['padding']/1000*sampling_rate)

        last_event = evt
        last_event_time = t_event
        # Here both cur_idx and cur_time are right post event window but pre-padding
        # print(f'cur_idx: {cur_idx}, padding_len: {padding_len}, event_window_len: {event_window_len} total:{total_len}')

    # Check if the interpolation time points extend beyond available data
    # This happens when trials are too close together or at the end of recording
    trial_data_start = trial_data.time.min().values
    trial_data_end = trial_data.time.max().values
    t_min = t.min()
    t_max = t.max()

    if t_min < trial_data_start or t_max > trial_data_end:
        raise ValueError(
            f'Interpolation time range [{t_min:.1f}, {t_max:.1f}] ms extends beyond '
            f'available trial data range [{trial_data_start:.1f}, {trial_data_end:.1f}] ms. '
            f'This usually means the next trial starts too soon or recording ended.'
        )

    # use linear interpolation to warp them
    data_interp  = trial_data.interp(time=t)

    assert cur_idx == total_len, 'time array not totally filled'
    data_interp['time'] = np.arange(total_len)/sampling_rate*1000 + trigger_specs['event_window'][0]

    return data_interp, interp_result

def extract_data(dataArray, start_time, end_time):
    # extract data specified by a start and ending time in ms
    ref_time = dataArray.time
    start_idx = np.searchsorted(ref_time, start_time)
    end_idx = np.searchsorted(ref_time, end_time)
    return dataArray[np.arange(start_idx, end_idx)]

def time_warp_data(df_events_cond, xr_signal, extraction_specs, trigger, Fs, verbose=False):
    """
    Time warps the data between events so that they can be aligned together.

    Parameters:
    - df_events_cond (pandas DataFrame): DataFrame containing event conditions.
    - xr_signal (xarray DataArray): DataArray containing the signal data.
    - extraction_specs (dict): Dictionary containing extraction specifications.
    - Fs (int): Sampling frequency.
    - verbose (bool): Whether to print detailed information about skipped trials.

    Returns:
    - xa (xarray DataArray): DataArray containing the time-warped data.
    - interp_results_list (list): List of interpolation results for each trial.

    Notes:
    - If there is insufficient data (e.g., trials too close together), the entire trial
      will be filled with NaN to avoid partial NaN values which complicate analysis.
    - The detection of insufficient data is handled by interp_data(), which raises
      ValueError when the interpolation range extends beyond available data.
    """
    data_list = []
    interp_results_list = []

    # Calculate expected warped data length (for creating NaN trials if needed)
    event_window_len = sum([int((v['event_window'][1]-v['event_window'][0])/1000*Fs)
                            for k, v in extraction_specs.items()])
    total_padding_len = sum([int(v['padding']/1000*Fs)
                             for k, v in extraction_specs.items()])
    total_len = total_padding_len + event_window_len

    for i in df_events_cond.trial_nb.unique():
        df_trial = df_events_cond[df_events_cond.trial_nb==i]

    
        try:
            pre_time = extraction_specs[trigger]['event_window'][0]-500
            post_time = list(extraction_specs.values())[-1]['event_window'][1]

            # Extract photometry data around trial

            trial_data = extract_data(xr_signal, df_trial.iloc[0].time+pre_time, df_trial.iloc[-1].time+post_time+100) # add a slight padding
            # Try to time warp it
            data_p, interp_results = interp_data(trial_data, df_trial, trigger, extraction_specs, Fs)
            interp_results_list.append(interp_results)
            data_p = data_p.expand_dims({'trial_nb':[i]})
            data_list.append(data_p)
        except KeyError as e:
            logger.warning(f'Cannot find event {e}. Skipping')
        except ValueError as e:
            # Create NaN trial when interpolation fails
            error_msg = str(e)

            # Print detailed reason
            if verbose:
                print(f'\nTrial {i}: Setting entire trial to NaN.')
                print(f'  Reason: {error_msg}')


    if len(data_list)>0:
        xa = xr.concat(data_list,dim='trial_nb')
    else:
        xa = xr.DataArray(
            data=np.array([]).reshape(0, 0),
            dims=['trial_nb', 'time'],
            coords={'trial_nb': [], 'time': []},
            attrs=xr_signal.attrs,
            name = xr_signal.name
        )

    return xa,interp_results_list

def add_break_mark(ax, x, y, d, w, h):
    # add break mark to the x axis
    xy = np.array([[x - w/2+d, y+h/2],
                   [x + w/2+d, y+h/2],
                   [x + w/2-d, y-h/2],
                   [x - w/2-d, y-h/2]])

    

    ax.add_patch(Polygon(xy, color='white', closed=True, zorder=3, clip_on=False))
    ax.plot([x - w/2+d, x - w/2-d], [y+h/2,  y-h/2], clip_on=False, color='k',ls='-', zorder=3)
    ax.plot([x + w/2+d, x + w/2-d], [y+h/2,  y-h/2], clip_on=False, color='k',ls='-', zorder=3)
    
def zigzag_path(x0,y0,segment_length=1, height=1):
    
    # Generate vertices and codes for Path
    vertices = [(x0, y0)]
    codes = [Path.MOVETO]
    
    points = [0,1,0,-1,0]
    
    for i in range(1,len(points)):
        x = i * segment_length + x0
        y = points[i]*height +y0
        vertices.append((x, y))
        codes.append(Path.LINETO)
    
    # Create Path object
    zigzag_path = Path(vertices, codes)
    # Create a PathPatch object
    patch = patches.PathPatch(zigzag_path, facecolor='none', edgecolor='k', lw=1,zorder=3, clip_on=False)
    
    return patch
    
def add_compressed_mark(ax,x, y, h, w):

    # Mask the axis
    mw = w*3.5 #half total width
    mh = h*2
    xy = np.array([[x - mw/2, y+h/2],
               [x + mw/2, y+h/2],
               [x + mw/2, y-h/2],
               [x - mw/2, y-h/2]])
    
    ax.add_patch(Polygon(xy, color='white', closed=True, zorder=3, clip_on=False))
    # draw the zigzag mark
    zigzag = zigzag_path(x-w*2,y, segment_length=w, height=h)
    ax.add_patch(zigzag)


def add_warp_info(ax, extraction_specs,trigger, adjust_ylim=True, draw_protected_region=True):
    # draw the event line and shaded region for warping info
    # add a bit of padding for text later
    ylim = ax.get_ylim()
    yrange = ylim[1] -ylim[0]
    if adjust_ylim:
        ax.set_ylim(ylim[0], ylim[1]+yrange*0.2)
    
    # plot the time point in the extraction_specs

    trigger_window = extraction_specs[trigger]['event_window']
    cur_time = trigger_window[0]
    colors = (c for c in plt.cm.tab10.colors)
    
    for i, (evt, specs) in enumerate(extraction_specs.items()):
        pre_time, post_time = specs['event_window']
        padding = specs['padding']
        
        color = next(colors)
        
        ax.axvline(cur_time-pre_time,color= color, ls='--')
        if draw_protected_region:
            ax.axvspan(cur_time, cur_time+(post_time-pre_time), alpha=0.1,color=color)
        label = specs.get('label', evt.replace('_', ' '))
        ax.text(cur_time-pre_time-10, ax.get_ylim()[1], label, rotation = 90, ha='right', va='top')        
        cur_time += (post_time-pre_time)+padding

        ylim = ax.get_ylim()
        marker_size = (ylim[1]-ylim[0])*0.05
        if i != len(extraction_specs.keys())-1 and padding>0:
            # only draw compressed mark when time warping is done
            add_compressed_mark(ax, cur_time-padding/2, ax.get_ylim()[0],ax.get_ylim()[1]/30,10)


    return ax

def compute_ticks(extraction_specs):
    # Calculate the tick location and labels from the specs
    # if the last region has no padding, continue counting

    ticks = []
    ticks_labels = []
    cur_time = None
    continue_count = False

    for k,v in extraction_specs.items():
        if k=='trial_end':
            # don't draw anything for the end of trial
            continue
        if cur_time is None:
            cur_time = v['event_window'][0]
        win_len = v['event_window'][1] - v['event_window'][0]
        
        t = [cur_time, cur_time + win_len]
        ticks += t
        
        tlab = v['event_window'][0], v['event_window'][1]
        if not continue_count:
            ticks_labels += tlab
        else:
            # continue counting from the last region
            ticks_labels += tlab[0]+last_tlab[1], tlab[1]+last_tlab[1]
    
        cur_time =  cur_time + win_len+ v['padding']
        continue_count = (v['padding'] == 0) # continue contining if no warping needs to be performed
        last_tlab = tlab
    
    # set the label to empty if it is zero, for better visualiztion
    ticks_labels = [f'{lbl}' if lbl!=0 else '' for lbl in ticks_labels]
    return ticks, ticks_labels


def plot_warpped_data(xa_cond, signal_var, extraction_specs,trigger, ylim=None,min_ylim=0.1,
                       ylabel=None,ax=None, hue='trial_outcome', palette_colors=None):
    
    if palette_colors is None:
        palette_colors = plt.cm.tab10.colors

    df = xa_cond[[signal_var,hue]].to_dataframe()
    
    # work with multiindex from multisession dataset
    if 'trial_id' in xa_cond.coords:
        df = df.droplevel([1,2])
        df['trial_id'] = df['session_id'].astype(str) + '_' + df['trial_nb'].astype(str)
        df = df.reset_index()    

    else:
        df = df.reset_index()
        df['trial_id'] = df['trial_nb']
    
    df = df.dropna()
    
    if len(df)>0:
        # sometime when the event time doesn't matter the extraction_specs
        # no trial can be extracted
        
         #add in the trial number information
        df_outcome = df.groupby('trial_id').first().dropna()
        df_outcome_count = df_outcome.groupby(hue).count().time
        labels = {k:f'{k} ({df_outcome_count.loc[k]})' for k in df_outcome_count.index}
        df[hue] = df[hue].replace(labels)
        
        outcomes = sorted(df[hue].unique())[::-1]
        
        palette = {k:palette_colors[i] for i,k in enumerate(outcomes)}
        
        sns.lineplot(df, x='time',y=signal_var, 
                    hue=hue, palette=palette, ax = ax, n_boot=100)
        
        # avoid plotting error when signal is close to zero
        # otherwise the zigzag mark will be very far away from the original axis
        yrange = ax.get_ylim()[1] - ax.get_ylim()[0]

        if yrange<min_ylim:
            ax.set_ylim([-min_ylim, min_ylim]) #prevent the axis range become too low

        if ylim is not None:
            ax.set_ylim(ylim)
       
        
        if ylabel is None:
            ylabel = 'z-scored dF/F'
        sns.move_legend(ax, "upper right", bbox_to_anchor=(1.25,1),title=None, frameon=False)
        ax.set(xlabel='Time around events (ms)', ylabel = ylabel)
        
        # add in the warp information
        
        add_warp_info(ax, extraction_specs, trigger)
        sns.move_legend(ax, 'upper left', bbox_to_anchor=[1,1], title=None, frameon=False)
        ticks, ticks_labels = compute_ticks(extraction_specs)
        ax.set_xticks(ticks, labels =ticks_labels, rotation=30) # duplicated tick will be overrided

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


        
def flatten_level(df):
    if 'trial_id' in df.index.names or 'trial_nb' in df.index.names:
        df = df.droplevel([1,2])
        df['trial_id'] = df['session_id'].astype(str) + '_' + df['trial_nb'].astype(str)
        df = df.reset_index()   
    else:
        df = df.reset_index()
        df['trial_id'] = df['trial_nb']

    return df

def plot_timewarp_dataframe(df, signal_var, extraction_specs,trigger, 
                      ax=None, hue='trial_outcome', palette_colors=None):
    # plot time wrapped dataframe, the dataframe must be already flatten by the flatten_evel function
    
    if palette_colors is None:
        palette_colors = plt.cm.tab10.colors

    df = df.dropna()
    
    if len(df)>0:
        # sometime when the event time doesn't matter the extraction_specs
        # no trial can be extracted
        
         #add in the trial number information
         
        df_outcome = df.groupby('trial_id').first().dropna()
        df_outcome_count = df_outcome.groupby(hue).count().time
        labels = {k:f'{k} ({df_outcome_count.loc[k]})' for k in df_outcome_count.index}
        df.loc[:,hue] = df[hue].replace(labels)
        
        outcomes = sorted(df[hue].unique())[::-1]
        
        palette = {k:palette_colors[i] for i,k in enumerate(outcomes)}
        

        sns.lineplot(df, x='time',y=signal_var, 
                    hue=hue, palette=palette, ax = ax, n_boot=100)
        
        sns.move_legend(ax, "upper right", bbox_to_anchor=(1.25,1),title=None, frameon=False)
        ax.set(xlabel='Time around events (ms)', ylabel = 'z-scored dF/F', xlim=[-500,2300])
        
        # add in the warp information
        
        add_warp_info(ax, extraction_specs, 'hold_for_water')
        sns.move_legend(ax, 'upper left', bbox_to_anchor=[1,1], title=None, frameon=False)
        ticks, ticks_labels = compute_ticks(extraction_specs)
        ax.set_xticks(ticks, labels =ticks_labels, rotation=30);

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    


def draw_event_line(extraction_specs,trigger, ax=None, show_label=True, draw_protected_region=False):
    
    palette_colors = plt.cm.tab10.colors


    trigger_window = extraction_specs[trigger]['event_window']
    cur_time = trigger_window[0]
    colors = (c for c in palette_colors)
    
    for evt, specs in extraction_specs.items():
        pre_time, post_time = specs['event_window']
        padding = specs['padding']
        
        color = next(colors)
        
        ax.axvline(cur_time-pre_time,color= color, ls='--')
        if draw_protected_region:
            ax.axvspan(cur_time, cur_time+(post_time-pre_time), alpha=0.1,color=color)
        
        if show_label:
            label = specs.get('label', evt.replace('_', ' '))
            ax.text(cur_time-pre_time-10, ax.get_ylim()[1], label, rotation = 90, ha='right', va='top')
        cur_time += (post_time-pre_time)+padding
        

def perform_linear_regression(xa_cond, data, formula, data2=None, scale_data=True, **predictor_vars):
    """
    Perform linear regression on the given data.

    Args:
        xa_cond (Xarray): Xarray object containing time data.
        data (ndarray): 2D array of shape (n_samples, n_timepoints) containing the dependent variable.
        **predictor_vars: Keyword arguments containing predictor variables as 2D arrays of shape (n_samples, n_timepoints).

    Returns:
        regress_res (DataFrame): DataFrame containing the regression results for each timepoint.
    """
    regress_res = []
    
    for t in range(data.shape[0]):
        
        y = data[t, :]
    
        # construct the dataframe for linear regression
        df2fit = pd.DataFrame({
            'signal': y,
        })
        
        if data2 is not None:
            df2fit['signal2'] = data2[t,:]
                
        for k, v in predictor_vars.items():
            df2fit[k] = v[t, :]
        
        if scale_data:
            #Normalize the data so that their ranges are the same
            scaler = preprocessing.StandardScaler()
            df2fit[df2fit.columns] = scaler.fit_transform(df2fit)
            
        
        # display(df2fit)
        mod = smf.ols(formula=formula, data=df2fit.dropna())
        res = mod.fit()
        for factor in res.params.index:
            if factor!='Intercept':
                regress_res.append({
                    'beta': res.params[factor],
                    'intercept': res.params['Intercept'] if 'Intercept' in res.params else None,  # the intercept represent the mean value
                    'pvalue': res.pvalues[factor],
                    'factor': factor,
                    'CI': res.conf_int().loc[factor].tolist(),
                    'time': xa_cond.time.data[t],
                    'residual': res.resid 
                })

    regress_res = pd.DataFrame(regress_res)

    return regress_res, res


def perform_mixed_effect_analysis(xa_cond, data, formula, data2=None, **predictor_vars):
    """
    Perform linear regression on the given data.

    Args:
        xa_cond (Xarray): Xarray object containing time data.
        data (ndarray): 2D array of shape (n_samples, n_timepoints) containing the dependent variable.
        **predictor_vars: Keyword arguments containing predictor variables as 2D arrays of shape (n_samples, n_timepoints).

    Returns:
        regress_res (DataFrame): DataFrame containing the regression results for each timepoint.
    """
    regress_res = []
    
    for t in range(data.shape[0]):
        
        y = data[t, :]
    
        # construct the dataframe for linear regression
        df2fit = pd.DataFrame({
            'signal': y,
        })
        
        if data2 is not None:
            df2fit['signal2'] = data2[t,:]
            
        
        for k, v in predictor_vars.items():
            df2fit[k] = v[t, :]
        
        # display(df2fit)
        mod = smf.ols(formula=formula, data=df2fit)
        res = mod.fit()

        for factor in res.params.index:
            if factor!='Intercept':
                regress_res.append({
                    'beta': res.params[factor],
                    'intercept': res.params['Intercept'],  # the intercept represent the mean value
                    'pvalue': res.pvalues[factor],
                    'factor': factor,
                    'CI': res.conf_int().loc[factor].tolist(),
                    'time': xa_cond.time.data[t],
                    'residual': res.resid 
                })

    regress_res = pd.DataFrame(regress_res)

    return regress_res, res


def highlight_pvalues(df_reg_res, ax, threshold=0.05,alpha=0.1):
    # highlight the significant time
    for _, row in df_reg_res.iterrows():
        if row.pvalue < threshold:
            ax.axvline(row.time, alpha=alpha, color='y')
            

def highlight_pvalues_consec_win(pvalues, time, ax, alpha=0.2, threshold=0.05, consec_win=4, consec_win_threshold=1):
    # Calculate the consecutive windows of significant and hightlight them in figure
    sig = (pvalues <threshold).astype(float)
    
    win = np.ones((consec_win,))
    consec_sig_idx = (np.convolve(sig,win,'same')>=consec_win*consec_win_threshold)
    
    sig_t = time[consec_sig_idx]
    
    for t in sig_t:
        ax.axvline(t, alpha=alpha, color='y')
            
            
def load_extraction_spec(task_name, df_conditions, specs):
    
    if task_name in ['pavlovian_spontanous_reaching_oct23',
                    'pavlovian_reaching_Oct26',
                    'pavlovian_spontanous_reaching_march23',
                    'pavlovian_spontanous_reaching_oct23']:
        
        extraction_specs = specs['spontanous_reaching']
        outcome2plot = df_conditions.trial_outcome.unique()
        
    elif task_name in ['reaching_go_spout_bar_VR_Dec23']:
        extraction_specs = specs['reaching_go_spout_bar_lick']
        outcome2plot = [['success','aborted'], 'no_reach', 'late_reach']
        
    elif task_name in ['reaching_go_spout_bar_mar23',
                    'reaching_go_spout_bar_june05',
                    'reaching_go_spout_bar_nov22',
                    'reaching_go_spout_bar_apr23']:
        extraction_specs = specs['reaching_go_spout_bar']
        outcome2plot = [['success','aborted'], 'no_reach', 'late_reach']
        
    elif task_name in ['reaching_go_spout_incr_break2_nov22']:
        extraction_specs = specs['break2']
        outcome2plot = df_conditions.trial_outcome.unique()
    else:
        extraction_specs = specs['default']
        #update the trigger
        extraction_specs['trigger'] = extraction_specs.pop('trigger')
        outcome2plot = df_conditions.trial_outcome.unique()
        
    return extraction_specs, outcome2plot

def draw_beta_values(reg_res, factor, ax,extraction_specs, display_pvalue=False):

    ax.set_title(factor)
    df2plot = reg_res[reg_res.factor==factor]
    
    #plot the CI
    ci = np.array(df2plot.CI.to_list())
    ax.fill_between(df2plot.time, ci[:,0], ci[:,1], alpha=0.2)
    
    #plot beta
    # TODO: make it easier to know whether it is strengthing the original signal or weaking
    # strengthen: making +ve signal more +ve and -ve signal more negative
    ax.axhline(0,ls='--',color='gray',alpha=0.5)
    
    ylim = ax.get_ylim()
    if ylim[1] < 0.05:
        ax.set_ylim([-1,1]) # don't give the wrong impression of a very small beta
    draw_event_line(extraction_specs, 'hold_for_water', ax, show_label=False)
    
    # add a bit of padding for text later
    ylim = ax.get_ylim()
    yrange = ylim[1] -ylim[0]
    ax.set_ylim(ylim[0], ylim[1]+yrange*0.3)
        
    add_warp_info(ax, extraction_specs, 'hold_for_water')
    
    sns.lineplot(df2plot, x='time',y='beta', ax=ax)
    ax.set_xlabel('Time around events (ms)')
    
    ticks, ticks_labels = compute_ticks(extraction_specs)
    ax.set_xticks(ticks, labels =ticks_labels, rotation=30);
    ax.margins(x=0)

    if display_pvalue:
        highlight_pvalues_consec_win(df2plot, ax, threshold=0.01, consec_win=3)


def get_warping_specs(df_events_cond, df_conditions, specs_path):

    with open('params/timewarp_spec.json') as f:
        specs = json.load(f)

        if 'task_name' in df_events_cond.attrs:
            task_name = df_events_cond.attrs['task_name']
        else:
            task_name = df_events_cond.attrs['Task name']
            
        trigger = df_events_cond.attrs['triggers'][0]

        if task_name in ['pavlovian_spontanous_reaching_oct23',
                        'pavlovian_reaching_Oct26',
                        'pavlovian_spontanous_bar_Sep24',
                        'pavlovian_spontanous_reaching_march23',
                        'pavlovian_spontanous_reaching_oct23',
                        'pavlovian_spontanous_reaching_April24']:
            
            extraction_specs = specs['spontanous_reaching']
            outcome2plot = df_conditions.trial_outcome.unique()
            
        elif task_name in ['reaching_go_spout_bar_VR_Dec23',
                        'reaching_go_spout_bar_apr23',
                        'reaching_go_spout_bar_mar23',
                        'reaching_go_spout_bar_june05',
                        'reaching_go_spout_bar_nov22']:
            extraction_specs = specs['reaching_go_spout_bar_reward']
            outcome2plot = [['success','aborted'], 'no_reach', 'late_reach']
            
        elif task_name in ['reaching_go_spout_bar_VR_April24',
                        'reaching_go_spout_bar_VR_April24_silent',
                        'reaching_go_spout_bar_VR_Feb25',
                            'reaching_go_spout_bar_VR_cued_random_June25']:
            extraction_specs = specs['reaching_go_spout_bar_reward_nogap']
            outcome2plot = ['success',['omission','jackpot'],'aborted', 'no_reach', 'late_reach']
            
        elif task_name in ['two_cue_reaching_task_January26']:
            extraction_specs = specs['two_cue_spout_reward']
            outcome2plot = [['CS_1_success','CS_2_success'],['CS_1_success','CS_1_omission'],['CS_2_success','CS_2_omission'],['CS_1_aborted','CS_2_aborted'], ['CS_1_no_reach', 'CS_2_no_reach'], ['CS_1_late_reach', 'CS_2_late_reach']]

        elif task_name in ['reaching_go_spout_incr_break2_nov22',
                        'reaching_go_spout_incr_break2_April24',
                        'reaching_go_spout_incr_break2_Feb25',
                        'reaching_go_spout_incr_break2_June25',
                        'cued_and_cued_reward_May25']:
            extraction_specs = specs['break2']
            outcome2plot = ['success', 'no_reach', 'late_reach']

        elif task_name in ['pavlovian_task_August25','pavlovian_task_August25_silent']:
            extraction_specs = specs['pavlovian_go_reward_nogap']
            outcome2plot = ['standard','omission','jackpot']

        elif task_name in ['pavlovian_task_February26']:
            extraction_specs = specs['pavlovian_reward']
            outcome2plot = ['standard','omission','jackpot']
            
        else:
            extraction_specs = specs['default']
            #update the trigger
            extraction_specs[trigger] = extraction_specs.pop('trigger')
            outcome2plot = df_conditions.trial_outcome.unique()

    return trigger,extraction_specs,outcome2plot

def normalize_signal(data_array, baseline_period:list):
    """
    Normalize a signal by subtracting the baseline period mean.

    Parameters:
    data_array (xarray.DataArray): The input data array containing the signal to be normalized.
    baseline_period (list): A list containing two elements [start_time, end_time] that define the period
                            over which the baseline mean is calculated.

    Returns:
    xarray.DataArray: The normalized data array with the baseline mean subtracted from the original signal.
    """
    # Substracting the baseline from the whole signal
    baseline = data_array.sel(time=slice(baseline_period[0],baseline_period[1])).mean(dim='time',skipna=True)
    da_norm = data_array-baseline
    return da_norm


def plot_timewarp_nan_debug(xr_warped, signal_var, extraction_specs, trigger,
                            trial_indices=None, max_trials=5, output_file=None,
                            figsize=(14, 3.5)):
    """
    Create diagnostic plots to visualize partial NaN patterns in time-warped data.

    This function helps debug why some trials have NaN values only at the end (or beginning)
    rather than the entire trial. It visualizes the valid data regions, NaN regions, and
    event boundaries to understand where interpolation is failing.

    Parameters:
    -----------
    xr_warped : xarray.Dataset
        The time-warped dataset containing the signal and trial metadata
    signal_var : str
        Name of the signal variable to plot (e.g., 'zscored_df_over_f')
    extraction_specs : dict
        Dictionary of extraction specifications defining event windows and padding
    trigger : str
        Name of the trigger event
    trial_indices : list, optional
        List of trial indices to plot. If None, will automatically find trials with partial NaN
    max_trials : int, default=5
        Maximum number of trials to plot (if trial_indices not specified)
    output_file : str, optional
        Path to save the figure. If None, returns the figure object without saving
    figsize : tuple, default=(14, 3.5)
        Figure size per subplot (width, height per trial)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    trial_info : list of dict
        Information about each plotted trial including NaN statistics

    Examples:
    ---------
    >>> fig, info = plot_timewarp_nan_debug(xr_warped, 'zscored_df_over_f',
    ...                                      extraction_specs, 'hold_for_water',
    ...                                      output_file='debug_nan.png')
    >>> print(f"Found {len(info)} trials with partial NaN")
    """

    data = xr_warped[signal_var].data

    # Find trials with partial NaN if not specified
    if trial_indices is None:
        trials_with_partial_nan = []
        for trial_idx in range(data.shape[0]):
            trial_data = data[trial_idx, :]
            nan_mask = np.isnan(trial_data)
            # Partial NaN: some but not all values are NaN
            if np.any(nan_mask) and not np.all(nan_mask):
                trials_with_partial_nan.append(trial_idx)

        trial_indices = trials_with_partial_nan[:max_trials]

        if len(trial_indices) == 0:
            print("No trials with partial NaN found!")
            return None, []

    # Prepare figure
    n_trials = len(trial_indices)
    fig, axes = plt.subplots(n_trials, 1, figsize=(figsize[0], figsize[1] * n_trials))
    if n_trials == 1:
        axes = [axes]

    trial_info = []

    for i, trial_idx in enumerate(trial_indices):
        trial_nb = xr_warped.trial_nb.data[trial_idx]
        trial_outcome = xr_warped.trial_outcome.data[trial_idx] if 'trial_outcome' in xr_warped else 'unknown'
        trial_data = data[trial_idx, :]
        time_data = xr_warped.time.data

        ax = axes[i]

        # Calculate NaN statistics
        nan_mask = np.isnan(trial_data)
        valid_mask = ~nan_mask
        nan_indices = np.where(nan_mask)[0]
        valid_indices = np.where(valid_mask)[0]

        first_nan_idx = nan_indices[0] if len(nan_indices) > 0 else None
        last_valid_idx = valid_indices[-1] if len(valid_indices) > 0 else None

        trial_info.append({
            'trial_nb': trial_nb,
            'trial_idx': trial_idx,
            'trial_outcome': trial_outcome,
            'n_valid': np.sum(valid_mask),
            'n_nan': np.sum(nan_mask),
            'first_nan_idx': first_nan_idx,
            'first_nan_time': time_data[first_nan_idx] if first_nan_idx is not None else None,
            'last_valid_idx': last_valid_idx,
            'last_valid_time': time_data[last_valid_idx] if last_valid_idx is not None else None,
        })

        # Plot valid data
        ax.plot(time_data[valid_mask], trial_data[valid_mask], 'b-',
               linewidth=1.5, label='Valid data', zorder=2)

        # Highlight NaN regions
        if np.any(nan_mask):
            nan_start = time_data[nan_mask][0]
            nan_end = time_data[nan_mask][-1]
            ax.axvspan(nan_start, nan_end, alpha=0.3, color='red',
                      label='NaN region', zorder=1)
            ax.axvline(nan_start, color='red', linestyle='--',
                      linewidth=2, alpha=0.7, zorder=3)

        # Mark event boundaries from extraction specs
        cur_time = extraction_specs[trigger]['event_window'][0]
        colors = plt.cm.tab10.colors
        for j, (evt, spec) in enumerate(extraction_specs.items()):
            pre, post = spec['event_window']
            padding = spec['padding']

            ax.axvline(cur_time + pre, color=colors[j], linestyle=':',
                      alpha=0.5, zorder=3)
            label = spec.get('label', evt.replace('_', ' '))
            ax.text(cur_time + pre, ax.get_ylim()[1], label, rotation=90,
                   ha='right', va='top', fontsize=8, alpha=0.7)

            cur_time += (post - pre) + padding

        # Set title with trial info
        title = f'Trial {trial_nb} (index {trial_idx})'
        if trial_outcome != 'unknown':
            title += f' - Outcome: {trial_outcome}'
        title += f' - {signal_var}'
        ax.set_title(title, fontweight='bold', fontsize=10)

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Signal')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file is not None:
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Debug plot saved: {output_file}")

    return fig, trial_info


def plot_trial_warp_debug(
    df_events_cond,
    extraction_specs,
    trigger,
    trial_nbs=None,
    max_trials=20,
    output_file=None,
    figsize=None,
):
    """Debug plot showing event geometry in original trial time.

    For each trial, draws a timeline with:
    - Protected event windows (colored solid bands)
    - Padding regions (hatched bands)
    - Overlapping regions (red bands with 'OVERLAP' label)
    - Trigger position (black vertical line at x=0)
    - Event timestamps as vertical dashed lines with annotations

    Parameters
    ----------
    df_events_cond : pd.DataFrame
        Events with columns: trial_nb, content, time, trial_outcome
    extraction_specs : dict
        Loaded from timewarp_spec.json (already filtered to task)
    trigger : str
        Trigger event name (must be a key in extraction_specs)
    trial_nbs : list[int], optional
        Explicit trial numbers to plot. If None, auto-detects failed trials.
    max_trials : int
        Max number of auto-detected failed trials to show (default 20)
    output_file : str or Path, optional
        Where to save the figure
    figsize : tuple, optional
        (width, height_per_subplot)

    Returns
    -------
    fig, axes
    """
    # ---- Determine which trials to plot --------------------------------
    if trial_nbs is None:
        # Auto-detect failed trials by simulating interp_data logic
        all_trial_nbs = df_events_cond['trial_nb'].unique()
        failed_trial_nbs = []
        for tnb in all_trial_nbs:
            df_trial = df_events_cond[df_events_cond['trial_nb'] == tnb]
            try:
                _simulate_warp(df_trial, extraction_specs, trigger)
            except ValueError:
                failed_trial_nbs.append(tnb)
        trial_nbs = failed_trial_nbs[:max_trials]
    else:
        trial_nbs = trial_nbs[:max_trials]

    if len(trial_nbs) == 0:
        logger.info('plot_trial_warp_debug: no trials to plot')
        fig, ax = plt.subplots(1, 1, figsize=(12, 2))
        ax.text(0.5, 0.5, 'No failed trials found', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.axis('off')
        if output_file is not None:
            fig.savefig(output_file, dpi=150, bbox_inches='tight')
        return fig, [ax]

    # ---- Color palette per event name ----------------------------------
    event_names = list(extraction_specs.keys())
    colors = plt.cm.tab10.colors
    event_colors = {evt: colors[i % len(colors)] for i, evt in enumerate(event_names)}

    # ---- Build figure --------------------------------------------------
    n_trials = len(trial_nbs)
    w, h_per = figsize if figsize else (14, 1.8)
    fig, axes = plt.subplots(n_trials, 1, figsize=(w, h_per * n_trials),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]

    # Shared legend handles collected during first trial
    legend_handles = {}

    for ax, trial_nb in zip(axes, trial_nbs):
        df_trial = df_events_cond[df_events_cond['trial_nb'] == trial_nb]

        # Determine trial outcome
        trial_outcome = 'unknown'
        if 'trial_outcome' in df_trial.columns and len(df_trial) > 0:
            trial_outcome = df_trial.iloc[0]['trial_outcome']

        # Find trigger time
        trig_rows = df_trial[df_trial['content'] == trigger]
        if len(trig_rows) == 0:
            ax.set_title(f'Trial {trial_nb} | {trial_outcome} — TRIGGER NOT FOUND',
                         color='red', fontweight='bold', fontsize=9)
            ax.axis('off')
            continue
        t_trigger = trig_rows.iloc[0]['time']

        trigger_specs = extraction_specs[trigger]
        event_specs = {k: v for k, v in extraction_specs.items() if k != trigger}

        # cur_time tracks end of last event window (absolute ms)
        cur_time = t_trigger + trigger_specs['event_window'][1]
        padding = trigger_specs['padding']

        failed = False
        failure_msgs = []

        # Draw trigger protected window
        pre_t, post_t = trigger_specs['event_window']
        rel_pre = pre_t   # relative to t_trigger
        rel_post = post_t
        color = event_colors[trigger]
        h = ax.axvspan(rel_pre, rel_post, alpha=0.45, color=color,
                       label=trigger if trigger not in legend_handles else '_nolegend_')
        if trigger not in legend_handles:
            legend_handles[trigger] = h
        ax.axvline(0, color='black', linewidth=2, zorder=5)
        ax.text(0, 1.02, f'{trigger}\n@0ms', ha='center', va='bottom',
                fontsize=7, transform=ax.get_xaxis_transform(), color='black')

        for evt, specs in event_specs.items():
            dependent_event = specs.get('dependent_event', None)
            alternative = specs.get('alternative', None)
            event_row = extract_event(df_trial, evt, specs['order'], dependent_event, alternative)

            if event_row is not None:
                t_event = event_row['time']
            else:
                # Fallback: place event at cur_time + padding - pre_window
                t_event = cur_time + padding - specs['event_window'][0]

            pre, post = specs['event_window']
            pre_abs = t_event + pre      # absolute
            post_abs = t_event + post    # absolute

            # Relative to trigger
            rel_evt = t_event - t_trigger
            rel_pre_evt = pre_abs - t_trigger
            rel_post_evt = post_abs - t_trigger
            rel_cur = cur_time - t_trigger

            color = event_colors[evt]

            # Padding region [cur_time, t_event+pre] — may be zero or negative
            pad_start = rel_cur
            pad_end = rel_pre_evt
            if pad_end > pad_start:
                h = ax.axvspan(pad_start, pad_end, alpha=0.18, color=color,
                               hatch='///', label=f'{evt} (padding)' if f'{evt}_pad' not in legend_handles else '_nolegend_')
                if f'{evt}_pad' not in legend_handles:
                    legend_handles[f'{evt}_pad'] = h

            # Check for overlap
            overlap_amount = cur_time - pre_abs  # positive = overlap
            if overlap_amount > 0:
                failed = True
                msg = (f'{evt}: cur_time={cur_time - t_trigger:.0f}ms > '
                       f'{evt}_pre={rel_pre_evt:.0f}ms (overlap={overlap_amount:.0f}ms)')
                failure_msgs.append(msg)
                # Draw overlap region
                overlap_start = rel_pre_evt
                overlap_end = rel_cur
                h = ax.axvspan(overlap_start, overlap_end, alpha=0.7, color='red',
                               label='OVERLAP' if 'OVERLAP' not in legend_handles else '_nolegend_',
                               zorder=4)
                if 'OVERLAP' not in legend_handles:
                    legend_handles['OVERLAP'] = h
                ax.text((overlap_start + overlap_end) / 2, 0.5, 'OVERLAP',
                        ha='center', va='center', fontsize=7, fontweight='bold',
                        color='white', transform=ax.get_xaxis_transform(), zorder=5)

            # Protected event window
            h = ax.axvspan(rel_pre_evt, rel_post_evt, alpha=0.45, color=color,
                           label=evt if evt not in legend_handles else '_nolegend_')
            if evt not in legend_handles:
                legend_handles[evt] = h

            # Event timestamp line and annotation
            ax.axvline(rel_evt, color=color, linestyle='--', linewidth=1.2, alpha=0.8, zorder=3)
            label = specs.get('label', evt)
            ax.text(rel_evt, 1.02, f'{label}\n@{rel_evt:.0f}ms',
                    ha='center', va='bottom', fontsize=7,
                    transform=ax.get_xaxis_transform(), color=color)

            # Advance cur_time
            cur_time = post_abs + 1
            padding = specs['padding']

        # Title
        title = f'Trial {trial_nb} | {trial_outcome}'
        title_color = 'red' if failed else 'green'
        ax.set_title(title, color=title_color, fontweight='bold', fontsize=9)

        # Failure reason text inside subplot
        if failure_msgs:
            ax.text(0.01, 0.1, '\n'.join(failure_msgs), ha='left', va='bottom',
                    fontsize=6.5, transform=ax.transAxes, color='darkred',
                    bbox=dict(boxstyle='round,pad=0.2', fc='lightyellow', alpha=0.8))

        ax.set_yticks([])
        ax.set_ylim(0, 1)
        ax.grid(axis='x', alpha=0.3)

    axes[-1].set_xlabel('Time relative to trigger (ms)')

    # Shared legend at figure top
    if legend_handles:
        fig.legend(list(legend_handles.values()), list(legend_handles.keys()),
                   loc='upper center', ncol=min(len(legend_handles), 6),
                   fontsize=8, bbox_to_anchor=(0.5, 1.0),
                   framealpha=0.9)

    fig.suptitle(f'Time Warp Debug — trigger: {trigger}  ({n_trials} trials)',
                 y=1.02 + 0.015, fontsize=11, fontweight='bold')
    fig.tight_layout()

    if output_file is not None:
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.info(f'Timewarp debug plot saved: {output_file}')

    return fig, axes


def _simulate_warp(df_trial, extraction_specs, trigger):
    """Replicate the overlap-check logic of interp_data() without signal data.

    Raises ValueError if any event's protected window overlaps cur_time.
    """
    trig_rows = df_trial[df_trial['content'] == trigger]
    if len(trig_rows) == 0:
        raise ValueError(f'Trigger {trigger} not found in trial')

    t_trigger = trig_rows.iloc[0]['time']
    trigger_specs = extraction_specs[trigger]
    cur_time = t_trigger + trigger_specs['event_window'][1]
    padding = trigger_specs['padding']

    event_specs = {k: v for k, v in extraction_specs.items() if k != trigger}
    for evt, specs in event_specs.items():
        dependent_event = specs.get('dependent_event', None)
        alternative = specs.get('alternative', None)
        event_row = extract_event(df_trial, evt, specs['order'], dependent_event, alternative)

        if event_row is not None:
            t_event = event_row['time']
        else:
            t_event = cur_time + padding - specs['event_window'][0]

        if cur_time > t_event + specs['event_window'][0]:
            raise ValueError(
                f'Overlap detected for event {evt}: '
                f'cur_time={cur_time:.1f} > t_event+pre={t_event + specs["event_window"][0]:.1f}'
            )

        cur_time = t_event + specs['event_window'][1] + 1
        padding = specs['padding']


def print_time_warping_summary(xr_warped, signal_var):
    """Print summary statistics of time warping results.
    
    Parameters
    ----------
    xr_warped : xr.Dataset
        Time-warped dataset containing signal variables
    signal2analyze : list
        List of signal variable names to analyze
    """
    logger.debug(f'\n{"="*70}')
    logger.debug("TIME WARPING SUMMARY")
    logger.debug('='*70)

    # Check each signal variable
    if signal_var in xr_warped.data_vars:
        data = xr_warped[signal_var].data

        # Count different types of trials
        fully_valid = np.sum(np.all(~np.isnan(data), axis=1))
        fully_nan = np.sum(np.all(np.isnan(data), axis=1))
        partial_nan = np.sum(np.any(np.isnan(data), axis=1) & ~np.all(np.isnan(data), axis=1))
        total = data.shape[0]

        logger.debug(f'\n{signal_var}:')
        logger.debug(f'  Total trials: {total}')
        logger.debug(f'  Fully valid trials: {fully_valid} ({fully_valid/total*100:.1f}%)')
        logger.debug(f'  Fully NaN trials: {fully_nan} ({fully_nan/total*100:.1f}%)')
        logger.debug(f'  Partial NaN trials: {partial_nan} ({partial_nan/total*100:.1f}%)')

        # Check successful trials specifically
        if 'trial_outcome' in xr_warped:
            xr_success = xr_warped.sel(trial_nb=(xr_warped.trial_outcome=='success'))
            if len(xr_success.trial_nb) > 0:
                success_data = xr_success[signal_var].data
                valid_success = np.sum(np.all(~np.isnan(success_data), axis=1))
                total_success = success_data.shape[0]
                logger.debug(f'  Valid successful trials: {valid_success}/{total_success} ({valid_success/total_success*100:.1f}%)')
