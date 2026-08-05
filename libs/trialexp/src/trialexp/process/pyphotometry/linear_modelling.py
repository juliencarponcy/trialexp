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
    t[cur_idx:(cur_idx+event_window_len)] = np.arange(t_trigger+trigger_specs['event_window'][0], 
                                                      t_trigger+trigger_specs['event_window'][1], 1/sampling_rate*1000)
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
            raise ValueError(f'\nEvent:  {evt}.\n'
                              'Error: no enough time to warp. \n'
                            f'Last event: {last_event} \n'
                            f'Time different from last event: {t_event-last_event_time}\n'
                            f'Available time for warping: {t_event-cur_time} \n'
                            f'Required min pre-event window: {-specs["event_window"][0]} \n'
                            f'trial outcome: {df_trial.iloc[0].trial_outcome} \n')

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

def time_warp_data(df_events_cond, xr_signal, extraction_specs, trigger, Fs,verbose=False):
    """
    Time warps the data between events so that they can be aligned together.

    Parameters:
    - df_events_cond (pandas DataFrame): DataFrame containing event conditions.
    - xr_signal (xarray DataArray): DataArray containing the signal data.
    - extraction_specs (dict): Dictionary containing extraction specifications.
    - Fs (int): Sampling frequency.

    Returns:
    - xa (xarray DataArray): DataArray containing the time-warped data.
    """
    data_list = []
    interp_results_list = []
    
    for i in df_events_cond.trial_nb.unique():
        df_trial = df_events_cond[df_events_cond.trial_nb==i]
    
        pre_time = extraction_specs[trigger]['event_window'][0]-500
        post_time = list(extraction_specs.values())[-1]['event_window'][1]
        # extract photometry data around trial
        trial_data = extract_data(xr_signal, df_trial.iloc[0].time+pre_time, df_trial.iloc[-1].time+post_time)
        
        #time wrap it
        try:
            data_p, interp_results = interp_data(trial_data, df_trial, trigger, extraction_specs, Fs)
            interp_results_list.append(interp_results)
            data_p = data_p.expand_dims({'trial_nb':[i]})
            data_list.append(data_p)
        except NotImplementedError as e:
            print(e)
        except ValueError as e:
            if verbose:
                print(f'Skipping trial {i}', e)
            
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

    with open(specs_path) as f:
        specs = json.load(f)

    if 'task_name' in df_events_cond.attrs:
        task_name = df_events_cond.attrs['task_name']
    else:
        task_name = df_events_cond.attrs['Task name']
    
    trigger = df_events_cond.attrs['triggers'][0]

    if task_name in ['pavlovian_spontanous_reaching_oct23',
                 'pavlovian_reaching_Oct26',
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
    
    elif task_name in ['reaching_go_spout_bar_VR_April24']:
        extraction_specs = specs['reaching_go_spout_bar_reward_nogap']
        outcome2plot = ['success',['omission','jackpot'],'aborted', 'no_reach', 'late_reach']
    

    elif task_name in ['reaching_go_spout_incr_break2_nov22']:
        extraction_specs = specs['break2']
        outcome2plot = df_conditions.trial_outcome.unique()
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