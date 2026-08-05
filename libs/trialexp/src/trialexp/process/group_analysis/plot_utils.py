import matplotlib.pylab as plt
import seaborn as sns
import numpy as np 
import pandas as pd
from trialexp.process.pycontrol.plot_utils import trial_outcome_palette
from scipy import signal
np.random.seed(0) #for reproducibility

def calculate_grand_trial_nb(df):
    # calcualte the grand trial nubmer across sessions

    df = df.sort_values('expt_datetime')
    
    # sort by expt date time and then cumsum over trial
    df = df.dropna()
    df = df[['animal_id','expt_datetime','trial_nb','trial_outcome']]
    df_trial_nb = df.groupby(['animal_id','expt_datetime','trial_nb','trial_outcome']).first().reset_index()
    df_trial_nb = df_trial_nb.sort_values(['expt_datetime','trial_nb']) # to be safe
    df_trial_nb['grand_trial_nb'] = 1
    df_trial_nb['grand_trial_nb'] = df_trial_nb.groupby(['animal_id','trial_outcome'])['grand_trial_nb'].transform(pd.Series.cumsum)
    return df_trial_nb

def sample_trials(df_subject, n_sample):
    # randomly sample the trial of each subject
    
    if not 'grand_trial_nb' in df_subject:
        raise ValueError('grand_trial_nb must be in dataframe')
    
    max_trial_nb = df_subject.grand_trial_nb.max() #note: the meaning of this max may be different depending on whether grand trial nb is counted for each trial outcome respectively
    
    if type(n_sample) is int:
        sel_trials = np.random.choice(np.arange(max_trial_nb)+1, n_sample, replace=False)
    elif type(n_sample) is pd.Series:
        # we assume that the data is already grouped by trial_outcome
        trial_outcome = df_subject.iloc[0].trial_outcome
        n = n_sample[trial_outcome]
        sel_trials = np.random.choice(np.arange(max_trial_nb)+1, n, replace=False)
        
    return df_subject[df_subject.grand_trial_nb.isin(sel_trials)]

def equal_subsample_trials(df2plot):
    # Randomly sample a fix number of trial for each subjects
    
    # calcualte the grand trial number across sessions
    df_trial_nb = calculate_grand_trial_nb(df2plot)
    
    # merge the grand trial number back to the original dataframe
    df2plot = df2plot.merge(df_trial_nb, on=['expt_datetime','trial_nb','animal_id','trial_outcome'])

    # Use the min no. trial of all animals as the sample, separate sampling according to trial_outcome
    n_sample_outcome = df2plot.groupby(['animal_id','trial_outcome']).grand_trial_nb.max()
    n_sample_outcome = n_sample_outcome.groupby('trial_outcome').min()
    # print(n_sample_outcome)
        
    # sample
    df2plot = df2plot.groupby(['animal_id','trial_outcome'], group_keys=False).apply(sample_trials, n_sample=n_sample_outcome).reset_index()
    #drop the animal_id level so that we can reset back to the normal dataframe
    
    return df2plot
    

def style_plot():
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['legend.frameon'] = False

def plot_subject_average(ds_combined, animal_id, var_name, errorbar='ci', n_boot=1000):
    
    style_plot()

    df2plot = ds_combined[[var_name, 'trial_outcome','session_id']].to_dataframe().reset_index()

    #only plot several outcomes
    sel_trial_outcome = ['success', 'aborted']
    df2plot = df2plot[df2plot.trial_outcome.isin(sel_trial_outcome)]


    #merge the animal_id back to the data frame
    df2plot = df2plot.merge(animal_id, on='session_id')
    
    g = sns.relplot(x='event_time',y=var_name, hue='animal_id',
                col='trial_outcome', col_wrap=3, kind='line', n_boot=n_boot, data=df2plot, height=6, errorbar=errorbar)


    
    #Styling
    
    for ax in g.axes:
        ax.axvline(0,ls='--',color='gray')

    # sns.move_legend(g, "upper right", bbox_to_anchor=[0.75,1])
    
        
    g.set(xlim=[-1000, 1500])
        
    label = var_name.replace('_zscored_df_over_f', '')
    label = label.upper().replace('_', ' ')
    
    g.set_xlabels(f'Relative time (ms)\n{label}')

    g.set_ylabels(f'z-scored dF/F')
    g.set_titles(col_template='{col_name}')
    
    # g.fig.tight_layout()
    
    #calculate the number of trial for each animal
    df = df2plot.groupby(['animal_id','trial_nb']).first().reset_index()
    trial_count = df.groupby('animal_id')['trial_nb'].count()
    
    # set the legend with number of trials
    handles, labels = ax.get_legend_handles_labels()
    print(labels)
    
    return g, g.figure,df2plot



def plot_subject_average_by_outcome(ds_combined, animal_id, var_name, trial_outcome, xlabel=None, ax=None, errorbar='ci', n_boot=1000):
    # use subplots for each trial_outcome so that we can have the clear indication of trials
    
    style_plot()
    
    if ax is None:
        fig, ax = plt.subplots(1,1,dpi=300, figsize=(6,6))

    df2plot = ds_combined[[var_name, 'trial_outcome','session_id']].to_dataframe().reset_index()

    #only plot several outcomes
    df2plot = df2plot[df2plot.trial_outcome==trial_outcome]


    #merge the animal_id back to the data frame
    df2plot = df2plot.merge(animal_id, on='session_id')
    
    sns.lineplot(x='event_time',y=var_name, hue='animal_id', n_boot=n_boot, data=df2plot, errorbar=errorbar, ax = ax)


    
    #Styling
    ax.axvline(0,ls='--',color='gray')

    # sns.move_legend(g, "upper right", bbox_to_anchor=[0.75,1])
    
        
    ax.set(xlim=[-1000, 1500])
        
    label = var_name.replace('_zscored_df_over_f', '')
    label = label.upper().replace('_', ' ')
    
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(f'Relative time (ms)\n{label}')
    
    ax.set_ylabel(f'z-scored dF/F')
    ax.set_title(trial_outcome)
    
    # g.fig.tight_layout()
    
    #calculate the number of trial for each animal
    df = df2plot.groupby(['animal_id','trial_nb']).first().reset_index()
    trial_count = df.groupby('animal_id')['trial_nb'].count()
    
    # set the legend with number of trials
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [f'{lbl} ($n_t$={trial_count[lbl]})' for lbl in labels]
    ax.legend(handles, new_labels,loc='upper right', bbox_to_anchor=[1,1], fontsize="10")
    
    return ax


def compute_num_trials_by_outcome(df2plot):
    # compute the number of trials in each trial_outcome
    if not 'grand_trial_nb' in df2plot.columns:
        raise ValueError('grand_trial_nb must be calculated first')
        
    df = df2plot.groupby(['trial_outcome','animal_id','grand_trial_nb']).first().reset_index()
    return df.groupby('trial_outcome').index.count()

def plot_group_average(ds_combined, animal_id, var_name, title='None', xlabel=None,
                       ax=None, n_boot=1000, errorbar='ci', average_method='trial'):
    # average_method = mean_of_mean, equal_subsample, trial 
    # animal_id: dataframe containing the correspondence between 
    
    style_plot()
    
    if ax is None:
        fig, ax = plt.subplots(1,1,dpi=300, figsize=(6,6))
    
    df2plot = ds_combined[[var_name, 'trial_outcome','session_id']].to_dataframe().reset_index()

    #only plot several outcomes
    sel_trial_outcome = ['success', 'aborted']
    df2plot = df2plot[df2plot.trial_outcome.isin(sel_trial_outcome)]


    #merge the animal_id back to the data frame
    df2plot = df2plot.merge(animal_id, on='session_id')
    
    # Use different ways to average the data
    if average_method =='mean_of_mean':
        df2plot = df2plot.groupby(['animal_id','event_time','trial_outcome']).mean().reset_index()
        n_sample = len(animal_id.animal_id.unique())
    elif average_method == 'equal_subsample':
        df2plot= equal_subsample_trials(df2plot)
        n_sample_outcome = compute_num_trials_by_outcome(df2plot)
    
    ax = sns.lineplot(x='event_time',y=var_name, hue_order=sel_trial_outcome,
                    hue='trial_outcome', n_boot=n_boot, errorbar=errorbar, palette=trial_outcome_palette,
                    data=df2plot,ax=ax)
    
    # styling
    ax.axvline(0,ls='--',color='gray')
    ax.set_xlim([-1000,1500])
    label = var_name.replace('_zscored_df_over_f', '')
    label = label.upper().replace('_', ' ')
    ax.set_ylabel(f'z-scored dF/F')
    
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(f'Relative time (ms)\n{label}')
    
    # determine the proper legend labels to use
    handles, labels = ax.get_legend_handles_labels()
    if average_method == 'equal_subsample':
        new_labels = [f'{s} ($n_t$={n_sample_outcome[s]})' for s in labels]
        ax.legend(handles, new_labels)
    elif average_method =='mean_of_mean':
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [f'{s} ($n_s$={n_sample})' for s in labels]
        ax.legend(handles, new_labels)
        
    if title is not None:
        ax.set_title(title)
    sns.move_legend(ax, title='', loc='upper right', bbox_to_anchor=[1.4,1])
    
    return ax, df2plot


def plot_subject_comparison(ds_combined, animal_id, var_name):

    df2plot = ds_combined[[var_name, 'trial_outcome','session_id']].to_dataframe().reset_index()

    #only plot several outcomes
    sel_trial_outcome = ['success', 'aborted']
    df2plot = df2plot[df2plot.trial_outcome.isin(sel_trial_outcome)]


    #merge the animal_id back to the data frame
    df2plot = df2plot.merge(animal_id, on='session_id')

    g = sns.relplot(x='event_time',y=var_name, hue='trial_outcome',
                col='animal_id', col_wrap=3, kind='line', n_boot=100, data=df2plot)
    
    g.set(xlim=[-1000, 1500])
    sns.move_legend(g, "upper right", bbox_to_anchor=[1,1])
    
    return g.figure

def get_average_curve(ds_combined, animal_id, var_name, average_method='equal_subsample'):
    df2plot = ds_combined[[var_name, 'trial_outcome','session_id']].to_dataframe().reset_index()
    sel_trial_outcome = ['success', 'aborted']
    df2plot = df2plot[df2plot.trial_outcome.isin(sel_trial_outcome)]
    df2plot = df2plot.merge(animal_id, on='session_id')
    
    if average_method =='mean_of_mean':
        df2plot = df2plot.groupby(['animal_id','event_time','trial_outcome']).mean().reset_index()
    elif average_method == 'equal_subsample':
        df2plot= equal_subsample_trials(df2plot)

    dfmean = df2plot.groupby(['event_time','trial_outcome'])[var_name].mean().reset_index()

    return dfmean

def filter_peaks(peaks, peak_props,max=3):
    # only select the first few prominent peak
    idx =  np.argsort(peak_props['prominences'])[::-1][:max]
    return peaks[idx]

def analyze_peak(curve):
    #smooth curve
    curve_smooth = signal.savgol_filter(curve,21,2)
    
    #find both positive and negative peaks
    peaks_neg, props_neg = signal.find_peaks(-curve_smooth, prominence=0.1)
    peaks_pos, props_pos = signal.find_peaks(curve_smooth, prominence=0.1)
    
    peaks_neg = filter_peaks(peaks_neg, props_neg)
    peaks_pos = filter_peaks(peaks_pos, props_pos)
    
    return curve_smooth, peaks_neg, peaks_pos

def plot_peak_curves(dfmean, outcome,var_name, title, ax, xlabel=None):
    # plot the smoothed curve and its peak position
    style_plot()
    x = dfmean[dfmean.trial_outcome == outcome]
    x = x.sort_values('event_time')
    curve = x[var_name].values
    event_time = x['event_time'].values

    curve_smooth, peaks_neg, peaks_pos = analyze_peak(curve)
    ax.plot(event_time, curve, alpha=0.4, color=trial_outcome_palette[outcome])
    ax.plot(event_time, curve_smooth, color=trial_outcome_palette[outcome])

    peaks = np.concatenate([peaks_neg, peaks_pos])

    for p in peaks_pos:
        ax.annotate(event_time[p], xy=(event_time[p], curve_smooth[p]), xytext=(10, 5),
                     textcoords='offset pixels')
        ax.vlines(x=event_time[p], ymin=curve_smooth[p]-0.1, ymax=curve_smooth[p]+0.1, color='gray', ls='--')

    for p in peaks_neg:
        ax.annotate(event_time[p], xy=(event_time[p], curve_smooth[p]), xytext=(10, -30),
                     textcoords='offset pixels')
        ax.vlines(x=event_time[p], ymin=curve_smooth[p]-0.1, ymax=curve_smooth[p]+0.1, color='gray', ls='--')

    ax.axvline(0,ls='--',color='gray')

    #styling
    
    ax.set_xlim([-1000,1500])
    
    label = var_name.replace('_zscored_df_over_f', '')
    label = label.upper().replace('_', ' ')
    
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(f'Relative time (ms)\n{label}')
    
    ax.set_ylabel(f'z-scored dF/F')
    ax.set_title(f'{title} \n {outcome}')


def plot_peak_analysis_comparison(ds_combined, animal_id, var_name, outcome, title,  ax, xlabel=None):
    dfmean = get_average_curve(ds_combined, animal_id, var_name)
    
    plot_peak_curves(dfmean, outcome,var_name, title, ax, xlabel=xlabel)
    
def get_session_time(df):
    p25 = np.percentile(df.success_trial_no, 25)
    p75 = np.percentile(df.success_trial_no, 75)    
    df.loc[df.success_trial_no<p25, 'session_time'] = 'early'
    df.loc[df.success_trial_no>p75, 'session_time'] = 'late'
    return df

def add_session_time(df2plot, outcome):
    df2plot = df2plot[df2plot.trial_outcome==outcome]
    
    df_session_time = df2plot.groupby(['session_id','trial_nb']).first().reset_index()
    df_session_time['success_trial_no'] = df_session_time.groupby('session_id').trial_nb.rank()
    df_session_time['session_time'] = 'middle'
    df_session_time = df_session_time.groupby('session_id').apply(get_session_time)
    df_session_time = df_session_time[['session_id','trial_nb','session_time']]
    df2plot = df2plot.merge(df_session_time, on=['session_id','trial_nb'])
    
    return df2plot


def plot_session_time_effect(ds_combined, v, outcome, title,xlabel, ax, errorbar='ci'):
    style_plot()
    
    df2plot = ds_combined[[v, 'trial_outcome','session_id']].to_dataframe().reset_index()
    df2plot = df2plot.dropna()
    df2plot = add_session_time(df2plot, outcome)
    
    ax.set_xlim([-1000,1500])
    
    sns.lineplot(df2plot, hue='session_time', errorbar=errorbar, x='event_time', y=v, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f'z-scored dF/F')
