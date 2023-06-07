import matplotlib.pylab as plt
import seaborn as sns
import numpy as np 
import pandas as pd
from trialexp.process.pycontrol.plot_utils import trial_outcome_palette

def calculate_grand_trial_nb(df):
    # calcualte the grand trial nubmer across sessions

    df = df.sort_values('expt_datetime')
    
    # sort by expt date time and then cumsum over trial
    df = df.dropna()
    df = df[['animal_id','expt_datetime','trial_nb']]
    df_trial_nb = df.groupby(['animal_id','expt_datetime','trial_nb']).first().reset_index()
    df_trial_nb = df_trial_nb.sort_values(['expt_datetime','trial_nb']) # to be safe
    df_trial_nb['grand_trial_nb'] = 1
    df_trial_nb['grand_trial_nb'] = df_trial_nb.groupby('animal_id')['grand_trial_nb'].transform(pd.Series.cumsum)
    return df_trial_nb

def sample_trials(df_subject, n_sample):
    # randomly sample the trial of each subject
    
    if not 'grand_trial_nb' in df_subject:
        raise ValueError('grand_trial_nb must be in dataframe')
    
    max_trial_nb = df_subject.grand_trial_nb.max()
    sel_trials = np.random.choice(np.arange(max_trial_nb)+1, n_sample, replace=False)
    return df_subject[df_subject.grand_trial_nb.isin(sel_trials)]

def equal_subsample_trials(df2plot):
    # Randomly sample a fix number of trial for each subjects
    
    # calcualte the grand trial number across sessions
    df_trial_nb = calculate_grand_trial_nb(df2plot)
    
    # merge the grand trial number back to the original dataframe
    df2plot = df2plot.merge(df_trial_nb, on=['expt_datetime','trial_nb','animal_id'])

    # Use the max no. trial of all animals as the sample
    n_sample = df2plot.groupby('animal_id').grand_trial_nb.max().min()
    
    print(f'Using {n_sample} trials for each subjects')
    
    # sample
    df2plot = df2plot.groupby('animal_id').apply(sample_trials, n_sample=n_sample).droplevel(0).reset_index()
    #drop the animal_id level so that we can reset back to the normal dataframe
    
    return df2plot
    

def style_plot():
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['legend.frameon'] = False

def plot_subject_average(ds_combined, animal_id, var_name, n_boot=1000):
    
    style_plot()

    df2plot = ds_combined[[var_name, 'trial_outcome','session_id']].to_dataframe().reset_index()

    #only plot several outcomes
    sel_trial_outcome = ['success', 'aborted']
    df2plot = df2plot[df2plot.trial_outcome.isin(sel_trial_outcome)]


    #merge the animal_id back to the data frame
    df2plot = df2plot.merge(animal_id, on='session_id')
    
    g = sns.relplot(x='event_time',y=var_name, hue='animal_id',
                col='trial_outcome', col_wrap=3, kind='line', n_boot=n_boot, data=df2plot, height=6)


    
    for ax in g.axes:
        ax.axvline(0,ls='--',color='gray')

    sns.move_legend(g, "upper right", bbox_to_anchor=[0.75,1])
    
        
    g.set(xlim=[-1000, 1500])
    g.fig.tight_layout()
        
    label = var_name.replace('_zscored_df_over_f', '')
    label = label.upper().replace('_', ' ')
    
    g.set_xlabels(f'Relative time (ms)\n{label}')

    g.set_ylabels(f'z-scored dF/F')
    g.set_titles(col_template='{col_name}')
    
    return g.figure

def plot_group_average(ds_combined, animal_id, var_name, n_boot=1000, average_method='trial'):
    # average_method = mean_of_mean, equal_subsample, trial 
    
    style_plot()
    
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
    elif average_method == 'equal_subsample':
        df2plot = equal_subsample_trials(df2plot)
        

    ax = sns.lineplot(x='event_time',y=var_name, hue_order=sel_trial_outcome,
                      hue='trial_outcome', n_boot=n_boot, palette=trial_outcome_palette,
                      data=df2plot,ax=ax)
    
    ax.axvline(0,ls='--',color='gray')
    ax.set_xlim([-1000,1500])
    label = var_name.replace('_zscored_df_over_f', '')
    label = label.upper().replace('_', ' ')
    ax.set_ylabel(f'z-scored dF/F')
    ax.set_xlabel(f'Relative time (ms)\n{label}')
    sns.move_legend(ax, title='', loc='upper right', bbox_to_anchor=[1.4,1])
    
    return fig,ax


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
