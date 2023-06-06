import matplotlib.pylab as plt
import seaborn as sns

def plot_subject_average(ds_combined, animal_id, var_name):

    df2plot = ds_combined[[var_name, 'trial_outcome','session_id']].to_dataframe().reset_index()

    #only plot several outcomes
    sel_trial_outcome = ['success', 'aborted']
    df2plot = df2plot[df2plot.trial_outcome.isin(sel_trial_outcome)]


    #merge the animal_id back to the data frame
    df2plot = df2plot.merge(animal_id, on='session_id')

    g = sns.relplot(x='event_time',y=var_name, hue='animal_id',
                col='trial_outcome', col_wrap=3, kind='line', n_boot=100, data=df2plot)
    
    g.set(xlim=[-1000, 1500])
    g.fig.tight_layout()
    sns.move_legend(g, "upper right", bbox_to_anchor=[0.75,1])
    
    return g.figure

def plot_group_average(ds_combined, animal_id, var_name):
    fig, ax = plt.subplots(1,1,dpi=90, figsize=(6,6))
    df2plot = ds_combined[[var_name, 'trial_outcome','session_id']].to_dataframe().reset_index()

    #only plot several outcomes
    sel_trial_outcome = ['success', 'aborted']
    df2plot = df2plot[df2plot.trial_outcome.isin(sel_trial_outcome)]


    #merge the animal_id back to the data frame
    df2plot = df2plot.merge(animal_id, on='session_id')

    ax = sns.lineplot(x='event_time',y=var_name, hue='trial_outcome', n_boot=100, data=df2plot, ax=ax)
    ax.axvline(0,ls='--',color='gray')
    ax.set_xlim([-1000,1500])
    
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
    