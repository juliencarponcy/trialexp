import matplotlib
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from matplotlib import cm, colors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

#define the color palette for trial_come
default_palette = sns.color_palette()
trial_outcome_palette = {
    'success': default_palette[0],
    'aborted' : default_palette[1],
    'button_press': default_palette[2],
    'late_reach': default_palette[3],
    'no_reach': default_palette[4],
    'water_by_bar_off': default_palette[5],
    'undefined': default_palette[6],
    'not success': default_palette[7],
    'free_reward_reach': default_palette[8],
    'free_reward_no_reach': default_palette[9],
    
}

def plot_event_distribution(df2plot, x, y, xbinwidth = 100, ybinwidth=100, xlim=None, **kwargs):
    # kwargs: keyword argument that will be passed to the underlying sns.scatterplot function
    #   can be used to configure additional plotting scales
    # # Use joingrid directly because the x and y axis are usually in different units

    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams["legend.frameon"] = False

    g = sns.JointGrid()
    
    #plot spout touch
    df_spout = df2plot[df2plot.name=='spout']
    ax = sns.scatterplot(y=y, x=x, marker='|' , hue='trial_outcome', palette=trial_outcome_palette,
                       data= df_spout, ax = g.ax_joint, **kwargs)
    
    sns.histplot(x=x, binwidth=xbinwidth, ax=g.ax_marg_x, data=df_spout)
    if ybinwidth>0 and len(df2plot[y].unique())>1:
        sns.histplot(y=y, binwidth=ybinwidth, ax=g.ax_marg_y, data=df_spout)
    
    #plot aborted bar off
    df_baroff = df2plot[(df2plot.name=='bar_off') & (df2plot.trial_outcome =='aborted')]
    ax = sns.scatterplot(y=y, x=x, marker='.' , hue='trial_outcome', palette=trial_outcome_palette,
                       data= df_baroff, ax = g.ax_joint, legend=False, **kwargs)

    
    # indicate the no reach condition
    df_trial = df2plot.groupby('trial_nb').first()
    df_noreach = df_trial[df_trial.trial_outcome.str.contains('no_reach')]
    ax = sns.scatterplot(y=y, x=0, marker='x' , hue='trial_outcome', palette=trial_outcome_palette,
                    data= df_noreach, ax = g.ax_joint, **kwargs)
    
    if xlim is not None:
        ax.set(xlim=xlim)


        
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.2, 1))

    return g


def style_event_distribution(g, xlabel, ylabel, trigger_name):
    # g: sns.JointGrid object from the plot_event_distribution
    g.ax_joint.axvline(0, ls='--');
    g.ax_joint.set(xlabel=xlabel, ylabel=ylabel)
    ylim = g.ax_joint.get_ylim()
    g.ax_joint.text(0, np.mean(ylim), trigger_name, ha='right',  rotation='vertical')
    
    return g