import matplotlib
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from matplotlib import cm, colors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import os
from loguru import logger

#define the color palette for trial_come
default_palette = plt.cm.tab20.colors
trial_outcome_palette = {
    'success': default_palette[0],
    'standard': default_palette[0],
    'aborted' : default_palette[1],
    'button_press': default_palette[2],
    'late_reach': default_palette[3],
    'no_reach': default_palette[4],
    'water_by_bar_off': default_palette[5],
    'undefined': default_palette[6],
    'not success': default_palette[7],
    'free_reward_reach': default_palette[8],
    'free_reward_no_reach': default_palette[9],
    'jackpot': default_palette[10],
    'omission': default_palette[11],
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
    if os.name == 'nt':
        plt.rcParams['font.family'] = ['Arial']
    elif os.name == 'posix':
        plt.rcParams['font.family'] = ['Lato']

    g = sns.JointGrid()
    
    #plot spout touch
    df_spout = df2plot[df2plot.content=='spout']
    if len(df_spout) == 0:
        return g
    
    ax = sns.scatterplot(y=y, x=x, marker='|' , hue='trial_outcome', palette=trial_outcome_palette,
                       data= df_spout, ax = g.ax_joint, **kwargs)
    
    if len(df_spout)>1:
        if xbinwidth>0 and len(df_spout[x].unique())>1:
            #known issue when bin range < binwidth, see https://github.com/mwaskom/seaborn/issues/3646
            # check for range of data before proceeding further
            if np.ptp(df_spout[x])>xbinwidth:
                sns.histplot(x=x, binwidth=xbinwidth, ax=g.ax_marg_x, data=df_spout)
            else:
                logger.warning('data range is smaller than bin width. Skipping histplot')
        if ybinwidth>0 and len(df_spout[y].unique())>1:
            if np.ptp(df_spout[y])>ybinwidth:
                sns.histplot(y=y, binwidth=ybinwidth, ax=g.ax_marg_y, data=df_spout)
            else:
                logger.warning('data range is smaller than bin width. Skipping histplot')

    
    #plot aborted bar off
    df_baroff = df2plot[(df2plot.content=='bar_off') & (df2plot.trial_outcome =='aborted')]
    ax = sns.scatterplot(y=y, x=x, marker='.' , hue='trial_outcome', palette=trial_outcome_palette,
                       data= df_baroff, ax = g.ax_joint, legend=False, **kwargs)

    
    # indicate the no reach condition
    df_trial = df2plot.groupby('trial_nb').first()
    df_noreach = df_trial[df_trial.trial_outcome.str.contains('no_reach')]
    ax = sns.scatterplot(y=y, x=0, marker='x' , hue='trial_outcome', palette=trial_outcome_palette,
                    data= df_noreach, ax = g.ax_joint, **kwargs)
    
    if xlim is not None:
        ax.set(xlim=xlim)


    try: 
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1.2, 1))
    except ValueError:
        pass
    
    # add another legend manually for the markers
    g.figure.text(1,0.3, '|    spout touch')
    g.figure.text(1,0.35, '*    bar off (aborted)')
    g.figure.text(1,0.4, 'x    no reach')

    return g


def style_event_distribution(g, xlabel, ylabel, trigger_name):
    # g: sns.JointGrid object from the plot_event_distribution
    g.ax_joint.axvline(0, ls='--');
    g.ax_joint.set(xlabel=xlabel, ylabel=ylabel)
    ylim = g.ax_joint.get_ylim()
    g.ax_joint.text(0, np.mean(ylim), trigger_name, ha='right',  rotation='vertical')
    
    return g


def reach_time(df_trial):
    if len(spouts := df_trial[df_trial['content']=='spout'])>0:
        first_spout_time = spouts.iloc[0].time
        df_win = df_trial[df_trial.time<first_spout_time]
        if len(bar_off := df_win[df_win['content']=='bar_off'])>0:
            last_bar_time = bar_off.iloc[-1].time
            return first_spout_time - last_bar_time

        else:
            return None
    return None