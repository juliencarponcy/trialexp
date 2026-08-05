#%%
import pandas as pd 
from trialexp.process.pycontrol.plot_utils import *
from snakehelper.SnakeIOHelper import getSnake
from trialexp import config
from trialexp.process.pycontrol.utils import get_windowed_discriminability_score, discrminability_before_after

#%%

(sinput, soutput) = getSnake(locals(), 'workflow/pycontrol.smk',
  [config.debug_folder+'/processed/log/pycontrol.done'],'pycontrol_figures')

#%%
df_events_cond = pd.read_pickle(sinput.event_dataframe)
trial_window = df_events_cond.attrs['trial_window']
triggers = df_events_cond.attrs['triggers']

df_pycontrol = pd.read_pickle(sinput.df_pycontrol)
#%% Plot the event plots
df2plot = df_events_cond.copy()
df2plot['trial_time'] = df2plot['trial_time']/1000
xlim = [trial_window[0]/1000, np.percentile(df2plot['trial_time'],95)]

#%%
g = plot_event_distribution(df2plot, 'trial_time', 'trial_nb', xbinwidth=0.1, ybinwidth=0, xlim=xlim)
trigger_text = triggers[0].replace('_', ' ')
style_event_distribution(g, 'Time (s)', 'Trial number', trigger_text)


g.savefig(soutput.event_histogram, dpi=100)

# %% Plot the reach time
fig,ax = plt.subplots(1,1,dpi=200)
reach_t = df_events_cond.groupby('trial_nb').apply(reach_time)
reach_t = reach_t.dropna()
sns.histplot(reach_t,bins = 50, binrange=[0,500], ax=ax)
ax.set(xlabel='Reach time (ms)')
fig.savefig(soutput.reach_histogram, dpi=300)

# %% Also calculate the discriminative index for the break2 task
fig,ax = plt.subplots(1,1,dpi=200)
task_name = df_pycontrol.attrs['task_name'] 
if 'break2' in task_name or 'cued_and_cued_reward' in task_name:
    df_dprime = get_windowed_discriminability_score(df_pycontrol, window_sec=3*60)
    sns.lineplot(df_dprime, x='time',y='dprime', ax=ax)
    ax.axhline(0,ls='--', color='gray')

fig.savefig(soutput.discrim_scores)

# %%
