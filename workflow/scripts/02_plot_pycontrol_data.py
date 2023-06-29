#%%
import pandas as pd 
from trialexp.process.pycontrol.plot_utils import *
from snakehelper.SnakeIOHelper import getSnake
from workflow.scripts import settings

#%%

(sinput, soutput) = getSnake(locals(), 'workflow/pycontrol.smk',
  [settings.debug_folder+'/processed/log/pycontrol.done'],'pycontrol_figures')

#%%
df_events_cond = pd.read_pickle(sinput.event_dataframe)
trial_window = df_events_cond.attrs['trial_window']
triggers = df_events_cond.attrs['triggers']

#%% Plot the event plots
# df2plot = df_events_cond[df_events_cond.name=='spout'].copy()
df2plot = df_events_cond.copy()
df2plot['trial_time'] = df2plot['trial_time']/1000
xlim = [trial_window[0]/1000, np.percentile(df2plot['trial_time'],95)]
g = plot_event_distribution(df2plot, 'trial_time', 'trial_nb', xbinwidth=0.1, ybinwidth=0, xlim=xlim)
trigger_text = triggers[0].replace('_', ' ')
style_event_distribution(g, 'Time (s)', 'Trial number', trigger_text)

# %% save
g.savefig(soutput.event_histogram, dpi=300)

# %%
