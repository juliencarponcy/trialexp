#%%
import pandas as pd 
from trialexp.process.pycontrol.plot_utils import *
from snakehelper.SnakeIOHelper import getSnake


#%%

(sinput, soutput) = getSnake(locals(), 'workflows/spout_bar_nov22.smk',
  ['Z:/Teris/ASAP/expt_sessions/JC317L-2022-11-23-163225/processed/task.done'],'pycontrol_figures')


#%%
df_events_cond = pd.read_pickle(sinput.event_dataframe)
trial_window = df_events_cond.attrs['trial_window']
triggers = df_events_cond.attrs['triggers']

#%% Plot the event plots
df2plot = df_events_cond[df_events_cond.name=='spout']
g = plot_event_distribution(df2plot, 'trial_time', 'trial_nb', ybinwidth=5, xlim=[trial_window[0], trial_window[1]])
trigger_text = triggers[0].replace('_', ' ')
style_event_distribution(g, 'Time (ms)', 'Trial number', trigger_text)

# %% save
g.savefig(soutput.event_histogram, dpi=300)

# %%
