#%%
import pandas as pd 
from trialexp.process.pycontrol.plot_utils import *

#%%
df_events_cond = pd.read_pickle('sample_data/df_events_cond.pkl')
trial_window = df_events_cond.attrs['trial_window']
triggers = df_events_cond.attrs['triggers']

#%% Plot the event plots
df2plot = df_events_cond[df_events_cond.name=='spout']
g = plot_event_distribution(df2plot, 'trial_time', 'trial_nb', ybinwidth=5, xlim=[trial_window[0], trial_window[1]])
trigger_text = triggers[0].replace('_', ' ')
style_event_distribution(g, 'Time (ms)', 'Trial number', trigger_text)

