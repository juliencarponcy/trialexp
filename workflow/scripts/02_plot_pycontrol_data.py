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
df2plot = df_events_cond[df_events_cond.name=='spout'].copy()
df2plot['trial_time'] = df2plot['trial_time']/1000
g = plot_event_distribution(df2plot, 'trial_time', 'trial_nb', xbinwidth=0.1, ybinwidth=0, xlim=[trial_window[0]/1000, trial_window[1]/1000])
trigger_text = triggers[0].replace('_', ' ')
style_event_distribution(g, 'Time (s)', 'Trial number', trigger_text)

# %% save
g.savefig(soutput.event_histogram, dpi=300)

# %% Compare between two states
# 
state1 = 'busy_win'
state2 = 'short_break'
event_name = 'spout'
event_time = df_events_cond[df_events_cond['name']==event_name].time.values
df1 = df_events_cond[df_events_cond['name']==state1]

# %%

# # Assume everything is sorted in ascending order according to time
# def filter_event_in_state(event_name, state_name, df_events):
#     event_time = df_events_cond[df_events_cond['name']==event_name]
#     df1 = df_events_cond[df_events_cond['name']==state_name]
    
#     sel_events = []
#     cur_state_idx = 0
#     start = df1.iloc[cur_state_idx].time
#     end = start + df1.iloc[cur_state_idx].duration

#     for i in range(len(event_time)):
#         # Keep moving the state window forward if it is too early
#         while event_time.iloc[i].time>end and cur_state_idx< (len(df1)-2):
#             cur_state_idx += 1
#             start = df1.iloc[cur_state_idx].time
#             end = start + df1.iloc[cur_state_idx].duration
            
#             if event_time.iloc[i].time>start and event_time.iloc[i].time<end:
#                 sel_events.append(i)

      
#     return event_time.iloc[sel_events]

# total_duration = {}
# spout_in_busy = filter_event_in_state('spout', 'busy_win', df_events_cond)
# total_duration['busy_win'] = df_events_cond[df_events_cond['name']=='busy_win'].duration.sum()
# spout_in_busy['state'] = 'busy_win'
# spout_in_break = filter_event_in_state('spout', 'short_break', df_events_cond)
# total_duration['busy_win'] = df_events_cond[df_events_cond['name']=='busy_win'].duration.sum()
# spout_in_busy['state'] = 'short_break'
# # %%
# spout_in_busy
# # %%
