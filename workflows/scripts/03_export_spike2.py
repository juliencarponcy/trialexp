'''
Export event data to spike2
'''
#%%
import pandas as pd 
from trialexp.process.pycontrol.utils import export_session
from snakehelper.SnakeIOHelper import getSnake


#%%

(sinput, soutput) = getSnake(locals(), 'workflows/spout_bar_nov22.smk',
#   ['Z:/Julien/Data/head-fixed/_Other/test_folder/by_session_folder/JC316L-2022-12-07-163252\processed/spike2.smrx'],
    ['Z:\Teris\ASAP\expt_sessions\kms064-2023-02-08-100449\processed\spike2.smrx'],
  'export_spike2')

#%% Load data
df_events = pd.read_pickle(sinput.event_dataframe)

#%%
# state_def = [{'name': 'hold_for_water', 
#                   'onset': 'hold_for_water', 'offset': 'waiting_for_spout'},
#             {'name': 'waiting_for_spout', 
#                  'onset': 'waiting_for_spout', 'offset': 'busy_win'},
#             {'name': 'busy_win', 
#                  'onset': 'busy_win', 'offset': 'break_after_water'},
#             {'name': 'break_after_water', 
#                  'onset': 'break_after_water',    'offset': 'waiting_for_bar'},
#             {'name': 'break_after_no_water', 
#                  'onset': 'break_after_no_water', 'offset': 'waiting_for_bar'}]

#TODO automatically plot all states
state_def = [
            {'name': 'hold_for_water', 
                 'onset': 'hold_for_water', 'offset': 'waiting_for_spout'},
            {'name': 'waiting_for_spout', 
                 'onset': 'waiting_for_spout', 'offset': 'break_after_trial'}]


#remove all state change event
df2plot = df_events[df_events.type != 'state']
keys = df2plot.name.unique()

export_session(df_events, keys, state_def,
             smrx_filename='test.smrx',verbose=False)
# %%
