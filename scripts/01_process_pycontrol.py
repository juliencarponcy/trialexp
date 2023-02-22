
#%%
import pandas as pd
import seaborn as sns
import numpy as np
from trialexp.process.pycontrol.session_analysis import *
from trialexp.process.pycontrol.utils import *
from trialexp.process.pycontrol.plot_utils import *
from trialexp.process.pycontrol.session_analysis import Session
import sys
from datetime import datetime
from snakehelper.SnakeIOHelper import getSnake
try:
    sys.path.append(r'../../pyControl')
    import tools.data_import as di
except ModuleNotFoundError:
    print('pyControl must be in the Python search path')
    
#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflows/spout_bar_nov22.smk',
  ['Z:/Teris/ASAP/pycontrol/reaching_go_spout_bar_nov22/processed/kms058-2023-02-20-120453.pkl'],'process_pycontrol')

#%% Read task definition

basefolder = r'C:\code\trialexp'
tasks = pd.read_csv(basefolder+'\\params\\tasks_params.csv', usecols=[1, 2, 3, 4], index_col=False)
task_name = 'reaching_go_spout_bar_nov22'

trial_window = [-2000, 4000]
timelim = [1000, 4000] # in ms

conditions, triggers, events_to_process = get_task_specs(tasks,  task_name)


#%% Read pycontrol file

filename = sinput.pycontrol_file

df_session = di.session_dataframe(filename)
df_pycontrol = parse_session_dataframe(df_session, conditions)

session_time = datetime.strptime(df_pycontrol.attrs['Start date'], '%Y/%m/%d %H:%M:%S')
subjectID = df_pycontrol.attrs['Subject ID']


#%% Extract trial-related information from events
df_events_trials, df_events = extract_trial_by_trigger(df_pycontrol, triggers[0], conditions+events_to_process, 
                                            trial_window, subjectID, session_time)

df_conditions = compute_conditions_by_trial(df_events_trials, conditions)

df_conditions = compute_success(df_events_trials, df_conditions, task_name)

#%%  Merge condition back with event dataframe
df_events_cond = df_events.merge(df_conditions, on='trial_nb')

#%% Add in all the meta information
df_events_cond.attrs.update(df_events.attrs)
df_events_cond.attrs.update(
    {'conditions': conditions,
     'triggers': triggers,
     'events_to_process': events_to_process}
)

# %% save
df_events_cond.to_pickle(soutput.event_dataframe)

