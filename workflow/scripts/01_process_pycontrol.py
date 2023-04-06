
#%%
import pandas as pd
import seaborn as sns
import numpy as np
from trialexp.process.pycontrol.session_analysis import *
from trialexp.process.pycontrol.utils import *
from trialexp.process.pycontrol.plot_utils import *
from trialexp.process.pycontrol.session_analysis import Session # do not look used
from trialexp.process.pycontrol.data_import import session_dataframe
from datetime import datetime
from snakehelper.SnakeIOHelper import getSnake
from pathlib import Path
from workflows.scripts import settings
import os 

#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflows/spout_bar_nov22.smk',
  [os.path.join(settings.debug_folder,'processed','df_events_cond.pkl')],
  'process_pycontrol')

#%% Read pycontrol file

filename = list(Path(sinput.session_path, 'pycontrol').glob('*.txt'))
if len(filename)>1:
    raise ValueError('There are more than one pycontrol file there', filename)

df_session = session_dataframe(filename[0])
df_pycontrol = parse_session_dataframe(df_session)
session_time = datetime.strptime(df_pycontrol.attrs['Start date'], '%Y/%m/%d %H:%M:%S')
subjectID = df_pycontrol.attrs['Subject ID']
task_name = df_pycontrol.attrs['Task name']
session_id = Path(sinput.session_path).name

df_pycontrol.attrs['session_id'] = session_id
df_pycontrol.to_pickle(soutput.pycontrol_dataframe)

#%% Read task definition
tasks = pd.read_csv('params/tasks_params.csv', usecols=[1, 2, 3, 4], index_col=False)
trial_window = settings.trial_window
timelim = settings.timelim

conditions, triggers, events_to_process = get_task_specs(tasks,  task_name)

#%% Extract trial-related information from events
df_pycontrol = df_pycontrol[~(df_pycontrol.name=='rsync')] #remove the sync pulse
df_pycontrol  = print2event(df_pycontrol, conditions)

df_events_trials, df_events = extract_trial_by_trigger(df_pycontrol, triggers[0], 
                                                       conditions+events_to_process+triggers, 
                                            trial_window, subjectID, session_time)

df_conditions = compute_conditions_by_trial(df_events_trials, conditions)

df_conditions = compute_success(df_events_trials, df_conditions, task_name, 
                                triggers, timelim)

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
df_conditions.to_pickle(soutput.condition_dataframe)
df_events_trials.to_pickle(soutput.trial_dataframe)

# %%
