'''
Script to fetch anatomy .csv file to infer brain structure of each cluster
'''
#%%
import os
from pathlib import Path

import pandas as pd
import numpy as np
import xarray as xr
from snakehelper.SnakeIOHelper import getSnake


from workflow.scripts import settings

#%% Load inputs


(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [settings.debug_folder + r'/processed/anatomy.done'],
  'cell_anatomy')


# %% Load Metadata and folders

verbose = True

anatomy_folders = list(Path(os.environ['ANATOMY_ROOT_DIR']).glob('*/'))
xr_spikes_trials_path = Path(sinput.xr_spikes_trials)
xr_spikes_trials_phases_path = Path(sinput.xr_spikes_trials_phases)
xr_spikes_session_path = Path(sinput.xr_spikes_session)

xr_spikes_trials_phases = xr.open_dataset(xr_spikes_trials_phases_path)
xr_spikes_session = xr.open_dataset(xr_spikes_session_path)


session_path = xr_spikes_session_path.parent.parent
session_ID = session_path.stem

borders_csv_path  = [anatomy_folder / 'borders_table_all.csv' for anatomy_folder in anatomy_folders if session_ID.split('-', 1)[0] in str(anatomy_folder.stem)]
probes_csv_path  = [anatomy_folder / 'T_probes_all.csv' for anatomy_folder in anatomy_folders if session_ID.split('-', 1)[0] in str(anatomy_folder.stem)]

if len(borders_csv_path) > 1 or len(probes_csv_path) > 1:
    raise ValueError(f'Several borders or probes file paths match the session {session_ID}:\n {borders_csv_path}')
elif len(borders_csv_path) == 0 or len(probes_csv_path) == 0:
    raise ValueError(f'No borders or probes file path match the session {session_ID}')
else:
    borders_csv_path = borders_csv_path[0]
    probes_csv_path = probes_csv_path[0]
#%% Loading anatomy information

borders_df = pd.read_csv(borders_csv_path)
borders_df = borders_df[borders_df.session_id == session_ID]

probes_df = pd.read_csv(probes_csv_path)
probes_df = probes_df[probes_df.session_id == session_ID]

                        
tip_depth = borders_df.lowerBorder.max()

#%% 