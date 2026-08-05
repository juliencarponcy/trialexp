'''
Merge localizatoin data if found
'''
#%%
import os, sys
from pathlib import Path
from pprint import pprint 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from snakehelper.SnakeIOHelper import getSnake
from trialexp import config
import xarray as xr
from glob import glob
from trialexp.process.anatomy.utils import construct_localization_dataset
from loguru import logger
# %%
(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [config.debug_folder + r'/processed/xr_localization.nc'],
  'localization')

#%%
xr_spikes_trials = xr.open_dataset(sinput.xr_spikes_trials)

session_id = xr_spikes_trials.attrs['session_id']
s = session_id.split('-')
animal_id = s[0]
date = '-'.join(s[1:4])


# %% Search for the localization file
root_path = config.SESSION_ROOT_DIR
localization_files = glob(f'{root_path}/*/histology/{animal_id}/RGB/Processed/aligned_trajectory_{date}.pkl')
if len(localization_files)> 0:
    df_local = pd.read_pickle(localization_files[0])
    
    '''
    Allen CCF coordinates:
    https://community.brain-map.org/t/how-to-transform-ccf-x-y-z-coordinates-into-stereotactic-coordinates/1858

    With the nose pointing foward.
    The origin is at upper left corner at the back
    First axis is left-right
    Second axis is top-down
    Third axis is back-front
    Note: the coordinates do not start at Bregma, and everything are positive only.
    '''
    xr_local = construct_localization_dataset(df_local, xr_spikes_trials)
    xr_local.to_netcdf(soutput.xr_local)
else:
    logger.warning(f'Cannot find the localization file for {animal_id} on {date}')
    Path(soutput.xr_local).touch()

