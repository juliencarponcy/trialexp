'''
This script clean up the raw coordinates from deeplabcut analysis
'''

#%%
from trialexp.process.deeplabcut.utils import extract_video_timestamp, preprocess_dlc, filter_and_merge, copy_coords
from workflow.scripts import settings
from snakehelper.SnakeIOHelper import getSnake
import numpy as np
from pathlib import Path
import pandas as pd
import os

(sinput, soutput) = getSnake(locals(), 'workflow/deeplabcut.smk',
  [settings.debug_folder + '/processed/dlc_results_clean.pkl'],
  'dlc_preprocess')

# %% load data
df = pd.read_hdf(sinput.dlc_result)
df.columns = df.columns.droplevel(level=0)

# %% Get centroid of tip
tip_markers = ['tip II','tip III','tip IV','tip V']
dftip = df.loc[:, df.columns.get_level_values(0).isin(tip_markers)].copy()
dftip = filter_and_merge(dftip)

#  add the new marker in
copy_coords(df, dftip, 'tip')


#%% Do preprocessing
df_clean = preprocess_dlc(df)

#%% Load video timestamp information

ts = extract_video_timestamp(sinput.side_video)

df_clean['time'] = ts
df_clean = df_clean.set_index('time')
df_clean.attrs['side_cam'] = str(sinput.side_video)

# %% Save
df_clean.to_pickle(soutput.dlc_processed)

