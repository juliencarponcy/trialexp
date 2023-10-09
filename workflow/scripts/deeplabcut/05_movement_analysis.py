'''
This script analyze the photometry response for different movement types
'''

#%%
import trialexp.process.deeplabcut.utils as dlc_utils
from snakehelper.SnakeIOHelper import getSnake
from workflow.scripts import settings
import numpy as np
from pathlib import Path
import pandas as pd
import xarray as xr
from moviepy.video.io.bindings import mplfig_to_npimage
import matplotlib.pylab as plt
from moviepy.editor import *
import seaborn as sns

(sinput, soutput) = getSnake(locals(), 'workflow/deeplabcut.smk',
  [settings.debug_folder + '/processed/deeplabcut/df_init_data.pkl'],
  'analyze_movement')

# %% Load data
xr_session = xr.load_dataset(sinput.xr_dlc)

# %% Calculate movement data
wrist_loc = xr_session['dlc_markers'].loc[:,'wrist',['x','y','likelihood']]
df_move = dlc_utils.dlc2movementdf(xr_session, wrist_loc)
valid_init, valid_init_time = dlc_utils.get_valid_init(df_move)
df_init_data = dlc_utils.extract_triggered_data(valid_init_time, xr_session, [-1000, 1500],
                                             sampling_rate=100)

# %% Extract video for checking
video_path = Path(os.environ['VIDEO_DIR'])
videofile = str(video_path/xr_session.attrs['side_cam'])

dlc_utils.extract_sample_video_multi(videofile, 'move_int', soutput.move_init_video, 
                                     valid_init_time[:5].values, video_type='mp4', resize_ratio=0.5)
# %% Analyze movement types
direction = dlc_utils.get_direction(df_move, valid_init)
speed = dlc_utils.get_average_speed(df_move, valid_init)
accel = dlc_utils.get_average_value(df_move,'accel', valid_init)
x = dlc_utils.get_average_value(df_move,'x', valid_init, win_dir='before')
y = dlc_utils.get_average_value(df_move,'y', valid_init, win_dir='before')
likelihood = dlc_utils.get_average_value(df_move,'likelihood', valid_init, win_dir='before')

mov_type = dlc_utils.get_movement_type(df_move, valid_init, 50, window=50)
speed_cls = pd.qcut(speed,3, labels=['slow','middle','fast'])
average_photom = dlc_utils.get_average_photom(df_move, valid_init)

df_init_type =  pd.DataFrame({'event_index':np.arange(len(direction)), 
                              'init_time': valid_init_time,
                              'speed': speed,
                              'x': x,
                              'y': y,
                              'likelihood': likelihood,
                              'accel': accel,
                              'speed_class': speed_cls,
                              'move_type': mov_type,
                              'df/f': average_photom,
                              'direction':direction})

df_init_data = df_init_data.merge(df_init_type, on='event_index')

df_init_type.to_pickle(soutput.df_init_type)
df_init_data.to_pickle(soutput.df_init_data)

# %% Plot figures
df2plot = df_init_data[df_init_data.likelihood>0.7] # only plot reliable data
sns.set_context('paper',font_scale=1.5)
g = sns.relplot(df2plot, kind='line', x='event_time',
                y='photometry', hue='direction', col='move_type')
g.set_xlabels('Time (ms)')
g.set_ylabels('zscore dF/F')
plt.savefig(soutput.movement_figure,dpi=200)


# %%
