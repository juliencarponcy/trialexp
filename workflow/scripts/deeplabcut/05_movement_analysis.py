'''
This script analyze the photometry response for different movement types
'''

#%%
import trialexp.process.deeplabcut.utils as dlc_utils
from snakehelper.SnakeIOHelper import getSnake
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

# %%
xr_session = xr.open_dataset(sinput.xr_dlc)
wrist_loc = xr_session['dlc_markers'].loc[:,'wrist',['x','y']]

def dlc2movementdf(marker_loc):
    # convert marker location to speed and acceleration data

    signal_time, coords, speed, accel = dlc_utils.get_movement_metrics(marker_loc)
    speed_mag = np.linalg.norm(speed,axis=1)
    accel_mag = np.diff(speed_mag, prepend=speed_mag[0])

    f = xr_session.zscored_df_over_f.data[0]

    df_move = pd.DataFrame({
        'accel': accel_mag,
        'accel_x': accel[:,0],
        'accel_y': accel[:,1],
        'speed': speed_mag,
        'speed_x': speed[:,0],
        'speed_y': speed[:,1],
        'x' : coords[:,0],
        'y' : coords[:,1],
        'time': xr_session.time,
        'df/f': f})
    
    is_moving = (df_move.speed>5)
    is_rest = ((df_move.speed<2) & (df_move.accel.abs()<3)).astype(np.int8)
    df_move['is_rest'] = is_rest
        
    return  df_move

df_move = dlc2movementdf(wrist_loc)
# %%
def get_valid_init(df_move):
    # find the time for movement initiation

    move_init_idx = np.where(np.diff(df_move.is_rest, prepend=False)==-1)[0]
    valid_init = dlc_utils.filter_init(df_move, move_init_idx,50, 10)
    valid_init_time = df_move.iloc[valid_init].time
    
    return valid_init, valid_init_time

valid_init, valid_init_time = get_valid_init(df_move)
# %%
df_init_data = dlc_utils.extract_triggered_data(valid_init_time, xr_session, [-1000, 1500],
                                             sampling_rate=100)

df_init_data.to_pickle(soutput.df_init_data)
# %%
ax = sns.lineplot(df_init_data, x='event_time', y='photometry')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('zscore dF/F')
ax.axvline(0,ls='--',color='g')
plt.savefig('photom_move_init.png',dpi=200)

# %%
video_path = Path(os.environ['VIDEO_DIR'])
videofile = str(video_path/xr_session.attrs['side_cam'])

dlc_utils.extract_sample_video_multi(videofile, 'move_int', soutput.move_init_video, 
                                     valid_init_time[:5].values, video_type='mp4', resize_ratio=0.5)
# %%
direction = dlc_utils.get_direction(df_move, valid_init)
speed = dlc_utils.get_average_speed(df_move, valid_init)
accel = dlc_utils.get_average_value(df_move,'accel', valid_init)
x = dlc_utils.get_average_value(df_move,'x', valid_init, win_dir='before')
y = dlc_utils.get_average_value(df_move,'y', valid_init, win_dir='before')

mov_type = dlc_utils.get_movement_type(df_move, valid_init, 100, window=50)
speed_cls = pd.qcut(speed,3, labels=['slow','middle','fast'])
average_photom = dlc_utils.get_average_photom(df_move, valid_init)

df_init_type =  pd.DataFrame({'event_index':np.arange(len(direction)), 
                              'init_time': valid_init_time,
                              'speed': speed,
                              'x': x,
                              'y': y,
                              'accel': accel,
                              'speed_class': speed_cls,
                              'move_type': mov_type,
                              'df/f': average_photom,
                              'direction':direction})
# %%
df_init_type.to_pickle(soutput.df_init_type)

# %%
df_init2 = df_init_data.merge(df_init_type, on='event_index')
df_init2
# %%
df2plot = df_init2[df_init2.speed>10]

ax = sns.lineplot(df2plot, x='event_time', y='photometry', hue='direction')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('zscore dF/F')
ax.axvline(0,ls='--',color='g')
plt.savefig('photom_movement_direction.png',dpi=200)
# %%
