#%%
from trialexp import config
from snakehelper.SnakeIOHelper import getSnake
import deeplabcut
import numpy as np
from pathlib import Path
import shutil
import glob
import os
import trialexp.process.deeplabcut.utils as dlc_utils
#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflow/deeplabcut.smk',
  [config.debug_folder + '/processed/dlc_results.h5'],
  'analyze_video')
# %%
path_config_file = os.environ['DLC_CONFIG_PATH']
video_path = Path(os.environ['VIDEO_DIR'])

#%% read the video files
filelist = np.loadtxt(sinput.video_list,dtype=str)
side_cam = [str(video_path/(f+'.mp4')) for f in filelist if 'Side' in f]

#%%
video = dlc_utils.rescale_video(side_cam[0], output_path=soutput.side_video, width=640, frame_rate=100)

