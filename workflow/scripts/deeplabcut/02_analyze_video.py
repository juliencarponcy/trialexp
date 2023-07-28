#%%
from workflow.scripts import settings
from snakehelper.SnakeIOHelper import getSnake
import deeplabcut
import numpy as np
from pathlib import Path
import shutil
import glob
#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflow/deeplabcut.smk',
  [settings.debug_folder + '/processed/dlc_results.h5'],
  'analyze_video')
# %%
path_config_file = '/home/MRC.OX.AC.UK/ndcn1330/ettin/Teris/ASAP/deeplabcut/side_2_hands_newobj-julien-2022-08-26/config.yaml'
video_path = Path('/home/MRC.OX.AC.UK/ndcn1330/ettin/Julien/Data/head-fixed/videos')

#%% read the video files
filelist = np.loadtxt(sinput.video_list,dtype=str)
side_cam = [str(video_path/(f+'.mp4')) for f in filelist if 'Side' in f]
dlc_result = Path(soutput.dlc_result)

# %% analyze video
deeplabcut.analyze_videos(path_config_file, side_cam, gputouse=0, 
                          destfolder=dlc_result.parent)

# %% rename the DLC results to better work with snakemake
dlc_file = glob.glob(str(dlc_result.parent/'*.h5'))[0]
shutil.copy(dlc_file, soutput.dlc_result)
