'''
Export event data to spike2
'''
#%%
import pandas as pd 
from trialexp.process.pycontrol.utils import export_session
from snakehelper.SnakeIOHelper import getSnake
from workflows.scripts import settings
from glob import glob 
from trialexp.process.pyphotometry.utils import *

#%%

(sinput, soutput) = getSnake(locals(), 'workflows/spout_bar_nov22.smk',
    [settings.debug_folder +'\processed\spike2.smrx'],
  'export_spike2')

#%% Photometry dict

fn = glob(sinput.photometry_folder+'\*.ppd')[0]
data_photometry = import_ppd(fn)
data_photometry = denoise_filter(data_photometry)
data_photometry = motion_correction(data_photometry)
data_photometry = compute_df_over_f(data_photometry, low_pass_cutoff=0.001)

photo_time = data_photometry['pulse_times_2']
# no down-sampling here

#%% Load data
df_pycontrol = pd.read_pickle(sinput.pycontrol_dataframe)

pycontrol_time = df_pycontrol[df_pycontrol.name == 'rsync'].time


#%%
photometry_aligner = Rsync_aligner(pycontrol_time, photo_time)
photometry_times_pyc = photometry_aligner.B_to_A(photo_time)

#remove all state change event
df_pycontrol = df_pycontrol.dropna(subset='name')
df2plot = df_pycontrol[df_pycontrol.type != 'state']
keys = df2plot.name.unique()

#%%
export_session(df_pycontrol, keys, 
             data_photometry = data_photometry,
             photometry_times_pyc = photometry_times_pyc,
             smrx_filename=soutput.spike2_file)
#%%
import xarray as xr
xr_photo = xr.load_dataset(sinput.df_photometry)
