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



#%% Load data
df_pycontrol = pd.read_pickle(sinput.pycontrol_dataframe)


#%%
#TODO how do I get photometry_rsync.pulse_times_A and photometry_rsync.pulse_times_B???
photometry_aligner = Rsync_aligner(exp_cohort.sessions[i].photometry_rsync.pulse_times_A,
                                   exp_cohort.sessions[i].photometry_rsync.pulse_times_B,
                                   chunk_size=5, plot=False, raise_exception=True)

photometry_times_pyc = photometry_aligner.B_to_A(data_photometry['time'])

#TODO I should hack the code of get_photometry_trials and process data similarly but for a session as well

#remove all state change event
df_pycontrol = df_pycontrol.dropna(subset='name')
df2plot = df_pycontrol[df_pycontrol.type != 'state']
keys = df2plot.name.unique()

export_session(df_pycontrol, keys,
             smrx_filename=soutput.spike2_file)
# %%
import xarray as xr
xr_photo = xr.load_dataset(sinput.df_photometry)
