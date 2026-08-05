#%%
import pandas as pd 
from trialexp.process.pycontrol.utils import extract_v_line_v2, export_session_v2, get_sync_time
from snakehelper.SnakeIOHelper import getSnake
from trialexp import config
from re import match
from pathlib import Path
from trialexp.process.pyphotometry.utils import *
from trialexp.process.pycontrol.utils import add_events_to_time_series
import os
import xarray as xr 
#%%

(sinput, soutput) = getSnake(locals(), 'workflow/pycontrol.smk',
    [config.debug_folder +'/processed/photometry.parquet'],
    'export_parquet')

# %% Convert photometry data 
xr_photom = xr.load_dataset(sinput.xr_photometry)
var2extract = [v for v in xr_photom.data_vars if 'time' in xr_photom[v].coords]
df = xr_photom[var2extract].to_dataframe().reset_index()
df = df.fillna(0)
# %%
# convert the pycontrol data to time series
df_pycontrol = pd.read_pickle(sinput.pycontrol_dataframe)
df = add_events_to_time_series(df_pycontrol, df)

#%% save to parquet to be read by plotJuggler
df.to_parquet(soutput.photometry_parquet, compression="gzip")
# %%
