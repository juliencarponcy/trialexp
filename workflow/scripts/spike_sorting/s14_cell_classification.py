#%%
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from snakehelper.SnakeIOHelper import getSnake
from trialexp import config
from kilosort.io import load_ops
from spikeinterface.core.template import Templates  
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import trialexp.process.ephys.plot_utils as ephys_plot 
#%% Load inputs


(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [config.debug_folder + r'/processed/df_celltypes.pkl'],
  'cell_classification')

#%%
df_quality_metrics = pd.read_pickle(sinput.df_quality_metrics)
    
#%% Cell type classification based on properties
def classify_celltype(cell):
    # Basd on Andy Peters 202
    cell_type = None

    if cell['peak_to_valley'] <= 0.4/1000:
        # Narrow spike: FSI or UIN
        if cell['long_isi_portion'] > 0.1:
            cell_type = 'UIN'
        else:
            # No long pause, almost tunic firing
            cell_type = 'FSI'
        
    else:
        if cell['post_spike_suppression_ms'] > 40:
            cell_type = 'TAN'
        else:
            cell_type = 'MSN'
    
    return cell_type

def classify_spike_origin(cell):
    # Based on Ye et al. 2025
    # Axon recording in Neuropixel 1.0 should be around 6%
    origin = None
    if cell['peak_to_valley'] <= 0.4/1000:
        # narrow spikes
        if cell['spatial_footprint_um'] <= 20:
            origin = 'axon' 
        else:
            origin = 'soma' # interneuron
    else:
        origin = 'soma'
        
    return origin
        
    

df_quality_metrics['cell_type'] = df_quality_metrics.apply(classify_celltype, axis=1)
df_quality_metrics['spike_origin'] = df_quality_metrics.apply(classify_spike_origin, axis=1)


#%% Save
df_cell_types = df_quality_metrics[['cell_type','spike_origin','unit_id','cluID']]
df_cell_types.to_pickle(soutput.df_cell_types)