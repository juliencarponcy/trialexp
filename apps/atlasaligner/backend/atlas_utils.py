#%%
import fastapi
from enum import Enum
from typing import Optional
from trialexp.process.folder_org.utils import build_session_info_cohort
from trialexp.process.anatomy.utils import load_ccf_data, shift_trajectory_depth, get_region_boundaries, trajectory2probe_coords
from fastapi.middleware.cors import CORSMiddleware
from pandas import Timestamp
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from datetime import datetime
#%%

root_path = '/mnt/Magill_Lab/Julien/ASAP/Data'
df_session_info = build_session_info_cohort(root_path, load_pycontrol=False)
# %%
df_sesssion_info = df_session_info.query('neuropixels_sorted==True')
#%%
df_sel = df_session_info.query('session_id=="TT013-2024-10-24-142256"').iloc[0]
# %%
# match with the track localization results
local_results = Path(root_path)/df_sel.cohort/'histology'/df_sel.animal_id/'RGB'/'Processed'
# %%

probe_ccf = sio.loadmat(str(local_results/'probe_ccf.mat'), simplify_cells=True)['probe_ccf']

#%% Match the date of the experimentn with the probe date

with open(local_results/'probe_names.txt') as f:
    probe_names = f.readlines()
    probes = []
    for p in probe_names:
        try: 
            probes.append(datetime.strptime(p.strip(), '%d/%m/%Y').date())
        except:
            probes.append(p)

probes_name = np.array(probes)


# # match the experiment datetime
probe_idx = np.where(probes_name == df_sel.expt_datetime.date())[0][0]
probe_idx


# %%
# trajectory coords is in ap, dv, ml
# coords is [start, end] in axis coordinate
# ref see https://community.brain-map.org/t/how-to-transform-ccf-x-y-z-coordinates-into-stereotactic-coordinates/1858

channel_position =np.load(df_sel.path/'processed/kilosort4/ProbeA/channel_positions.npy')
channel_map =np.load(df_sel.path/'processed/kilosort4/ProbeA/channel_map.npy')

probe_coords = trajectory2probe_coords(probe_ccf[probe_idx], channel_position)


#%% atlas
atlas, structure_tree = load_ccf_data(Path('/mnt/Magill_Lab/Julien/ASAP/software/allenccf'))


# %%
shifted_coords1 = shift_trajectory_depth(probe_coords, -619)
trajectory_areas = get_region_boundaries(shifted_coords1, atlas, structure_tree)
trajectory_areas = trajectory_areas.sort_values('depth_start')
trajectory_areas[['acronym','depth_start','depth_end']]
# %% Plot the fiing rate map

df_quality_metrics = pd.read_pickle(df_sel.path/'processed/df_quality_metrics.pkl')

#%%
df_fr = df_quality_metrics.groupby('ks_chan_pos_y')[['firing_rate']].mean().reset_index()
df_fr.to_dict(orient='list')

#%%
bin_size = 200
bins = np.arange(0,3840+bin_size,60)
df_fr['pos_y_bin'] = pd.cut(df_fr.ks_chan_pos_y,bins)

df_binned = df_fr.groupby('pos_y_bin')['firing_rate'].agg(['mean','count']).reset_index()
df_binned['pos_y_bin'] = bins[:-1]
df_binned
#%% check spike locations

spike_positions =np.load(df_sel.path/'processed/kilosort4/ProbeA/spike_positions.npy')
spike_clusters = np.load(df_sel.path/'processed/kilosort4/ProbeA/spike_clusters.npy')
idx = (spike_clusters==421)
plt.plot(spike_positions[idx,0], spike_positions[idx,1],'.')
# %%
df_quality_metrics[df_quality_metrics.cluID=='TT013-2024-10-24-142256_ProbeA_421']
# check TT012-2024-11-27-160737, too much activity in the internal capsule
# %%
