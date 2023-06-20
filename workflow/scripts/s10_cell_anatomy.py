'''
Script to fetch anatomy .csv file to infer brain structure of each cluster
'''
#%%
import os
from pathlib import Path

import pandas as pd
import numpy as np
import xarray as xr
from snakehelper.SnakeIOHelper import getSnake


from workflow.scripts import settings

#%% Load inputs


(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [settings.debug_folder + r'/processed/anatomy.done'],
  'cell_anatomy')


#%% To move to package

def parse_cell_UID(UID: pd.Index) -> pd.DataFrame:
    '''
    This function takes a pandas Index as UIDs for cells / neurons / clusters
    and returns a DataFrame with session_ID, probe_name and cell_ID
    '''
    # create new DataFrame from index (cell UIDs)
    df = pd.DataFrame(UID)
    # parse UIDs into their following components
    df[['session_ID','probe_name','cell_ID']] = pd.DataFrame(df['UID'].apply(lambda x: x.split('_')).to_list())

    return df

# %% Load Metadata and folders

verbose = True
tip_to_first_contact_um = 150 # distance from the tip of the Neuropixel to the first active contact


anatomy_folders = list(Path(os.environ['ANATOMY_ROOT_DIR']).glob('*/'))
ks3_folder = Path(sinput.xr_spikes_trials).parent / 'si' / 'kilosort3' 
probe_names = [x.stem for x in ks3_folder.iterdir() if x.is_dir()]

xr_spikes_trials_path = Path(sinput.xr_spikes_trials)
xr_spikes_trials_phases_path = Path(sinput.xr_spikes_trials_phases)
xr_spikes_session_path = Path(sinput.xr_spikes_session)

xr_spikes_trials_phases = xr.open_dataset(xr_spikes_trials_phases_path)
xr_spikes_session = xr.open_dataset(xr_spikes_session_path)


session_path = xr_spikes_session_path.parent.parent
session_ID = session_path.stem

borders_csv_path  = [anatomy_folder / 'borders_table.csv' for anatomy_folder in anatomy_folders if session_ID.split('-', 1)[0] in str(anatomy_folder.stem)]
probes_csv_path  = [anatomy_folder / 'T_probes.csv' for anatomy_folder in anatomy_folders if session_ID.split('-', 1)[0] in str(anatomy_folder.stem)]

if len(borders_csv_path) > 1 or len(probes_csv_path) > 1:
    raise ValueError(f'Several borders or probes file paths match the session {session_ID}:\n {borders_csv_path}')
elif len(borders_csv_path) == 0 or len(probes_csv_path) == 0:
    raise ValueError(f'No borders or probes file path match the session {session_ID}')
else:
    borders_csv_path = borders_csv_path[0]
    probes_csv_path = probes_csv_path[0]
#%% Loading anatomy information

borders_df = pd.read_csv(borders_csv_path)
borders_df = borders_df[borders_df.session_id == session_ID]
# 'upward_from_tim_um' column should be in the T_probe file instead
# TODO request change to Kouichi and implement from T_probes.csv (probes_df)
probes_df = pd.read_csv(probes_csv_path)
probes_df = probes_df[probes_df.session_id == session_ID]

# make a dataframe with session_ID, probe_name and cell_ID from the xarray UIDs
UID_df = parse_cell_UID(xr_spikes_trials_phases.UID.to_index())
# create DataArray for anatomical depth of cluster
xr_spikes_trials_phases['anat_depth'] = xr.DataArray(
    xr_spikes_trials_phases.y.values, 
    name =  'anat_depth', 
    dims={'UID'})

for probe_name in probe_names:
    probe_anat = probes_df[[probe_anat in probe_name for probe_anat in probes_df.probe_AB.to_list()]]
    probe_borders = borders_df[borders_df.probe_AB.str.contains(probe_name[-1])]

    # get tip depth
    tip_depth_from_anatomy_um = probe_anat.depth_from_anatomy_um.values[0]
    # compute values only for current probe
    # substract also tip_to_first_contact_um and
    probe_mask = UID_df['probe_name'].str.contains(probe_name).values 
    # computation of depth from tip depth and tip_to_first_contact_um (triangular [contactless] tip of neuropixels depth)
    xr_spikes_trials_phases['anat_depth'][probe_mask] = \
        tip_depth_from_anatomy_um - tip_to_first_contact_um - xr_spikes_trials_phases['y'][probe_mask]

    for struct_ID in probe_borders.index:
        struct_mask = ((xr_spikes_trials_phases['anat_depth'] > probe_borders.upperBorder_um[struct_ID]) 
                       & (xr_spikes_trials_phases['anat_depth'] <= probe_borders.lowerBorder_um[struct_ID]) 
                       & (probe_mask)).values
        xr_spikes_trials_phases.brainRegion[struct_mask] = probe_borders.acronym[struct_ID]
        xr_spikes_trials_phases['brainRegion_acr'] = xr_spikes_trials_phases.brainRegion.copy()
        xr_spikes_trials_phases.brainRegion.to_dataframe().apply(lambda x: probe_borders.name[probe_borders.acronym == x])
xr_spikes_trials_phases['anat_depth'].to_dataframe().hist()    
# %%
