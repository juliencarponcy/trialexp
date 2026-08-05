'''
Script to fetch anatomy .csv file to infer brain structure of each cluster
'''
#%%
import os
from pathlib import Path

import pandas as pd
import numpy as np
import xarray as xr

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec

from snakehelper.SnakeIOHelper import getSnake


from trialexp import config
#%% Load inputs


(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [config.debug_folder + r'/processed/ephys_anatomy.done'],
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

# %% Define variables and folders

verbose = True
tip_to_first_contact_um = 150 # distance from the tip of the Neuropixel to the first active contact


anatomy_folders = list(Path(os.environ['ANATOMY_ROOT_DIR']).glob('*/'))
ks3_folder = Path(sinput.xr_spikes_trials).parent / 'si' / 'kilosort3' 
probe_names = [x.stem for x in ks3_folder.iterdir() if x.is_dir()]

xr_spikes_trials_path = Path(sinput.xr_spikes_trials)
xr_spikes_trials_phases_path = Path(sinput.xr_spikes_trials_phases)
xr_spikes_session_path = Path(sinput.xr_spikes_full_session)

figures_path = xr_spikes_trials_path.parent / 'figures' / 'ephys'
if not figures_path.exists():
    figures_path.mkdir()

session_path = xr_spikes_session_path.parent.parent
session_ID = session_path.stem

#%% Opening datasets
xr_spikes_trials_phases = xr.open_dataset(xr_spikes_trials_phases_path)

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

# rename DataArray for brain region acronym 
xr_spikes_trials_phases['brain_region_short']  = xr_spikes_trials_phases['brainRegion'].rename('brain_region_short')
# create new DataArray for brain region name 
xr_spikes_trials_phases['brain_region_long'] = xr_spikes_trials_phases.brainRegion.copy()
# removing old camel-back formatted variable
xr_spikes_trials_phases = xr_spikes_trials_phases.drop_vars(['brainRegion'])
# NOTE TODO package formatting of variables of the xrray into separate function

for probe_name in probe_names:
    probe_anat = probes_df[[probe_anat in probe_name for probe_anat in probes_df.probe_AB.to_list()]]
    # compute values only for current probe
    probe_mask = UID_df['probe_name'].str.contains(probe_name).values 

    # If no corresponding anatomy
    if probe_anat.empty:
        xr_spikes_trials_phases['anat_depth'][probe_mask] = np.nan
        xr_spikes_trials_phases['brain_region_short'][probe_mask] = np.nan
        xr_spikes_trials_phases['brain_region_long'][probe_mask] = np.nan

    # If anatomy found
    else:
        probe_borders = borders_df[borders_df.probe_AB.str.contains(probe_name[-1])]

        # get tip depth
        tip_depth_from_anatomy_um = probe_anat.depth_from_anatomy_um.values[0]
        # substract also tip_to_first_contact_um and

        # TODO THIS IS MOST LIKELY WRONG FOR COMPUTATION NEED TO WORK ON THIS WITH KOUICHI
        # computation of depth from tip depth and tip_to_first_contact_um (triangular [contactless] tip of neuropixels depth)
        xr_spikes_trials_phases['anat_depth'][probe_mask] = \
            tip_depth_from_anatomy_um - tip_to_first_contact_um - xr_spikes_trials_phases['y'][probe_mask]

        for struct_ID in probe_borders.index:
            struct_mask = ((xr_spikes_trials_phases['anat_depth'] > probe_borders.upperBorder_um[struct_ID]) 
                            & (xr_spikes_trials_phases['anat_depth'] <= probe_borders.lowerBorder_um[struct_ID]) 
                            & (probe_mask)).values

            if struct_mask.any():
                xr_spikes_trials_phases.brain_region_short[struct_mask] = probe_borders.acronym[struct_ID]
                xr_spikes_trials_phases.brain_region_long[struct_mask] = probe_borders.name[struct_ID]

#%% Plotting anatomy summary

# TODO: implement in an external function

structs, struct_count = np.unique(xr_spikes_trials_phases['brain_region_short'].values.astype(str), return_counts=True)

# create grid for different subplots
spec = gridspec.GridSpec(ncols=2, nrows=1,
                         width_ratios=[2, 1], wspace=0.1)

# create a figure
fig = plt.figure(figsize=(20,5))
# to change size of subplot's
# set height of each subplot as 8
# fig.set_figheight(8)
 
# set width of each subplot as 8
fig.set_figwidth
axes = list()
axes.append(fig.add_subplot(spec[0]))
plt.suptitle(f'Clusters anatomical distribution: {session_ID}')

sns.barplot(x=structs, y=struct_count, ax=axes[0])
axes[0].set_xlabel('Brain structure acronym')
axes[0].set_ylabel('Number of clusters')

axes.append(fig.add_subplot(spec[1]))

sns.histplot(data=xr_spikes_trials_phases[['probe_name_x','anat_depth']].to_dataframe(),x='anat_depth', hue='probe_name_x', kde=True)
axes[1].set_xlabel('Anatomical depth (micrometers)')
axes[1].set_ylabel('Number of clusters')

fig.savefig(figures_path / 'cluster_anat_distrib.png')
# %% Copy brain regions to other xarrays and save

xr_spikes_session = xr.open_dataset(xr_spikes_session_path, engine='h5netcdf')
xr_spikes_trials =  xr.open_dataset(xr_spikes_trials_path, engine='h5netcdf')

# copying anatomial variables to other xarrays
xr_spikes_session[['anat_depth', 'brain_region_short', 'brain_region_long']] = xr_spikes_trials_phases[['anat_depth', 'brain_region_short', 'brain_region_long']].copy(deep=True)
xr_spikes_trials[['anat_depth', 'brain_region_short', 'brain_region_long']] = xr_spikes_trials_phases[['anat_depth', 'brain_region_short', 'brain_region_long']].copy(deep=True)

# xr_spikes_trials_path = xr_spikes_trials_path.parent / 'xr_spikes_trials_anat.nc'
# xr_spikes_trials_phases_path = xr_spikes_trials_phases_path.parent / 'xr_spikes_trials_phases_anat.nc'
# xr_spikes_session_path = xr_spikes_session_path.parent / 'xr_spikes_full_session_anat.nc'

# xr_spikes_trials_path = xr_spikes_trials_path.parent / 'xr_spikes_trials_anat.nc'
# xr_spikes_trials_phases_path = xr_spikes_trials_phases_path.parent / 'xr_spikes_trials_phases_anat.nc'
# xr_spikes_session_path = xr_spikes_session_path.parent / 'xr_spikes_full_session_anat.nc'

#%% Saving new xarrays

# remove from disk previous version of the anat xarray
if xr_spikes_trials_path.exists():
    xr_spikes_trials_path.unlink() 
if xr_spikes_trials_phases_path.exists():
    xr_spikes_trials_phases_path.unlink() 
if xr_spikes_session_path.exists():
    xr_spikes_session_path.unlink() 

# writing updated xarrays versions
xr_spikes_session.to_netcdf(xr_spikes_session_path, engine='h5netcdf')
xr_spikes_trials.to_netcdf(xr_spikes_trials_path, mode='a', engine='h5netcdf')
xr_spikes_trials_phases.to_netcdf(xr_spikes_trials_phases_path, mode='a', engine='h5netcdf')

# closing xarrays
xr_spikes_session.close()
xr_spikes_trials.close()
xr_spikes_trials_phases.close()


# %% 
#%% Define trials of interest

# Plot the distrubtion of cluster in each brain regions
# trial_types_to_plot = ('water by spout')

# structs, struct_count = np.unique(xr_spikes_trials_phases['brain_region_short'].values.astype(str), return_counts=True)

# # create grid for different subplots
# spec = gridspec.GridSpec(ncols=2, nrows=1,
#                          width_ratios=[2, 1], wspace=0.1)

# # create a figure
# fig = plt.figure(figsize=(20,5))
# # to change size of subplot's
# # set height of each subplot as 8
# # fig.set_figheight(8)
 
# # set width of each subplot as 8
# fig.set_figwidth
# axes = list()
# axes.append(fig.add_subplot(spec[0]))
# plt.suptitle(f'Clusters anatomical distribution: {session_ID}')

# sns.barplot(x=structs, y=struct_count, ax=axes[0])
# axes[0].set_xlabel('Brain structure acronym')
# axes[0].set_ylabel('Number of clusters')

# axes.append(fig.add_subplot(spec[1]))

# sns.histplot(data=xr_spikes_trials_phases[['probe_name_x','anat_depth']].to_dataframe(),x='anat_depth', hue='probe_name_x', kde=True)
# axes[1].set_xlabel('Anatomical depth (micrometers)')
# axes[1].set_ylabel('Number of clusters')

# fig.savefig(figures_path / 'cluster_anat_distrib.png')

