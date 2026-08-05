'''
Script to create the session folder structure
'''
#%%
import os
from pathlib import Path
import numpy as np
import pandas as pd
from snakehelper.SnakeIOHelper import getSnake
from trialexp import config
from matplotlib import pyplot as plt
import seaborn as sns

#%% Load inputs


(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [config.debug_folder + r'/processed/cell_metrics_aggregation.done'],
  'cell_metrics_aggregation')


# %% Load Metadata and folders

sorter_name = 'kilosort3'
verbose = True

root_path = Path(config.SESSION_ROOT_DIR)
# Where to store globally computed figures
clusters_figure_path = Path(os.environ['CLUSTERS_FIGURES_PATH'])
# where to store global processed data
clusters_data_path = Path(os.environ['PROCCESSED_CLUSTERS_PATH'])
# list all cell metrics dataframes processed so far.
cell_metrics_paths = list(root_path.glob(f'*/*/processed/{sorter_name}/*/sorter_output/cell_metrics_df_full.pkl'))

# list all raw_waveforms numpy arrays processed so far.
raw_waveforms_paths = list(root_path.glob(f'*/*/processed/{sorter_name}/*/sorter_output/all_raw_waveforms.npy'))

# Check if cell_metrics and raw_waveforms paths have the same length 
# TODO potentially identify culprit file(s) by disimilarity of cell_metrics_paths and raw_waveforms_paths
assert len(cell_metrics_paths) == len(raw_waveforms_paths), f"cell_metrics_paths and raw_waveforms_paths dont have the same length are: {len(cell_metrics_paths)}, {len(raw_waveforms_paths)}"

if len(cell_metrics_paths) == 0:
   raise FileNotFoundError('There were no results of cell metrics found, check your paths / or data')
# %% Aggregate CellExplorer Cell metrics

# Define empty DataFrame for global cell_metrics aggreation (all recordings)  
aggregate_cell_metrics_df = pd.DataFrame()
# delete potential previous global waveforms np.array (debugging)
if 'aggregate_raw_waveforms' in locals():
   del aggregate_raw_waveforms # for debugging purpose

# Define empty np.ndarray for global raw_waveforms aggregation (all recordings)
for path_idx, cell_metrics_path in enumerate(cell_metrics_paths):

    # Load a single-session/probe cell_metrics dataframe
    cell_metrics_df = pd.read_pickle(cell_metrics_path)
    
    # add current cell_metrics to global DataFrame (all recordings)
    if aggregate_cell_metrics_df.empty:
      aggregate_cell_metrics_df = cell_metrics_df
    else:
      aggregate_cell_metrics_df = pd.concat([aggregate_cell_metrics_df, cell_metrics_df], axis=0)

    # Load a single-session/probe raw waverforms array
    if 'aggregate_raw_waveforms' not in locals():
      aggregate_raw_waveforms = np.load(raw_waveforms_paths[path_idx])
    else:
      # concatenate along cluster 3rd dimension in a 3D array
      aggregate_raw_waveforms = np.dstack((aggregate_raw_waveforms, np.load(raw_waveforms_paths[path_idx])))
    
# %%
# Reset index to use new index equivalent to raw waveforms np.ndarray index
aggregate_cell_metrics_df = aggregate_cell_metrics_df.reset_index()
# # Remove clusters with too few spikes
# aggregate_cell_metrics_df = aggregate_cell_metrics_df[aggregate_cell_metrics_df.spikeCount > min_spike_count]

# Define variables meaningless for clustering
invalid_cols_for_clustering = ['UID','animal', 'brainRegion','cellID', 'cluID',
                                'electrodeGroup', 'labels', 'maxWaveformCh', 'maxWaveformCh1', 
                                'maxWaveformChannelOrder', 'putativeCellType', 'sessionName', 
                                'shankID', 'synapticConnectionsIn', 'synapticConnectionsOut', 
                                'synapticEffect', 'thetaModulationIndex', 'total', 'trilat_x', 
                                'trilat_y', 'deepSuperficial', 'deepSuperficialDistance',
                                'spikeCount', 'burstIndex_Doublets', # because lot of absent values
                                'subject_ID',	'datetime', 'task_folder', 'probe_name']


clustering_cols = [col for col in aggregate_cell_metrics_df.columns if col not in invalid_cols_for_clustering]

# turn aggregate_cell_metrics_df into np.ndarray with only columns useful for clustering
aggregate_cell_metrics_array = aggregate_cell_metrics_df[clustering_cols].values

aggregate_cell_metrics_df_clustering = aggregate_cell_metrics_df[clustering_cols]

# Compute stats about the dataset
cell_metrics_stats = aggregate_cell_metrics_df_clustering.describe().T
# In the stats find non-full columns
non_full_cols = cell_metrics_stats[
   (cell_metrics_stats['count'].values != cell_metrics_stats['count'].values.max())
   ].index.tolist()

# remove non-full columns
aggregate_cell_metrics_df_clustering = aggregate_cell_metrics_df_clustering[
   [col for col in aggregate_cell_metrics_df_clustering.columns if col not in non_full_cols]
  ]

# %% Visualize log normalized distribution of the different clustering variables
for col in aggregate_cell_metrics_df_clustering.columns.values:
  ditrib_fig = plt.figure(figsize=(5,5))
  log_distrib = sns.kdeplot(aggregate_cell_metrics_df_clustering[col], levels=4, color=".2", log_scale=True)
  plt.show()
  ditrib_fig.savefig(clusters_figure_path / str('cell_metrics_log_distrib_' + col + '.png'))

# %% Plot exploratory cross correlation between cell metrics
sns.set_style("white")
display_cols = [ 'cv2',
                'firingRate',
                'acg_refrac',
                'troughToPeak',
                'std_std_waveform',
                'peak_average_all_wf',
                ]
fig = plt.figure(figsize=(15,10))
pair_plot_metrics = sns.pairplot(aggregate_cell_metrics_df[display_cols], diag_kind="kde")#, hue="peak_average_all_wf")
pair_plot_metrics.map_lower(sns.kdeplot, levels=4, color=".2")
pair_plot_metrics.map_upper(sns.histplot)
# %% Set details of figure and save

# set the figure title
pair_plot_metrics.fig.suptitle("2D distribution of example cell metrics")

# save the figure
pair_plot_metrics.savefig(clusters_figure_path / 'cell_metrics_example_pairplot.png', dpi=300)


# %% Save global cell_metrics DataFrame

# DataFrame version with only the variables used for clustering
aggregate_cell_metrics_df_clustering.to_pickle(clusters_data_path / 'aggregate_cell_metrics_df_clustering.pkl')
# Save a full version of the cell-metrics dataframe  
aggregate_cell_metrics_df.to_pickle(clusters_data_path / 'aggregate_cell_metrics_df_full.pkl')

# Save all aggregated raw waveforms
np.save(clusters_data_path / 'all_raw_waveforms.npy', aggregate_raw_waveforms)

# %%
