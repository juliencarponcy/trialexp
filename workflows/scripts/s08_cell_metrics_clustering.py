'''
Script to create the session folder structure
'''
#%%
import os
from pathlib import Path
from itertools import cycle, islice

import numpy as np
import pandas as pd

from snakehelper.SnakeIOHelper import getSnake

from workflows.scripts import settings

from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn import cluster, mixture

from matplotlib import pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
#%% Load inputs


(sinput, soutput) = getSnake(locals(), 'workflows/spikesort.smk',
  [settings.debug_folder + r'/processed/cell_metrics_aggregation.done'],
  'cell_metrics_aggregation')


# %% Load Metadata and folders

sorter_name = 'kilosort3'
verbose = True

root_path = Path(os.environ['SESSION_ROOT_DIR'])
clusters_figure_path = Path(os.environ['CLUSTERS_FIGURES_PATH'])
# list all cell metrics dataframes processed so far.
cell_metrics_paths = list(root_path.glob(f'*/*/processed/{sorter_name}/*/sorter_output/cell_metrics_df_full.pkl'))

# list all raw_waveforms numpy arrays processed so far.
raw_waveforms_paths = list(root_path.glob(f'*/*/processed/{sorter_name}/*/sorter_output/all_raw_waveforms.npy'))

# Check if cell_metrics and raw_waveforms paths have the same length 
# TODO potentially identify culprit file(s) by disimilarity of cell_metrics_paths and raw_waveforms_paths
assert len(cell_metrics_paths) == len(raw_waveforms_paths), f"cell_metrics_paths and raw_waveforms_paths dont have the same length are: {len(cell_metrics_paths)}, {len(raw_waveforms_paths)}"

# %% Aggregate CellExplorer Cell metrics

# Define empty DataFrame for global cell_metrics aggreation (all recordings)  
aggregate_cell_metrics_df = pd.DataFrame()
# delete potential previous global waveforms np.array (debugging)
if 'aggregate_raw_waveforms' in locals():
   del aggregate_raw_waveforms

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
min_spike_count = 100
# Remove clusters with too few spikes
aggregate_cell_metrics_df = aggregate_cell_metrics_df[aggregate_cell_metrics_df.spikeCount > min_spike_count]

# Define variables meaningless for clustering
invalid_cols_for_clustering = ['UID','animal', 'brainRegion','cellID', 'cluID',
                                'electrodeGroup', 'labels', 'maxWaveformCh', 'maxWaveformCh1', 
                                'maxWaveformChannelOrder', 'putativeCellType', 'sessionName', 
                                'shankID', 'synapticConnectionsIn', 'synapticConnectionsOut', 
                                'synapticEffect', 'thetaModulationIndex', 'total', 'trilat_x', 
                                'trilat_y', 'deepSuperficial', 'deepSuperficialDistance',
                                'spikeCount',
                                'subject_ID',	'datetime', 'task_folder', 'probe_name']


clustering_cols = [col for col in aggregate_cell_metrics_df.columns if col not in invalid_cols_for_clustering]

# turn aggregate_cell_metrics_df into np.ndarray with only columns useful for clustering
aggregate_cell_metrics_array = aggregate_cell_metrics_df[clustering_cols].values

aggregate_cell_metrics_df_clustering = aggregate_cell_metrics_df[clustering_cols]

cell_metrics_stats = aggregate_cell_metrics_df_clustering.describe().T
non_full_cols = cell_metrics_stats[
   (cell_metrics_stats['count'].values != cell_metrics_stats['count'].values.max())
   ].index.tolist()

# remove non-full columns
aggregate_cell_metrics_df_clustering = aggregate_cell_metrics_df_clustering[
   [col for col in aggregate_cell_metrics_df_clustering.columns if col not in non_full_cols]
  ]
# %%

pd.set_option('display.max_columns', None)

# %% Params definition cell
color_column = 'peak_average_all_wf'
size_column = 'std_std_waveform'
# choose 2 raw values for scatter plot
raw_scatter_dims = ('peak_average_all_wf', 'std_std_waveform')

# Params for dimensionality reduction
dim_reduc_params = {
  'n_components' : 2,
  'perplexity' : 50,
  'random_state' : 33
}

# Params for clustering
cluster_params= {
  "n_clusters": 2,
  "eps": 0.3,
  "n_init": 10
}

# print cluster_UID as data point label
text_datapoints = aggregate_cell_metrics_df_clustering.index.values


# %% Dimensionality reduction functions definition

def compute_PCA(
        data: np.ndarray,
        n_components: int = 2,
        random_state: int = 3,
        standardize: bool = True
    ):
    pca = PCA(n_components, random_state=random_state)
    
    if standardize:
      scaler = StandardScaler()
      scaled_data = scaler.fit_transform(data)
      eigen_values = pca.fit_transform(scaled_data)

    else:
      eigen_values = pca.fit_transform(data)
    
    
    return eigen_values
    

def compute_tSNE(
      data: np.ndarray,
      n_components: int = 2,
      perplexity: int = 30,
      random_state: int = 33,
      standardize: bool = True
    ):
    scaler = StandardScaler()

    tsne = manifold.TSNE(
        n_components=n_components,
        init="random",
        random_state=random_state,
        perplexity=perplexity,
        n_iter=300,
    )
    if standardize:
      scaler = StandardScaler()
      scaled_data = scaler.fit_transform(data)
      Y = tsne.fit_transform(scaled_data)
    else:
      Y = tsne.fit_transform(data)

    return Y

# %% Dimensionality reduction computation
pca =  compute_PCA(aggregate_cell_metrics_df_clustering.values, 
                    dim_reduc_params['n_components'], 
                    dim_reduc_params['random_state'],
                    standardize = True)


tSNE = compute_tSNE(aggregate_cell_metrics_df_clustering.values, 
                    dim_reduc_params['n_components'],
                    dim_reduc_params['perplexity'],
                    dim_reduc_params['random_state'],
                    standardize = True)
 
tSNE_of_PCA = compute_tSNE(pca, 
                    dim_reduc_params['n_components'],
                    dim_reduc_params['perplexity'],
                    dim_reduc_params['random_state'],
                    standardize = False)
 
# %% Create cluster objects

# Initialize clustering algorithms with cluster_params
two_means = cluster.MiniBatchKMeans(
   n_clusters=cluster_params["n_clusters"], 
   n_init=cluster_params["n_init"])

spectral = cluster.SpectralClustering(
    n_clusters=cluster_params["n_clusters"],
    eigen_solver="arpack",
    affinity="nearest_neighbors",
)
dbscan = cluster.DBSCAN(eps=cluster_params["eps"])

gmm = mixture.GaussianMixture(
    n_components=cluster_params["n_clusters"], covariance_type="full"
)

# Define algorithms names
clustering_algorithms = (
    ("MiniBatch\nKMeans", two_means),
    ("Spectral\nClustering", spectral),
    ("DBSCAN", dbscan),
    ("Gaussian\nMixture", gmm),
)

# %% Run different clustering algo on the different transmorms
datasets = (
  aggregate_cell_metrics_df_clustering[[raw_scatter_dims[0],raw_scatter_dims[1]]].values,
  pca,
  tSNE_of_PCA,
  tSNE
)
#(aggregate_cell_metrics_df_clustering[[raw_scatter_dims[0],raw_scatter_dims[1]]].values, cluster_params)
plot_num = len(datasets)+1

# %% Dimensionality reduction display

fig = make_subplots(len(datasets), cols=2,
                    shared_xaxes='rows', shared_yaxes='rows')

for i_dataset, dataset in enumerate(datasets):
   

  fig.add_trace(
        go.Histogram2d(x= dataset[:,0],
                    y= dataset[:,1]),
                    row = i_dataset+1, col=1
  )

  fig.add_trace(
        go.Scatter(x= dataset[:,0],
                    y= dataset[:,1], 
                    mode='markers',
                    marker=dict(color=aggregate_cell_metrics_df_clustering[color_column],
                                size=aggregate_cell_metrics_df_clustering[size_column]/10,
                                line=dict(color=aggregate_cell_metrics_df_clustering[color_column], width=0.5))
                    ),
                    row = i_dataset+1, col=2
  )

fig.update_layout(height=1000, width=1500, title_text="Scatter plots")
fig.show()

# %% Clustering display
titles = [algo[0] for algo in clustering_algorithms]
title_list = list()
for title in titles:
   title_list.append(title)
for row in datasets:
  for title in titles:
     title_list.append('')

fig = make_subplots(len(datasets), cols=len(clustering_algorithms),
                    shared_xaxes='rows', shared_yaxes='rows',
                    subplot_titles=title_list)

for i_cluster, (name, algorithm) in enumerate(clustering_algorithms):
  # datasets[1] is pca so clustering all on PCA and display on othe dim reduction
  algorithm.fit(tSNE_of_PCA) 
  if hasattr(algorithm, "labels_"):
      y_pred = algorithm.labels_.astype(int)
  else:
      y_pred = algorithm.predict(datasets[1])

  for i_dataset, dataset in enumerate(datasets):
    #   subplots[i_dataset, i_cluster].set_title(name, size=18)


    colors = np.array(
      list(
          islice(
              cycle(
                  [
                      "#377eb8",
                      "#ff7f00",
                      "#4daf4a",
                      "#f781bf",
                      "#a65628",
                      "#984ea3",
                      "#999999",
                      "#e41a1c",
                      "#dede00",
                  ]
              ),
              int(max(y_pred) + 1),
          )
      )
    )
    # add black color for outliers (if any)
    colors = np.append(colors, ["#000000"])

    fig.add_trace(
            go.Scatter(x=dataset[:,0],
                          y=dataset[:,1],
                          mode='markers',
                          marker=dict(color=y_pred,
                                      size = 2,
                                      opacity = 0.3),
                                      
                    ), row=i_dataset+1, col=i_cluster+1
    )

    plot_num += 1

fig.update_layout(height=1000, width=1500, title_text="Scatter plots")
fig.show()





# %% 


fig = px.scatter(x=tSNE[:,0], y=tSNE[:,1], 
                 color = aggregate_cell_metrics_df_clustering[color_column],
                 size = aggregate_cell_metrics_df_clustering[size_column])
fig.show()

fig = px.scatter(x=pca[:,0], y=pca[:,1],
                 color = aggregate_cell_metrics_df_clustering[color_column],
                 size = aggregate_cell_metrics_df_clustering[size_column])
fig.show()

fig = px.scatter(x = aggregate_cell_metrics_df_clustering[raw_scatter_dims[0]], 
                  y = aggregate_cell_metrics_df_clustering[raw_scatter_dims[1]],
                  color = aggregate_cell_metrics_df_clustering[color_column],
                  size = aggregate_cell_metrics_df_clustering[size_column])
fig.show()

# %% ChatGPT plot:



fig = make_subplots(rows=3, cols=2, shared_xaxes='rows', shared_yaxes='rows')

fig.add_trace(
    go.Histogram2d(x=aggregate_cell_metrics_df_clustering[raw_scatter_dims[0]], 
               y=aggregate_cell_metrics_df_clustering[raw_scatter_dims[1]],
               ),
    row=1, col=1
)


fig.add_trace(
    go.Histogram2d(x=pca[:,0], y=pca[:,1],
               ),
    row=2, col=1
)

fig.add_trace(
    go.Histogram2d(x=tSNE[:,0], y=tSNE[:,1],
               ),
    row=3, col=1
)


fig.add_trace(
    go.Scatter(x=aggregate_cell_metrics_df_clustering[raw_scatter_dims[0]], 
               y=aggregate_cell_metrics_df_clustering[raw_scatter_dims[1]],
               mode='markers',
               marker=dict(color=aggregate_cell_metrics_df_clustering[color_column],
                           size=aggregate_cell_metrics_df_clustering[size_column]/100)),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(x=tSNE[:,0], y=tSNE[:,1], 
               mode='markers',
               marker=dict(color=aggregate_cell_metrics_df_clustering[color_column],
                           size=aggregate_cell_metrics_df_clustering[size_column]/100)),
    row=2, col=2
)

fig.add_trace(
    go.Scatter(x=pca[:,0], y=pca[:,1],
               mode='markers',
               marker=dict(color=aggregate_cell_metrics_df_clustering[color_column],
                           size=aggregate_cell_metrics_df_clustering[size_column]/100)),
    row=3, col=2
)



fig.update_layout(height=600, width=900, title_text="Scatter plots")
fig.show()
# %% Save global cell_metrics DataFrame
  # np.save(probe_folder / 'all_raw_waveforms.npy', all_raw_wf)
      # Save a full version of the cell-metrics dataframe  
aggregate_cell_metrics_df.to_pickle(probe_folder / 'aggregate_cell_metrics_df.pkl')
# %%
