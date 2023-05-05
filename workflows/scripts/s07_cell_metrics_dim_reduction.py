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
# Where to store globally computed figures
clusters_figure_path = Path(os.environ['CLUSTERS_FIGURES_PATH'])
# where to store global processed data
clusters_data_path = Path(os.environ['PROCCESSED_CLUSTERS_PATH'])
# list all cell metrics dataframes processed so far.
cell_metrics_paths = list(root_path.glob(f'*/*/processed/{sorter_name}/*/sorter_output/cell_metrics_df_full.pkl'))

# %% Load and check cell metrics and waveforms aggregated data

aggregate_cell_metrics_df_clustering = pd.read_pickle(clusters_data_path/ 'aggregate_cell_metrics_df_clustering.pkl')
aggregate_cell_metrics_df = pd.read_pickle(clusters_data_path/ 'aggregate_cell_metrics_df_full.pkl')
aggregate_raw_waveforms = np.load(clusters_data_path / 'all_raw_waveforms.npy')

# Check if cell_metrics and raw_waveforms paths have the same length 
# TODO potentially identify culprit file(s) by disimilarity of cell_metrics_paths and raw_waveforms_paths
assert len(aggregate_cell_metrics_df) == len(aggregate_cell_metrics_df_clustering) == aggregate_raw_waveforms.shape[2], \
  f"cell_metrics_df and raw_waveforms dont have the same lengths: {len(aggregate_cell_metrics_df_clustering)}, {len(aggregate_cell_metrics_df)}, {aggregate_raw_waveforms.shape[2]}"



# %% Params and variables definition cell

color_column = 'peak_average_all_wf'
size_column = 'std_std_waveform'
# choose 2 raw values for scatter plot
raw_scatter_dims = ('peak_average_all_wf', 'std_std_waveform')
# print cluster_UID as data point label
text_datapoints = aggregate_cell_metrics_df_clustering.index.values


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

## IMPORTANT: Define which dataset or dimensionality reduction will be used for clustering
dataset_to_cluster_on = pca


# %% Dimensionality reduction display

fig_reduc = make_subplots(len(datasets), cols=2,
                    shared_xaxes='rows', shared_yaxes='rows')

for i_dataset, dataset in enumerate(datasets):
   

  fig_reduc.add_trace(
        go.Histogram2d(x= dataset[:,0],
                    y= dataset[:,1]),
                    row = i_dataset+1, col=1
  )

  fig_reduc.add_trace(
        go.Scatter(x= dataset[:,0],
                    y= dataset[:,1], 
                    mode='markers',
                    marker=dict(color=aggregate_cell_metrics_df_clustering[color_column],
                                size=aggregate_cell_metrics_df_clustering[size_column]/10,
                                line=dict(color=aggregate_cell_metrics_df_clustering[color_column], width=0.5))
                    ),
                    row = i_dataset+1, col=2
  )

# %% Add info and save interactive figure to HTML file

fig_reduc.update_xaxes(title='1st Dimension', title_font_family="Arial")
fig_reduc.update_yaxes(title='2nd Dimension', title_font_family="Arial")

fig_reduc.update_layout(height=1000, width=1500,
                   title_text=f"Density and scatter plots \
                    of different embeddings for {aggregate_cell_metrics_df_clustering.shape[0]} \
                    clusters and {aggregate_cell_metrics_df_clustering.shape[1]} cells metrics")

# Examples to improve formatting of figure
# fig_reduc.update_layout(
#     font_family="Courier New",
#     font_color="blue",
#     title_font_family="Times New Roman",
#     title_font_color="red",
#     legend_title_font_color="green"
# )
fig_reduc.show()

fig_reduc.write_html(clusters_figure_path / 'cell_metrics_dim_reduc.html')
fig_reduc.show()

# %% Clustering display
titles = [algo[0] for algo in clustering_algorithms]
title_list = list()
for title in titles:
   title_list.append(title)
for row in datasets:
  for title in titles:
     title_list.append('')

fig_cluster = make_subplots(len(datasets), cols=len(clustering_algorithms),
                    shared_xaxes='rows', shared_yaxes='rows',
                    subplot_titles=title_list)

for i_cluster, (name, algorithm) in enumerate(clustering_algorithms):

  algorithm.fit(dataset_to_cluster_on) 
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

    fig_cluster.add_trace(
            go.Scatter(x=dataset[:,0],
                          y=dataset[:,1],
                          mode='markers',
                          marker=dict(color=y_pred,
                                      size = 2,
                                      opacity = 0.3),
                                      
                    ), row=i_dataset+1, col=i_cluster+1
    )

# %% Add info and save interactive figure to HTML file

fig_cluster.update_xaxes(title='1st Dimension', title_font_family="Arial")
fig_cluster.update_yaxes(title='2nd Dimension', title_font_family="Arial")

fig_cluster.update_layout(height=1000, width=1500,
  title_text=f"Cell/Artifacts clustering for {aggregate_cell_metrics_df_clustering.shape[0]} clusters and {aggregate_cell_metrics_df_clustering.shape[1]} cells metrics")

# Formatting examples
# fig_cluster.update_layout(
#     font_family="Courier New",
#     font_color="blue",
#     title_font_family="Times New Roman",
#     title_font_color="red",
#     legend_title_font_color="green"
# )
fig_cluster.show()

fig_cluster.write_html(clusters_figure_path / 'cell_metrics_clustering.html')


# %% 



# %% Save global cell_metrics DataFrame
  # np.save(probe_folder / 'all_raw_waveforms.npy', all_raw_wf)
      # Save a full version of the cell-metrics dataframe  
aggregate_cell_metrics_df.to_pickle(probe_folder / 'aggregate_cell_metrics_df.pkl')
# %%
