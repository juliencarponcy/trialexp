'''
Script to create the session folder structure
'''
#%%
import os
from pathlib import Path
from itertools import cycle, islice

import numpy as np
import pandas as pd

from sklearn import cluster, mixture

from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from snakehelper.SnakeIOHelper import getSnake
from workflows.scripts import settings

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
    
# %% Params for clustering
cluster_params= {
  "n_clusters": 2,
  "eps": 0.3,
  "n_init": 10
}

 
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
      y_pred = algorithm.predict(dataset_to_cluster_on)

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



# %% Save global cell_metrics DataFrame
  # np.save(probe_folder / 'all_raw_waveforms.npy', all_raw_wf)
      # Save a full version of the cell-metrics dataframe  
aggregate_cell_metrics_df.to_pickle(clusters_data_path / 'aggregate_cell_metrics_df.pkl')
aggregate_cell_metrics_df.to_pickle(clusters_data_path / 'aggregate_cell_metrics_df.pkl')


# %%



# %% 

gmm.fit(pca)
# clusters_labels = pd.DataFrame([gmm.predict(pca).T, pca[:,0], pca[:,1]], columns=['cluster_label', 'pca_x', 'pca_y'])
aggregate_cell_metrics_df['pca_label'] = gmm.predict(pca)
aggregate_cell_metrics_df['pca_pos_x'] = pca[:,0]
aggregate_cell_metrics_df['pca_pos_y'] = pca[:,1]

# %% Plot random waveforms from both clusters

side_channels = 4
nb_examples = 4



labels = aggregate_cell_metrics_df['pca_label'].unique()
random_ids = np.ndarray((nb_examples,len(labels)))

colors = plt.cm.jet(np.linspace(0,1,side_channels*2+1))

for label_id, label in enumerate(labels):

  fig, axes = plt.subplots(nrows=nb_examples, ncols=3, figsize=(12,nb_examples*3))
  fig.suptitle(f'cluster nb {str(label)}')
  random_ids[:, label_id] = np.random.choice(
     aggregate_cell_metrics_df[aggregate_cell_metrics_df['pca_label'] == label].index.values, nb_examples)

  random_ids = random_ids.astype(int)

  
  # select max waveform trace and neighboring channels
  for i_plot, i_cell in enumerate(random_ids[:,label_id]):
    if i_plot == 0:
      axes[0, 0].set_title(f'{side_channels*2+1} waveforms')
      axes[0, 1].set_title(f'{aggregate_raw_waveforms.shape[0]} waveforms')
      axes[0, 2].set_title(f'PCA location')

    if aggregate_cell_metrics_df.maxWaveformCh.iloc[i_cell] - side_channels > 0 & aggregate_cell_metrics_df.maxWaveformCh.iloc[i_cell] + side_channels < aggregate_raw_waveforms.shape[0]: 
      waveforms = aggregate_raw_waveforms[
        aggregate_cell_metrics_df.maxWaveformCh.iloc[i_cell] - side_channels : aggregate_cell_metrics_df.maxWaveformCh.iloc[i_cell] + side_channels,:,i_cell]
    elif aggregate_cell_metrics_df.maxWaveformCh.iloc[i_cell] - side_channels < 0:
      waveforms = aggregate_raw_waveforms[
        0 : aggregate_cell_metrics_df.maxWaveformCh.iloc[i_cell] + side_channels,:,i_cell]
    elif aggregate_cell_metrics_df.maxWaveformCh.iloc[i_cell] + side_channels >=  aggregate_raw_waveforms.shape[0]:
      waveforms = aggregate_raw_waveforms[
        aggregate_cell_metrics_df.maxWaveformCh.iloc[i_cell] - side_channels : ,:,i_cell] 
   
    for ch in range(side_channels*2):
      axes[i_plot, 0].plot(waveforms[ch,:], color=colors[ch])

    axes[i_plot, 1].pcolor(aggregate_raw_waveforms[:,:,i_cell].squeeze())
    axes[i_plot, 2].scatter(aggregate_cell_metrics_df.pca_pos_x, aggregate_cell_metrics_df.pca_pos_y,  s=1, marker='.', c='k', alpha=0.2)
    axes[i_plot, 2].scatter(aggregate_cell_metrics_df.pca_pos_x.iloc[i_cell], aggregate_cell_metrics_df.pca_pos_y.iloc[i_cell],  s=30, marker='o', c='r')

# for label in :
   
# %%
display_cols = [ 'cv2',
                'firingRate',
                'acg_refrac',
                'troughToPeak',
                'std_std_waveform',
                'peak_average_all_wf',
                ]
for col in display_cols:
  fig = plt.figure(figsize=(5,5))
  sns.violinplot(data = aggregate_cell_metrics_df, x ='pca_label', y=col, hue='pca_label')
  # sns.swarmplot(x = aggregate_cell_metrics_df.pca_label, y = aggregate_cell_metrics_df[col], color="white")
  plt.show()