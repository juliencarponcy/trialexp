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
import seaborn as sns

from snakehelper.SnakeIOHelper import getSnake
from trialexp import config
#%% Load inputs


(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [config.debug_folder + r'/processed/cell_metrics_clustering.done'],
  'cell_metrics_clustering')


# %% Path definitions

sorter_name = 'kilosort3'
verbose = True

root_path = Path(config.SESSION_ROOT_DIR)
# Where to store globally computed figures
clusters_figure_path = Path(os.environ['CLUSTERS_FIGURES_PATH'])
# where to store global processed data
clusters_data_path = Path(os.environ['PROCCESSED_CLUSTERS_PATH'])


# %% Loading data

# Loading dataframe containing whole dataset dimensionality reductions for cell metrics
dim_reduc_aggregate = pd.read_pickle(clusters_data_path / 'aggregate_cell_metrics_dim_reduc.pkl')
# Loading dataframe containing whole dataset for cell metrics
aggregate_cell_metrics_df = pd.read_pickle(clusters_data_path/ 'aggregate_cell_metrics_df_full.pkl')
# Loading dataframe containing clustering data for cell metrics
aggregate_cell_metrics_df_clustering = pd.read_pickle(clusters_data_path/ 'aggregate_cell_metrics_df_clustering.pkl')

# Loading numpy array of raw waveforms across all channels
aggregate_raw_waveforms = np.load(clusters_data_path / 'all_raw_waveforms.npy')

# Check if aggregate_cell_metrics_df and dim_reduc_aggregate have the same length 
# TODO potentially identify culprit file(s)/sessions by disimilarity of the indices
assert len(dim_reduc_aggregate) == len(aggregate_cell_metrics_df) == aggregate_raw_waveforms.shape[2], \
  f" aggregate_cell_metrics_df, dim_reduc_aggregate and raw_waveforms don't have the same length: {len(dim_reduc_aggregate)}, {len(aggregate_cell_metrics_df)}, {aggregate_raw_waveforms.shape[2]}"

# %% Params for clustering
cluster_params= {
  "n_clusters": 2,
  "eps": 0.3,
  "n_init": 10
}

# Raw variables to plot for comparison / visualisation
raw_scatter_dims = ('peak_average_all_wf', 'std_std_waveform')
 
# Plotting params
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams["legend.frameon"] = False
plt.rcParams['font.family'] = ['Arial']
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

# %% Building the various datasets out of the dim_reduction DataFrame and cell_metrics DataFrame
# Turning DataFrame columns into numpy arrays

pca_cols = [col for col in dim_reduc_aggregate.columns if 'PCA' in col]
tsne_cols = [col for col in dim_reduc_aggregate.columns if 'tSNE' in col]
tsne_on_pca_cols = [col for col in dim_reduc_aggregate.columns if 'tSNE_of_pca' in col]

raw_variables = aggregate_cell_metrics_df[[raw_scatter_dims[0],raw_scatter_dims[1]]].values
pca = dim_reduc_aggregate[pca_cols].values
tSNE = dim_reduc_aggregate[tsne_cols].values
tSNE_of_pca = dim_reduc_aggregate[tsne_on_pca_cols].values

# Grouping all the different datasets in a tuple
datasets = (
  raw_variables,
  pca,
  tSNE,
  tSNE_of_pca
)
# %% THe following section defines key parameters:

# Namely, the dimensionality reduction embedding on which to cluster, and the clustering algo to use:

## IMPORTANT: Define which dataset or dimensionality reduction embedding will be used for clustering
dataset_to_cluster_on = pca
## IMPORTANT: Clustering algorithm used for storing labels:
clustering_algo_used_to_label = gmm # gaussian mixture model

# All the algorithm will be tested and plotted below, but the one defined above will
# be the one from which the labels will be stored in the dim_reduc_aggregate DataFrame
# This can be changed to another clustering algo for different data/purpose


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

fig_cluster.update_xaxes(title='1st Component', title_font_family="Arial")
fig_cluster.update_yaxes(title='2nd Component', title_font_family="Arial")

fig_cluster.update_layout(height=1000, width=1500,
  title_text=f"Cell/Artifacts clustering for {aggregate_cell_metrics_df.shape[0]} clusters")

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


# %% Save the results of the clustering in the dim_reduc_aggregate DataFrame

dim_reduc_aggregate['cluster_label'] = clustering_algo_used_to_label.predict(dataset_to_cluster_on)

# %% Plot random waveforms from all clusters

# nb of channels to plot around (both sides) the Max channel
side_channels = 4
# Nb of random examples to draw from each cluster
nb_examples = 4

labels = dim_reduc_aggregate['cluster_label'].unique()
random_ids = np.ndarray((nb_examples,len(labels)))

colors = plt.cm.jet(np.linspace(0,1,side_channels*2+1))

for label_id, label in enumerate(labels):

  fig, axes = plt.subplots(nrows=nb_examples, ncols=3, figsize=(12,nb_examples*3))
  fig.suptitle(f'cluster nb {str(label)}')
  random_ids[:, label_id] = np.random.choice(
     aggregate_cell_metrics_df[dim_reduc_aggregate['cluster_label'] == label].index.values, nb_examples)

  random_ids = random_ids.astype(int)

  
  # select max waveform trace and neighboring channels
  for i_plot, i_cell in enumerate(random_ids[:,label_id]):
    if i_plot == 0:
      axes[0, 0].set_title(f'{side_channels*2+1} waveforms')
      axes[0, 1].set_title(f'{aggregate_raw_waveforms.shape[0]} waveforms')
      axes[0, 2].set_title(f'Dim. reduction location')

    # Adapt to cases where the max Channel for the waveform is close to the tip/bottom contact
    if aggregate_cell_metrics_df.maxWaveformCh.iloc[i_cell] - side_channels > 0 & aggregate_cell_metrics_df.maxWaveformCh.iloc[i_cell] + side_channels < aggregate_raw_waveforms.shape[0]: 
      waveforms = aggregate_raw_waveforms[
        aggregate_cell_metrics_df.maxWaveformCh.iloc[i_cell] - side_channels : aggregate_cell_metrics_df.maxWaveformCh.iloc[i_cell] + side_channels,:,i_cell]
    elif aggregate_cell_metrics_df.maxWaveformCh.iloc[i_cell] - side_channels < 0:
      waveforms = aggregate_raw_waveforms[
        0 : aggregate_cell_metrics_df.maxWaveformCh.iloc[i_cell] + side_channels,:,i_cell]
    elif aggregate_cell_metrics_df.maxWaveformCh.iloc[i_cell] + side_channels >=  aggregate_raw_waveforms.shape[0]:
      waveforms = aggregate_raw_waveforms[
        aggregate_cell_metrics_df.maxWaveformCh.iloc[i_cell] - side_channels : ,:,i_cell] 
   
    for ch in range(waveforms.shape[0]):
      axes[i_plot, 0].plot(waveforms[ch,:], color=colors[ch])

    # Careful here, plotting in position in the PCA dim reduction is hard-coded below
    axes[i_plot, 1].pcolor(aggregate_raw_waveforms[:,:,i_cell].squeeze())
    axes[i_plot, 2].scatter(dim_reduc_aggregate.PCA_1, dim_reduc_aggregate.PCA_2,  s=1, marker='.', c='k', alpha=0.2)
    axes[i_plot, 2].scatter(dim_reduc_aggregate.PCA_1.iloc[i_cell], dim_reduc_aggregate.PCA_2.iloc[i_cell],  s=30, marker='o', c='r')

# for label in :
   
# %% Plot a few cell_metrics of choice (just modify the display cols) 
# in the different clusters detected
# allow to get an idea of the 
display_cols = [ 'cv2',
                'firingRate',
                'acg_refrac',
                'troughToPeak',
                'std_std_waveform',
                'peak_average_all_wf',
                ]
for col in display_cols:
  distrib_cluster_fig = plt.figure(figsize=(5,5))
  log_distrib = sns.kdeplot(aggregate_cell_metrics_df, x=col,  hue=dim_reduc_aggregate['cluster_label'], levels=4, color=".2", log_scale=True)

  # sns.violinplot(data = aggregate_cell_metrics_df, x = dim_reduc_aggregate['cluster_label'], y=col, hue=dim_reduc_aggregate['cluster_label'])
  # sns.swarmplot(x = aggregate_cell_metrics_df.cluster_label, y = aggregate_cell_metrics_df[col], color="white")
  plt.show()
  distrib_cluster_fig.savefig(clusters_figure_path / str('cell_metrics_log_clusters_distrib_' + col + '.png'))


# %%
