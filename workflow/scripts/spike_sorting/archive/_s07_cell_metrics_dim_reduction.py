'''
Script to create the session folder structure
'''
#%%
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.preprocessing import StandardScaler

from snakehelper.SnakeIOHelper import getSnake
from trialexp import config
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
# %% Dimensionality reduction functions definition

# These should be upgraded to include variance explained output and weights?
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
 
# %% Save dimensionality reduction of cell metrics in a separate aggregated dataframe
dim_reductions = (pca, tSNE, tSNE_of_PCA)
# attribute names to the dim reduction for column naming in the DataFrame
dim_reduc_names = ('PCA', 'tSNE', 'tSNE_of_pca')

dim_reduc_aggregate = pd.DataFrame()
# import the UID columns from the aggregate_cell_metrics_df for further comparison of the
# integrity and consistency of datasets
dim_reduc_aggregate['UID'] = aggregate_cell_metrics_df.loc[:,('UID')]

for (dim_reduc_name, dim_reduction) in dict(zip(dim_reduc_names, dim_reductions)).items():

  for component_nb in range(dim_reduction.shape[1]):
    # Component nb will be named with 1-based indexing (eg: PCA_1, PCA_2, etc.) hence str(component_nb+1)
    dim_reduc_aggregate[dim_reduc_name + '_' + str(component_nb+1)] = dim_reduction[:, component_nb]

# Save the DataFrame of dim reduction to a pickle file.
dim_reduc_aggregate.to_pickle(clusters_data_path / 'aggregate_cell_metrics_dim_reduc.pkl')
# %%
