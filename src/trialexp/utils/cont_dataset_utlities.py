import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import DBSCAN
from sklearn import metrics


def DBSCAN_cluster_on_compoonents(
        np_comp: np.ndarray, 
        eps: float,
        min_samples_by_cluster: int = 20, 
        plot: bool = True,
        plot_lim_pctile: float = 0.1
        ):
        
    db = DBSCAN(eps=eps, min_samples = min_samples_by_cluster).fit(np_comp)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    # print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
    # print(
    #     "Adjusted Mutual Information: %0.3f"
    #     % metrics.adjusted_mutual_info_score(labels_true, labels)
    #)
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(np_comp, labels))

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    if plot:
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

        fig, axs = plt.subplots(1,1, figsize=(10,10))

        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = labels == k

            xy = np_comp[class_member_mask & core_samples_mask]
            axs.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=6,
                alpha=0.2
            )

            xy = np_comp[class_member_mask & ~core_samples_mask]
            axs.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=3,
                alpha=0.2
            )

    axs.set_title("Estimated number of clusters: %d" % n_clusters_)
    xlim = [np.percentile(np_comp[:,0],plot_lim_pctile), np.percentile(np_comp[:,0],100-plot_lim_pctile)]
    ylim = [np.percentile(np_comp[:,1],plot_lim_pctile), np.percentile(np_comp[:,1],100-plot_lim_pctile)]
    
    axs.set_xlim(xlim)
    axs.set_ylim(ylim)

    pd_labels = pd.Series(labels)
    print('samples per cluster: ', pd_labels.value_counts())

    return labels

def extract_feature_components(panel_df,
        variable: str = 'analog_2', 
        type: str = 'PCA', 
        scaled: bool = True,
        plot: bool = False,
        plot_lim_pctile: float = 0.1
        ):
    
    df_feat = ts_to_features(panel_df[variable])

    if scaled:

        scaler = StandardScaler()
        np_scaled = scaler.fit_transform(df_feat)

    else:
        np_scaled = df_feat
    
    if type == 'PCA':
        
        comp_obj = PCA(n_components = np_scaled.shape[1])
    
    if type == 'ICA':
        
        comp_obj = FastICA(n_components = np_scaled.shape[1])

    
    np_comp = comp_obj.fit_transform(np_scaled)

    if plot:
        fig, axs = plt.subplots(1,1, figsize=(10,10))
        axs.scatter(np_comp[:,0],np_comp[:,1],marker='o', s=7, alpha=0.1)
        xlim = [np.percentile(np_comp[:,0],plot_lim_pctile), np.percentile(np_comp[:,0],100-plot_lim_pctile)]
        ylim = [np.percentile(np_comp[:,1],plot_lim_pctile), np.percentile(np_comp[:,1],100-plot_lim_pctile)]
        
        axs.set_xlim(xlim)
        axs.set_ylim(ylim)
 
    return np_comp

def ts_to_features(ts):
    # progress_apply exists after running tqdm.pandas() and yields a progress bar
    return pd.DataFrame(ts.apply(base_feature_extraction).tolist())

def base_feature_extraction(x):
    # no corr() or cov() because it is a single variate time series
    return {
        #"mean": x.mean(),
        "median": x.median(),
        "max-min": x.max() - x.min(),
        "var": x.var(),
        "std": x.std(),
        "kurt": x.kurt(),
        "skew": x.skew()
    }

