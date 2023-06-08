"Helper functions for visualizing CoalMappers"

import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import sys

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns

from umap import UMAP
import hdbscan
import pickle

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.decomposition import PCA

pio.renderers.default = "browser"
load_dotenv
root = os.getenv("root")
src = os.getenv("src")
sys.path.append(src)


def plot_dendrogram(model, labels, distance, p, n, distance_threshold, **kwargs):
    """Create linkage matrix and then plot the dendrogram for Hierarchical clustering."""

    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    d = dendrogram(
        linkage_matrix,
        p=p,
        distance_sort=True,
        color_threshold=distance_threshold,
    )
    for leaf, leaf_color in zip(plt.gca().get_xticklabels(), d["leaves_color_list"]):
        leaf.set_color(leaf_color)
    plt.title(f"Clustering Models with {n} Policy Groups")
    plt.xlabel("Coordinates: Model Parameters.")
    plt.ylabel(f"{distance} distance between persistence diagrams")
    plt.show()
    return d


def UMAP_grid(dir='../../data/projections/UMAP/'):
    """function reads in a df, outputs a grid visualization with n by n UMAP projected dataset visualizations

    grid search the UMAP parameter space, choose the representations that occur most often in the given parameter space, based on the generated histogram"""
    neighbors, dists = [], []
    for umap in os.listdir(dir):
        with open (f"{dir}/{umap}", 'rb') as f:
            params = pickle.load(f)
        if params['hyperparameters'][0] in neighbors:
            ""
        else:
            neighbors.append(params['hyperparameters'][0])
        if params['hyperparameters'][1] in dists:
            ''
        else:
            dists.append(params['hyperparameters'][1])
        neighbors.sort()
        dists.sort()

    fig = make_subplots(
            rows=len(dists),
            cols=len(neighbors),
            column_titles=list(map(str, neighbors)),
            x_title="n_neighbors",
            row_titles=list(map(str, dists)),
            y_title="min_dist",
            #vertical_spacing=0.05,
            #horizontal_spacing=0.03,
            # shared_xaxes=True,
            # shared_yaxes=True,
        )

    cluster_distribution = []
    row = 1
    col = 1
    for umap in os.listdir(dir):
        with open (f"{dir}/{umap}", 'rb') as f:
            params = pickle.load(f)
        proj_2d = params['projection']
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5).fit(proj_2d)
        outdf = pd.DataFrame(proj_2d, columns=["0", "1"])
        outdf["labels"] = clusterer.labels_

        num_clusters = len(np.unique(clusterer.labels_))
        cluster_distribution.append(num_clusters)
        df = outdf[outdf["labels"] == -1]
        fig.add_trace(
            go.Scatter(
                x=df["0"],
                y=df["1"],
                mode="markers",
                marker=dict(
                    size=2.3,
                    color="red",
                    #line=dict(width=0.2, color="Black"),
                ),
                hovertemplate=df["labels"],
            ),
            row=row, col = col,
        )
        df = outdf[outdf["labels"] != -1]
        fig.add_trace(go.Scatter(
                x=df["0"],
                y=df["1"],
                mode="markers",
                marker=dict(
                    size=4,
                    color= df["labels"],
                    cmid=0.5,
                ),
                hovertemplate=df["labels"],
            ),
            row=row, col = col,
        )
        row+=1
        if row == len(dists)+1:
            row = 1
            col+=1


    fig.update_layout(height=900,
        template="simple_white", showlegend=False, font=dict(color="black",)
    )

    fig.update_xaxes(showticklabels=False, tickwidth=0, tickcolor="rgba(0,0,0,0)")
    fig.update_yaxes(showticklabels=False, tickwidth=0, tickcolor="rgba(0,0,0,0)")

    pio.show(fig)

def view_kmeans(df: str, numclusters: int):

    # data

    X = StandardScaler().fit_transform(df)

    # #############################################################################
    # Clustering with KMeans

    k_means = KMeans(init="k-means++", n_clusters=numclusters, n_init=10)
    t0 = time.time()
    k_means.fit(X)
    t_batch = time.time() - t0

    # #############################################################################
    # Clustering with MiniBatchKMeans

    mbk = MiniBatchKMeans(
        init="k-means++",
        n_clusters=numclusters,
        batch_size=numclusters,
        n_init=10,
        max_no_improvement=10,
        verbose=0,
    )
    t0 = time.time()
    mbk.fit(X)
    t_mini_batch = time.time() - t0

    # #############################################################################
    # Plot result

    fig = plt.figure(figsize=(8, 3))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    colors = ["#4EACC5", "#FF9C34", "#4E9A06", "#3082be"]

    # COLOR MiniBatchKMeans and the KMeans algorithm.
    k_means_cluster_centers = k_means.cluster_centers_
    order = pairwise_distances_argmin(k_means.cluster_centers_, mbk.cluster_centers_)
    mbk_means_cluster_centers = mbk.cluster_centers_[order]

    k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)
    mbk_means_labels = pairwise_distances_argmin(X, mbk_means_cluster_centers)

    # KMeans
    ax = fig.add_subplot(1, 3, 1)
    for k, col in zip(range(numclusters), colors):
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        ax.plot(
            X[my_members, 0], X[my_members, 1], "w", markerfacecolor=col, marker="."
        )
        ax.plot(
            cluster_center[0],
            cluster_center[1],
            "o",
            markerfacecolor=col,
            markeredgecolor="k",
            markersize=6,
        )
    ax.set_title("KMeans")
    ax.set_xticks(())
    ax.set_yticks(())
    plt.text(-3.5, 1.8, "train time: %.2fs\ninertia: %f" % (t_batch, k_means.inertia_))

    # MiniBatchKMeans
    ax = fig.add_subplot(1, 3, 2)
    for k, col in zip(range(numclusters), colors):
        my_members = mbk_means_labels == k
        cluster_center = mbk_means_cluster_centers[k]
        ax.plot(
            X[my_members, 0], X[my_members, 1], "w", markerfacecolor=col, marker="."
        )
        ax.plot(
            cluster_center[0],
            cluster_center[1],
            "o",
            markerfacecolor=col,
            markeredgecolor="k",
            markersize=6,
        )
    ax.set_title("MiniBatchKMeans")
    ax.set_xticks(())
    ax.set_yticks(())
    plt.text(-3.5, 1.8, "train time: %.2fs\ninertia: %f" % (t_mini_batch, mbk.inertia_))

    # Initialise the different array to all False
    different = mbk_means_labels == 4
    ax = fig.add_subplot(1, 3, 3)

    for k in range(numclusters):
        different += (k_means_labels == k) != (mbk_means_labels == k)

    identic = np.logical_not(different)
    ax.plot(X[identic, 0], X[identic, 1], "w", markerfacecolor="#bbbbbb", marker=".")
    ax.plot(X[different, 0], X[different, 1], "w", markerfacecolor="m", marker=".")
    ax.set_title("Difference")
    ax.set_xticks(())
    ax.set_yticks(())

    plt.show()

def _define_zscore_df(model):
    df_builder = pd.DataFrame()
    dfs = model.tupper.clean

    column_to_drop = [col for col in dfs.columns if dfs[col].nunique() == 1]
    dfs = dfs.drop(column_to_drop, axis=1)

    dfs['cluster_IDs']=(list(model.cluster_ids))

    #loop through all policy group dataframes
    for group in list(dfs['cluster_IDs'].unique()):
        zscore0 = pd.DataFrame()
        group0 = dfs[dfs['cluster_IDs']==group].drop(columns={'cluster_IDs'})
        #loop through all columns in a policy group dataframe
        for col in group0.columns:
            if col != "cluster_IDs":
                mean = dfs[col].mean()
                std = dfs[col].std()
                zscore0[col] = group0[col].map(lambda x: (x-mean)/std)
        zscore0_temp = zscore0.copy()
        zscore0_temp['cluster_IDs'] = group
        df_builder = pd.concat([df_builder,zscore0_temp])
    return df_builder