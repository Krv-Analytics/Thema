"Helper functions for visualizing and evaluating jmap runs "

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
from scipy.cluster.hierarchy import dendrogram

import hdbscan
import pickle

from sklearn.preprocessing import StandardScaler
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

def visualize_PCA(model, colors=True):
        df=model.tupper.clean.copy()
        df['cluster_IDs'] = model.cluster_ids
        # Standardize the data
        X = StandardScaler().fit_transform(df)
        # Perform PCA
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X)
        # Create a DataFrame for the principal components
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        # Add the original cluster IDs to the PCA DataFrame
        pca_df['cluster_IDs'] = df['cluster_IDs'].values
        # plot with cluster-label based color scheme
        if colors:
            fig = go.Figure()
            cluster_list = list(pca_df['cluster_IDs'].unique())
            cluster_list.sort()
            for cluster in cluster_list:
                plot = pca_df[pca_df['cluster_IDs']==cluster]
               # fig.add_trace(go.Scatter( x=plot['PC1'], y=plot['PC2'], mode='markers', name=cluster, marker=dict(color=custom_color_scale()[cluster_list.index(cluster)][1])))
        # plot with no colors
        else:
            fig = px.scatter(data_frame=pca_df, x='PC1', y='PC2', color_discrete_sequence=['grey'])
        fig.update_layout(template='simple_white', width=800, height=600)
        fig.show()