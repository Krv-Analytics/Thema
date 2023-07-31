"Helper functions for visualizing and evaluating jmap runs "

import os
import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

pio.renderers.default = "browser"

from dotenv import load_dotenv
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



################################################################################################
#  Handling Local Imports  
################################################################################################

load_dotenv()
root = os.getenv("root")
src = os.getenv("src")
sys.path.append(src + "jmapping/selecting/")
sys.path.append(root + "logging/")

from jmap_selector_helper import (
    unpack_policy_group_dir,
    get_viable_jmaps
)


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


def plot_curvature_histogram(dir, coverage):
    num_curvature_profiles = {}
    for folder in os.listdir(dir):
        i = unpack_policy_group_dir(folder)
        folder = os.path.join(dir, folder)
        if os.path.isdir(folder):
            holder = []
            for file in os.listdir(folder):
                if file.endswith(".pkl"):
                    file = os.path.join(folder, file)
                    with open(file, "rb") as f:
                        matrix = pickle.load(f)["distances"]
                holder.append(int(len(matrix)))
            # For now take the maximum number of unique curvature profiles over different metrics
            num_curvature_profiles[i] = max(holder)

    fig = plt.figure(figsize=(15, 10))
    ax = sns.barplot(
        x=list(num_curvature_profiles.keys()),
        y=list(num_curvature_profiles.values()),
    )
    ax.set(xlabel="Number of Policy Groups", ylabel="Number of Curvature Profiles")
    ax.set_title(f"{coverage*100} % Coverage Filter")
    plt.show()
    return fig


def plot_jmapper_histogram(dir, coverage=0.8):
    """
    Plots a histogram of the number of viable jmappers for each rank
    of policy groupings. This function will count the jmappers
    (per `num_policy_groups`) that have been generated
    according to a hyperparameter grid search.

    Parameters:
    -----------
    coverage_filter: float
        The minimum coverage percentage required for a jmap
        to be considered viable.

    Returns:
    --------
    fig: matplotlib.figure.Figure
        The plotted figure object.
    """
    mappers = os.path.join(root, dir)
    # Get list of folder names in the directory
    policy_groups = os.listdir(mappers)
    # Initialize counting dictionary
    counts = {}
    for folder in policy_groups:
        n = unpack_policy_group_dir(folder)
        path_to_jmaps = dir + folder
        jmaps = get_viable_jmaps(path_to_jmaps, n, coverage)
        counts[n] = len(jmaps)
    keys = list(counts.keys())
    keys.sort()
    sorted_counts = {i: counts[i] for i in keys}
    # plot the histogram
    fig = plt.figure(figsize=(15, 10))
    ax = sns.barplot(
        x=list(sorted_counts.keys()),
        y=list(sorted_counts.values()),
    )
    ax.set(xlabel="Number of Policy Groups", ylabel="Number of Viable jmaps")
    ax.set_title(f"{coverage*100} % Coverage Filter")
    plt.show()
    return fig


def plot_stability_histogram(dir, coverage):

    jmap_dir = os.path.join(root, dir + "/jmaps/")
    metric_files = os.path.join(
        root,
        dir 
        + f"/jmap_analysis/distance_matrices/{coverage}_coverage/",
    )
    
    num_jmaps = {}
    for folder in os.listdir(jmap_dir):
        i = unpack_policy_group_dir(folder)
        folder = os.path.join(jmap_dir, folder)
        jmaps = get_viable_jmaps(folder, i, coverage_filter=coverage)
        num_jmaps[i] = len(jmaps)

    num_curvature_profiles = {}
    
    for folder in os.listdir(metric_files):
        i = unpack_policy_group_dir(folder)
        folder = os.path.join(metric_files, folder)
        if os.path.isdir(folder):
            holder = []
            for file in os.listdir(folder):
                if file.endswith(".pkl"):
                    file = os.path.join(folder, file)
                    with open(file, "rb") as f:
                        matrix = pickle.load(f)["distances"]
                holder.append(int(len(matrix)))
            # For now take the maximum number of unique curvature profiles over different metrics
            num_curvature_profiles[i] = max(holder)
    stability_ratio = {}
    for key in num_curvature_profiles.keys():
        stability_ratio[key] = num_jmaps[key] / num_curvature_profiles[key]
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(15, 20))
    fig.suptitle(f"{coverage*100}% Coverage Filter")
    sns.barplot(
        x=list(stability_ratio.keys()),
        y=list(stability_ratio.values()),
        ax=ax,
    )
    ax.set(xlabel="Number of Policy Groups", ylabel="Stability Ratio")
    plt.show()
    return fig
