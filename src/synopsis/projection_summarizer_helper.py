# Projection summary functionality

import os
import sys

from __init__ import env
from dotenv import load_dotenv

root, src = env()  # Load .env

import pickle

import hdbscan
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

import jmapping as jm


def create_umap_grid(dir):
    """
    Create a grid visualization of UMAP projections.

    Parameters:
    -----------
    dir : str
        The directory containing the UMAP projections.

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The Plotly figure object representing the UMAP grid visualization.
    """

    umap_data = []
    for umap in os.listdir(dir):
        if not umap.endswith('.pkl'):
            raise ValueError('Non-UMAP files present, likely culprit: .DS_Store')
        else:
            with open(f"{dir}/{umap}", "rb") as f:
                params = pickle.load(f)
            umap_data.append((umap, params))

    umap_data.sort(
        key=lambda x: (x[1]["hyperparameters"][0], x[1]["hyperparameters"][1])
    )

    neighbors = sorted(
        list(set([params["hyperparameters"][0] for _, params in umap_data]))
    )
    dists = sorted(list(set([params["hyperparameters"][1] for _, params in umap_data])))

    fig = make_subplots(
        rows=len(dists),
        cols=len(neighbors),
        column_titles=list(map(str, neighbors)),
        x_title="n_neighbors",
        row_titles=list(map(str, dists)),
        y_title="min_dist",
    )

    cluster_distribution = []
    row = 1
    col = 1

    for umap_file, params in umap_data:
        with open(f"{dir}/{umap_file}", "rb") as f:
            params = pickle.load(f)
        proj_2d = params["projection"]
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5).fit(proj_2d)
        outdf = pd.DataFrame(proj_2d, columns=["0", "1"])
        outdf["labels"] = clusterer.labels_

        badness = []
        UMAP_goodness = outdf.groupby("labels").count().reset_index().drop(columns="1")
        if (
            UMAP_goodness[UMAP_goodness["labels"] != -1]["0"].sum()
            < UMAP_goodness[UMAP_goodness["labels"] == -1]["0"].sum()
        ):
            badness.append(params["hyperparameters"])

        num_clusters = len(np.unique([x for x in clusterer.labels_ if x != -1]))

        # colorscale = md.custom_color_scale()
        # if num_clusters > len(colorscale)-1:
        colorscale = px.colors.qualitative.Alphabet

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
                    # line=dict(width=0.2, color="Black"),
                ),
                hovertemplate=df["labels"],
            ),
            row=row,
            col=col,
        )
        df = outdf[outdf["labels"] != -1]
        fig.add_trace(
            go.Scatter(
                x=df["0"],
                y=df["1"],
                mode="markers",
                marker=dict(
                    size=4,
                    color=df["labels"],
                    colorscale=colorscale,
                    # line=dict(width=0.05, color="Black"),
                    cmid=0.3,
                ),
                hovertemplate=df["labels"],
                hoverinfo=["all"],
            ),
            row=row,
            col=col,
        )
        row += 1
        if row == len(dists) + 1:
            row = 1
            col += 1

    fig.update_layout(
        # height=900,
        template="simple_white",
        showlegend=False,
        font=dict(color="black"),
        title="Projection Gridsearch Plot",
    )

    fig.update_xaxes(showticklabels=False, tickwidth=0, tickcolor="rgba(0,0,0,0)")
    fig.update_yaxes(showticklabels=False, tickwidth=0, tickcolor="rgba(0,0,0,0)")

    return fig


def create_cluster_distribution_histogram(dir):
    """
    Create a histogram of the cluster distribution.

    Parameters:
    -----------
    dir : str
        The directory containing the projections.

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The Plotly figure object representing the histogram.
    """

    cluster_distribution = []
    for umap in os.listdir(dir):
        with open(f"{dir}/{umap}", "rb") as f:
            params = pickle.load(f)
        proj_2d = params["projection"]
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5).fit(proj_2d)
        num_clusters = len(np.unique(clusterer.labels_))
        cluster_distribution.append(num_clusters)

    unique_values, value_counts = np.unique(cluster_distribution, return_counts=True)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=unique_values,
            y=value_counts,
            marker_color=px.colors.qualitative.Alphabet,
            text=value_counts,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        xaxis=dict(tickmode="array", tickvals=unique_values, ticktext=unique_values),
        bargap=0.05,
        # yaxis=dict(autorange="reversed"),
        template="simple_white",
        showlegend=False,
        xaxis_title="Number of HDBSCAN Clusters",
        font=dict(color="black"),
        title="HDBSCAN Cluster Histogram",
    )

    return fig


def find_min_max_values(data, buffer_percent=5):
    """
    Find the minimum and maximum values for each column (first and second number) in a list of sublists and apply a buffer.

    Parameters:
    -----------
    data : list
        List of sublists containing numeric values.
    buffer_percent : float, optional
        Percentage value for the buffer/padding. Default is 5%.

    Returns:
    --------
    tuple
        Two lists, each with two values representing the minimum and maximum values of each column including the buffer.

    Raises:
    -------
    ValueError
        If the input data is empty.
    TypeError
        If the input data is not a list of sublists or the sublists do not have two elements.
    """

    if not data:
        raise ValueError("Input data is empty.")

    if not isinstance(data, list) or not all(
        isinstance(sublist, list) and len(sublist) == 2 for sublist in data
    ):
        raise TypeError("Input data should be a list of sublists with two elements.")

    column1 = [sublist[0] for sublist in data]
    column2 = [sublist[1] for sublist in data]

    min_col1 = min(column1)
    max_col1 = max(column1)

    min_col2 = min(column2)
    max_col2 = max(column2)

    x_buffer = (max_col1 - min_col1) * buffer_percent / 100
    y_buffer = (max_col2 - min_col2) * buffer_percent / 100

    min_col1 -= x_buffer
    max_col1 += x_buffer
    min_col2 -= y_buffer
    max_col2 += y_buffer

    return [min_col1, max_col1], [min_col2, max_col2]


def analyze_umap_projections(dir):
    """
    Analyze UMAP projections in a directory and visualize the results.

    Parameters:
    -----------
    dir : str
        The directory path containing UMAP projections.

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The Plotly figure object representing the scatter plot of UMAP projections with corresponding badness scores.

    Raises:
    -------
    FileNotFoundError
        If the specified directory does not exist.
    """

    # Create an empty list to store badness scores
    badness = []
    params_list = []

    # Iterate over the UMAP files in the specified directory
    for umap in os.listdir(dir):
        umap_path = os.path.join(dir, umap)

        # Load UMAP parameters from file
        with open(umap_path, "rb") as f:
            params = pickle.load(f)

        proj_2d = params["projection"]

        # Cluster the UMAP projections using HDBSCAN
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5).fit(proj_2d)

        # Create a DataFrame with the projection coordinates and cluster labels
        outdf = pd.DataFrame(proj_2d, columns=["0", "1"])
        outdf["labels"] = clusterer.labels_

        # Calculate the UMAP goodness metric by counting unclustered items
        UMAP_goodness = outdf.groupby("labels").count().reset_index().drop(columns="1")

        params_list.append(params["hyperparameters"][:2])

        # Calculate the badness score for the current UMAP projection
        if (
            UMAP_goodness[UMAP_goodness["labels"] != -1]["0"].sum()
            < UMAP_goodness[UMAP_goodness["labels"] == -1]["0"].sum()
        ):
            badness.append(
                (
                    tuple(params["hyperparameters"][:2]),
                    UMAP_goodness[UMAP_goodness["labels"] == -1]["0"].sum(),
                )
            )

    fig = go.Figure()
    x = [item[0] for item in params_list]
    y = [item[1] for item in params_list]
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(color="white", line=dict(width=0.3, color="black")),
            name="All Parameters",
        )
    )

    # Extract data for plotting
    data = badness
    x = [item[0][0] for item in data]
    y = [item[0][1] for item in data]
    colors = [item[1] for item in data]

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Unclustered UMAP parameters",
            hovertemplate=colors,
            marker=dict(
                color=colors,
                colorscale="Bluered",
                # cmin=min(colors),
                # cmax=max(colors),
                colorbar=dict(
                    title="Number of Unclustered Items<br>per Projection",
                    len=0.4,  # Adjust the length of the colorbar
                    y=0.65,  # Adjust the position of the colorbar
                    yanchor="middle",
                ),
            ),
        )
    )

    # Customize the plot layout
    fig.update_layout(
        template="simple_white",
        xaxis_title="n_neighbors",
        yaxis_title="min_dist",
        yaxis_range=find_min_max_values(params_list)[1],
        xaxis_range=find_min_max_values(params_list)[0],
        title="Parameters creating UMAPs with more <i>Unclustered</i> items than <i>Clustered</i>",
        # yaxis=dict(autorange="reversed"),
    )

    # Display the plot
    return fig


def save_visualizations_as_html(visualizations, output_file):
    """
    Saves a list of Plotly visualizations as an HTML file.

    Parameters:
    -----------
    visualizations : list
        A list of Plotly visualizations (plotly.graph_objects.Figure).
    output_file : str
        The path to the output HTML file.
    """

    # Create the HTML file and save the visualizations
    with open(output_file, "w") as f:
        f.write("<html>\n<head>\n</head>\n<body>\n")
        for i, viz in enumerate(visualizations):
            div_str = pio.to_html(viz, full_html=False, include_plotlyjs="cdn")
            f.write(f'<div id="visualization{i+1}">{div_str}</div>\n')
        f.write("</body>\n</html>")
