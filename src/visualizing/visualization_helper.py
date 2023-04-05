"Helper functions for visualizing CoalMappers"

import datetime
import os
import sys
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.io as pio
from dotenv import load_dotenv
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from persim import plot_diagrams
import matplotlib.pyplot as plot
import seaborn as sns

load_dotenv()
root = os.getenv("root")
src = os.getenv("src")
sys.path.append(src)

from processing.cleaning.tupper import Tupper
from modeling.coal_mapper import CoalMapper


def plot(
    coal_mapper: CoalMapper,
):
    """"""
    assert (
        len(coal_mapper.complex) > 0
    ), "First run `fit()` to generate a nonempty simplicial complex."

    path_html = mapper_plot_outfile(coal_mapper.n_cubes, coal_mapper.perc_overlap)

    numeric_data, labels = config_plot_data(coal_mapper.tupper)
    colorscale = custom_color_scale()

    coal_mapper.mapper.visualize(
        coal_mapper.complex,
        node_color_function=["mean", "median", "std", "min", "max"],
        color_values=numeric_data,
        color_function_name=labels,
        colorscale=colorscale,
        path_html=path_html,
    )
    print(f"Go to {path_html} for a visualization of your CoalMapper!")
    return path_html


def plot_curvature(coal_mapper: CoalMapper, bins="auto", kde=False):
    """Visualize Curvature of a mapper graph as a histogram."""

    ax = sns.histplot(
        coal_mapper.curvature,
        discrete=True,
        stat="probability",
        kde=kde,
        bins=bins,
    )
    ax.set(xlabel="Ollivier Ricci Edge Curvatures")

    return ax


def plot_diagrams(coal_mapper):
    """Visualize persistence diagrams of a mapper graph."""
    persim_diagrams = [
        np.asarray(coal_mapper.diagram[0]._pairs),
        np.asarray(coal_mapper.diagram[1]._pairs),
    ]
    return plot_diagrams(persim_diagrams, show=True)


def connected_component_heatmaps(coal_mapper):
    viz = coal_mapper.data

    targetCols = [
        "NAMEPCAP",
        "GENNTAN",
        "weighted_coal_CAPFAC",
        "weighted_coal_AGE",
        "Retrofit Costs",
        "forwardCosts",
        "PLSO2AN",
    ]

    # mean data
    viz2 = viz.groupby("cluster_labels", as_index=False).mean()[targetCols]
    c_labels = viz.groupby("cluster_labels").mean().reset_index()["cluster_labels"]
    scaler = MinMaxScaler()
    data = scaler.fit_transform(viz2)
    df = pd.DataFrame(data, columns=list(viz2.columns)).set_index(c_labels)

    #% of max data
    # TODO: fix the below to append 'cluster_labels' to targetCols so targetCols can be a user input
    df2 = viz[
        [
            "NAMEPCAP",
            "GENNTAN",
            "weighted_coal_CAPFAC",
            "weighted_coal_AGE",
            "Retrofit Costs",
            "forwardCosts",
            "PLSO2AN",
            "cluster_labels",
        ]
    ]
    df2 = df2.groupby(["cluster_labels"]).sum() / df2[targetCols].sum()

    fig = make_subplots(
        rows=1,
        cols=2,
        vertical_spacing=0.1,
        subplot_titles=(
            "Connected Component Averages",
            "Connected Component % of Total",
        ),
        specs=[[{"type": "Heatmap"}, {"type": "Heatmap"}]],
        y_title="Cluster Label",
    )

    col = 1
    for df in [df, df2]:
        if col == 1:
            text = viz2.round(2).values.tolist()
            texttemplate = "%{text}"
        else:
            text = df.round(2).values.tolist()
            texttemplate = ("%{text}") + "%"

        fig.add_trace(
            go.Heatmap(
                x=df.columns.to_list(),
                y=df.index.to_list(),
                z=df.values.tolist(),
                text=text,
                colorscale="rdylgn_r",
                xgap=2,
                ygap=2,
                texttemplate=texttemplate,
            ),
            row=1,
            col=col,
        )
        col = col + 1

    fig.update_xaxes(tickangle=45)
    fig.update_layout(font=dict(size=10))
    fig.update_traces(showscale=False)
    print(
        "Number of Plants in each Connected Component (-1 indicates not displayed on mapper):\n"
    )
    print(
        viz.groupby("cluster_labels")
        .count()
        .rename(columns={"NAMEPCAP": "#CoalPlants"})["#CoalPlants"]
    )
    pio.show(fig)


def plot_dendrogram(model, labels, distance, **kwargs):
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
    d = dendrogram(linkage_matrix, labels=labels, **kwargs)
    plt.title("Hyperparameter Dendrogram")
    plt.xlabel("Coordinates: (n_cubes,perc_overlap,min_intersection).")
    plt.ylabel(f"{distance} distance between persistence diagrams")
    plt.show()
    return d


def mapper_plot_outfile(n, p):
    time = int(datetime.datetime.now().timestamp())

    output_file = f"mapper_ncubes{n}_{int(p*100)}perc_{time}.html"
    output_dir = os.path.join(root, "data/visualizations/mapper_htmls/")

    if os.path.isdir(output_dir):
        output_file = os.path.join(output_dir, output_file)
    else:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, output_file)

    return output_file


def config_plot_data(tupper: Tupper):
    temp_data = tupper.clean
    string_cols = temp_data.select_dtypes(exclude="number").columns
    numeric_data = temp_data.drop(string_cols, axis=1).dropna()
    labels = list(numeric_data.columns)
    return numeric_data, labels


def custom_color_scale():
    colorscale = [
        [0.0, "#001219"],
        [0.1, "#005f73"],
        [0.2, "#0a9396"],
        [0.3, "#94d2bd"],
        [0.4, "#e9d8a6"],
        [0.5, "#ee9b00"],
        [0.6, "#ca6702"],
        [0.7, "#bb3e03"],
        [0.8, "#ae2012"],
        [0.9, "#9b2226"],
        [1.0, "#a50026"],
    ]
    return colorscale
