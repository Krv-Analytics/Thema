"Helper functions for visualizing CoalMappers"

import time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.cluster.hierarchy import dendrogram


from umap import UMAP
import hdbscan


from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.decomposition import PCA


pio.renderers.default = "browser"


def plot_dendrogram(model, labels, distance, p, n, **kwargs):
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
        labels=None,
        p=p,
        truncate_mode="level",
    )
    plt.title(f"Clustering Models with {n} Policy Groups")
    plt.xlabel("Coordinates: Model Parameters.")
    plt.ylabel(f"{distance} distance between persistence diagrams")
    plt.show()
    return d


def connected_component_heatmaps(self):
    viz = self.mapper.data

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


def UMAP_grid(df, dists, neighbors):
    """function reads in a df, outputs a grid visualization with n by n UMAP projected dataset visualizations

    grid search the UMAP parameter space, choose the representations that occur most often in the given parameter space, based on the generated histogram"""
    # example function inputs
    # dists = [0, 0.01, 0.05, 0.1, 0.5, 1]
    # neighbors = [3, 5, 10, 20, 4_0]

    # TODO
    # marl outlying points in a different color - check the cluster_distribution output
    # figure out a way around this .dropna() call that removes all rows with missing data
    data = df.dropna()
    assert type(dists) == list, "Not list"
    assert type(neighbors) == list, "Not list"
    print(f"Visualizing UMAP Grid Search! ")
    print(
        "--------------------------------------------------------------------------------"
    )
    print(f"Choices for n_neighbors: {neighbors}")
    print(f"Choices for m_dist: {dists}")
    print(
        "-------------------------------------------------------------------------------- \n"
    )

    # define custom color scale
    map_colors = [
        [0.0, "#043e4a"],
        [0.1, "#005f73"],
        [0.2, "#0a9396"],
        [0.3, "#94d2bd"],
        [0.4, "#e9d8a6"],
        [0.5, "#ee9b00"],
        [0.6, "#ca6702"],
        [0.7, "#bb3e03"],
        [0.8, "#e82f1e"],
        [0.9, "#a30309"],
        [1.0, "#780a23"],
    ]

    # generate subplot titles
    fig = make_subplots(
        rows=len(dists),
        cols=len(neighbors),
        column_titles=list(map(str, neighbors)),
        x_title="n_neighbors",
        row_titles=list(map(str, dists)),
        y_title="min_dist",
        vertical_spacing=0.05,
        horizontal_spacing=0.03,
        # shared_xaxes=True,
        # shared_yaxes=True,
    )
    cluster_distribution = []
    # generate figure
    for d in range(0, len(dists)):
        for n in range(0, len(neighbors)):
            umap_2d = UMAP(
                min_dist=dists[d],
                n_neighbors=neighbors[n],
                n_components=2,
                init="random",
                random_state=0,
            )
            proj_2d = umap_2d.fit_transform(data)
            clusterer = hdbscan.HDBSCAN(min_cluster_size=5).fit(proj_2d)
            outdf = pd.DataFrame(proj_2d, columns=["0", "1"])
            outdf["labels"] = clusterer.labels_

            num_clusters = len(np.unique(clusterer.labels_))
            cluster_distribution.append(num_clusters)
            df = outdf[outdf["labels"] != -1]
            fig.add_trace(
                go.Scatter(
                    x=df["0"],
                    y=df["1"],
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=df["labels"],
                        cmid=0.5,
                        colorscale="Turbo",  # colorscale=map_colors
                    ),
                    hovertemplate=df["labels"],
                ),
                row=d + 1,
                col=n + 1,
            )

            df = outdf[outdf["labels"] == -1]
            fig.add_trace(
                go.Scatter(
                    x=df["0"],
                    y=df["1"],
                    mode="markers",
                    marker=dict(
                        size=4,
                        color="yellow",
                        line=dict(width=0.1, color="DarkSlateGrey"),
                    ),
                    hovertemplate=df["labels"],
                ),
                row=d + 1,
                col=n + 1,
            )

    fig.update_layout(
        template="simple_white", showlegend=False, font=dict(color="black")
    )

    fig.update_xaxes(showticklabels=False, tickwidth=0, tickcolor="rgba(0,0,0,0)")
    fig.update_yaxes(showticklabels=False, tickwidth=0, tickcolor="rgba(0,0,0,0)")

    file_name = f"figures/UMAPgrid_min_dist({dists[0]}-{dists[len(dists)-1]})_neigh({neighbors[0]}-{neighbors[len(neighbors)-1]}).html"
    fig.write_html(file_name)
    pio.show(fig)

    # ax = sns.histplot(
    #     cluster_distribution,
    #     discrete=True,
    #     stat="percent",
    # )
    # ax.set(xlabel="Num_Clusters based on HDBSCAN", Num_Clusters based on HDBSCAN)
    # plt.show()

    colors = pd.DataFrame(cluster_distribution, columns=["Num"])
    fig2 = px.histogram(
        colors,
        x="Num",
        color="Num",
        nbins=max(cluster_distribution),
        color_discrete_map={
            1: "#005f73",
            2: "#0a9396",
            3: "#94d2bd",
            4: "#e9d8a6",
            5: "#ee9b00",
            6: "#ca6702",
            7: "#bb3e03",
            8: "#e82f1e",
        },
    )
    fig2.update_layout(
        template="simple_white",
        showlegend=False,
        xaxis_title="Num_Clusters based on HDBSCAN",
        title="Cluster Histogram",
    )
    pio.show(fig2)

    return file_name


#### IMPORTANT FUNCTIONS
def ORISPLmerge(df1, df2):
    return df1.merge(df2, how="left", left_on="ORISPL", right_on="ORISPL")


######
#### OTHER FUNCTIONS
######


def coal_PCA(df: str):
    """
    Graphs a Principle component analysis with 2 principle components
    df can only consist of numbers, no words/labels. Ensure you supply a subset of the dataset that conforms to this restriction"""
    x = df
    y = StandardScaler().fit_transform(x)

    # apply PCA
    pca = PCA(n_components=2)
    X = pca.fit_transform(y)

    loadings = pd.DataFrame(
        pca.components_.T, columns=["PC1", "PC2"], index=list(x.columns)
    )
    loadings

    def loading_plot(coeff, labels):
        n = coeff.shape[0]
        for i in range(n):
            plt.arrow(
                0,
                0,
                coeff[i, 0],
                coeff[i, 1],
                head_width=0.05,
                head_length=0.05,
                color="#21918C",
                alpha=0.5,
            )
            plt.text(
                coeff[i, 0] * 1.15,
                coeff[i, 1] * 1.15,
                labels[i],
                color="#21918C",
                ha="center",
                va="center",
            )
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid()

    fig, ax = plt.subplots(figsize=(7, 7))
    loading_plot(pca.components_.T, list(x.columns))


def mapper_labels(df: pd.DataFrame):

    """This function takes in dataframe to use as the data labels for the mapper.
    Returns a dataframe
    """

    names = df.to_numpy()
    return names


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


def percentCO2reduxMap(mapdf: pd.DataFrame, all: pd.DataFrame):
    def make_chloropheth(subset: pd.DataFrame, all: pd.DataFrame):
        state_total = (
            all[all["decom"] == 0].groupby("PSTATABB")["PLCO2AN"].sum().reset_index()
        )
        state_reduction = (
            subset[subset["decom"] == 0]
            .groupby("PSTATABB")["PLCO2AN"]
            .sum()
            .reset_index()
        )
        state_total = state_total.merge(
            state_reduction, how="left", on="PSTATABB"
        ).fillna(0)
        state_total["percentRedux"] = (
            state_total["PLCO2AN_y"] / state_total["PLCO2AN_x"]
        )
        return state_total

    state_total = make_chloropheth(mapdf, all)

    num = len(mapdf)
    redux = int(
        100 * (mapdf["PLCO2AN"].sum() / all[all["decom"] == 0]["PLCO2AN"].sum())
    )

    fig = go.Figure(
        data=[
            go.Choropleth(
                locations=state_total["PSTATABB"],  # Spatial coordinates
                z=state_total["percentRedux"].astype(float),  # Data to be color-coded
                locationmode="USA-states",  # set of locations match entries in `locations`
                colorscale="earth",
                colorbar_title="% State Reduction",
            ),
            go.Scattergeo(
                lon=mapdf["LON"],
                lat=mapdf["LAT"],
                text=mapdf["label"],
                marker=dict(
                    color=list(range(0)),
                    colorscale="viridis",
                    size=mapdf["PLNGENAN"] / 400000,
                ),
                # marker_color = df['CO2limits'],
            ),
        ]
    )

    fig.update_layout(
        title_text=f"{num} Plant Closures - {redux}% CO2 Reduction Nationwide",
        geo_scope="usa",  # limite map scope to USA
    )

    fig.show()
