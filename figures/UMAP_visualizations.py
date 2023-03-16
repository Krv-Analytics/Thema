import argparse
import os
import sys
import pandas as pd
import pickle

from umap import UMAP
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"


def UMAP_grid(df, dists, neighbors):
    """function reads in a df, outputs a grid visualization with n by n UMAP projected dataset visualizations"""
    # example function inputs
    # dists = [0, 0.01, 0.05, 0.1, 0.5, 1]
    # neighbors = [3, 5, 10, 20, 40]

    # TODO
    # make the colors nicer/give more meaning here
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
        shared_xaxes=True,
        shared_yaxes=True,
    )

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
            outdf = pd.DataFrame(proj_2d, columns=["0", "1"])
            fig.add_trace(
                go.Scatter(
                    x=outdf["0"],
                    y=outdf["1"],
                    mode="markers",
                    marker=dict(size=3, color="red"),
                ),
                row=d + 1,
                col=n + 1,
            )

    fig.update_layout(
        template="simple_white", showlegend=False, font=dict(color="black")
    )
    fig.update_xaxes(range=[-25, 25], showticklabels=False)
    fig.update_yaxes(range=[-25, 25], showticklabels=False)
    file_name = f"figures/UMAPgrid_min_dist({dists[0]}-{dists[len(dists)-1]})_neigh({neighbors[0]}-{neighbors[len(neighbors)-1]}).html"
    fig.write_html(file_name)
    pio.show(fig)
    return file_name


######################################################################################################
##############################################################################
###################################################

cwd = os.path.dirname(__file__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=os.path.join(cwd, "./../data/coal_mapper_one_hot_scaled.pkl"),
        help="Select location of local data set, as pulled from Mongo.",
    )

    parser.add_argument(
        "-n",
        "--neighbors_list",
        type=int,
        nargs="+",
        default=[3, 5, 10, 20, 40],
        help="Insert a list of n_neighbors to grid search",
    )

    parser.add_argument(
        "-d",
        "--min_dists",
        type=float,
        nargs="+",
        default=[0, 0.01, 0.05, 0.1, 0.5, 1],
        help="Insert a list of min_dists to grid search",
    )

    args = parser.parse_args()
    this = sys.modules[__name__]

    assert os.path.isfile(args.path), "Invalid Input Data"
    # Load Dataframe
    with open(args.path, "rb") as f:
        df = pickle.load(f)

    name = UMAP_grid(df, dists=args.min_dists, neighbors=args.neighbors_list)

    print(
        "\n################################################################################## \n\n"
    )
    print(
        f"Successfully created the UMAP Grid Search Visualization\n Saved in {name} for future reference!"
    )

    print(
        "\n\n##################################################################################\n"
    )