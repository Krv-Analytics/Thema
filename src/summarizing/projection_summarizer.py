
from dotenv import load_dotenv
import os
import sys
import plotly.io as pio

import argparse
import json

from projection_summarizer_helper import (
    create_cluster_distribution_histogram,
    save_visualizations_as_html,
    create_umap_grid,
    analyze_umap_projections
)

pio.renderers.default = "browser"
load_dotenv
root = os.getenv("root")
src = os.getenv("src")
sys.path.append(src)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    root = os.getenv("root")
    JSON_PATH = os.getenv("params")
    if os.path.isfile(JSON_PATH):
        with open(JSON_PATH, "r") as f:
            params_json = json.load(f)
    else:
        print("params.json file note found!")

    parser.add_argument(
        "-d",
        "--projection_directory",
        type=str,
        default=params_json["projected_data"],
        help="Specify the directory containing the projections you would like to summarize",
    )

    parser.add_argument(
        "-p",
        "--projection_viz",
        type=bool,
        default=True,
        help="Specify if you want a projection visualization included in your out file",
    )

    parser.add_argument(
        "-D",
        "--DBSCAN_cluster_histogram",
        type=bool,
        default=True,
        help="Specify if you want a HDBSCAN projection cluster histogram included in your out file",
    )

    parser.add_argument(
        "-E",
        "--projection_evaluation",
        type=bool,
        default=True,
        help="Specify if you want a metric of projection coverage included in your outfile",
    )
   
    parser.add_argument(
        "-v",
        "--Verbose",
        default=True,
        action="store_true",
        help="If set, will print messages detailing computation and output.",
    )

    args = parser.parse_args()
    this = sys.modules[__name__]

    fig_list = []
    dir = args.projection_directory

    if args.projection_viz:
        fig_list.append(create_umap_grid(dir))

    if args.DBSCAN_cluster_histogram:
        fig_list.append(create_cluster_distribution_histogram(dir))

    if args.projection_evaluation:
        fig_list.append(analyze_umap_projections(dir))

    output_file = 'ProjectionSummary.html'
    
    save_visualizations_as_html(fig_list, output_file)

    if args.Verbose:
        print("\n")
        print(
            "-------------------------------------------------------------------------------- \n\n"
        )
        print(f"The visualizations have been saved to {output_file}.")

        print(
            "\n\n -------------------------------------------------------------------------------- "
        )
