import argparse
import os
import sys

import plotly.io as pio
from omegaconf import OmegaConf
from projection_summarizer_helper import (
    analyze_umap_projections,
    create_cluster_distribution_histogram,
    create_umap_grid,
    save_visualizations_as_html,
)

from . import env

pio.renderers.default = "browser"
root, src = env()  # Load .env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    YAML_PATH = os.getenv("params")
    if os.path.isfile(YAML_PATH):
        with open(YAML_PATH, "r") as f:
            params = OmegaConf.load(f)
    else:
        print("params.yaml file note found!")

    parser.add_argument(
        "-d",
        "--projection_directory",
        type=str,
        default=params["projected_data"],
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

    output_file = "ProjectionSummary.html"
    output_dir = "data/" + params["Run_Name"] + "/plots/"

    if os.path.isdir(output_dir):
        output_file = os.path.join(output_dir, output_file)

    else:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, output_file)

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
