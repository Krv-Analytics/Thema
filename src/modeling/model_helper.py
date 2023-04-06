"Helper functions for Mapper Policy Model"

import datetime
import os
import sys

from dotenv import load_dotenv


load_dotenv()
root = os.getenv("root")
src = os.getenv("src")
sys.path.append(src)


def mapper_plot_outfile(
    hyper_parameters,
):
    n, p, nbors, d, hdbscan_params = hyper_parameters
    output_file = f"mapper_ncubes{n}_{int(p*100)}perc_hdbscan{hdbscan_params[0]}_UMAP_{nbors}Nbors_minD{d}.html"
    output_dir = os.path.join(root, "data/visualizations/mapper_htmls/")

    if os.path.isdir(output_dir):
        output_file = os.path.join(output_dir, output_file)
    else:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, output_file)

    return output_file


def config_plot_data(tupper):
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
