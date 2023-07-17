import math
import os


def get_subplot_specs(n):
    """
    Returns subplot specs based on the number of subplots.
    
    Parameters:
        n (int): number of subplots
        
    Returns:
        specs (list): 2D list of subplot specs
    """
    num_cols = 3
    num_rows = max(math.ceil(n / num_cols), 1)
    specs = [[{"type": "pie"} for c in range(num_cols)] for r in range(num_rows)]
    return specs




# TODO: Write a consistent and dynamic color assignment function 

def custom_color_scale():
    "Our own colorscale, feel free to use!"

    # colorscale = [
    # [0.0, "#001219"],
    # [0.04, "#004165"],
    # [0.08, "#0070b3"],
    # [0.12, "#00a1d6"],
    # [0.16, "#00c6eb"],
    # [0.20, "#00e0ff"],
    # [0.24, "#2cefff"],
    # [0.28, "#64ffff"],
    # [0.32, "#9bfff3"],
    # [0.36, "#ceffea"],
    # [0.40, "#e8ffdb"],
    # [0.44, "#f8ffb4"],
    # [0.48, "#ffff7d"],
    # [0.52, "#ffd543"],
    # [0.56, "#ffae00"],
    # [0.60, "#ff9000"],
    # [0.64, "#ff7300"],
    # [0.68, "#ff5500"],
    # [0.72, "#ff3500"],
    # [0.76, "#ff1600"],
    # [0.80, "#ff0026"],
    # [0.84, "#d70038"],
    # [0.88, "#b2004a"],
    # [0.92, "#8e0060"],
    # [0.96, "#690075"],
    # [1.0, "#a50026"]]

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


    extended_colorscale = []
    for i in range(100):
        t = i / 99.0
        for j in range(len(colorscale) - 1):
            if t >= colorscale[j][0] and t <= colorscale[j + 1][0]:
                r1, g1, b1 = (
                    colorscale[j][1][1:3],
                    colorscale[j][1][3:5],
                    colorscale[j][1][5:],
                )
                r2, g2, b2 = (
                    colorscale[j + 1][1][1:3],
                    colorscale[j + 1][1][3:5],
                    colorscale[j + 1][1][5:],
                )
                r = int(r1, 16) + int(
                    (t - colorscale[j][0])
                    / (colorscale[j + 1][0] - colorscale[j][0])
                    * (int(r2, 16) - int(r1, 16))
                )
                g = int(g1, 16) + int(
                    (t - colorscale[j][0])
                    / (colorscale[j + 1][0] - colorscale[j][0])
                    * (int(g2, 16) - int(g1, 16))
                )
                b = int(b1, 16) + int(
                    (t - colorscale[j][0])
                    / (colorscale[j + 1][0] - colorscale[j][0])
                    * (int(b2, 16) - int(b1, 16))
                )
                hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
                extended_colorscale.append([t, hex_color])
                break

    return colorscale

def reorder_colors(colors):
    n = len(colors)
    ordered = []
    for i in range(n):
        if i % 2 == 0:
            ordered.append(colors[i // 2])
        else:
            ordered.append(colors[n - (i // 2) - 1])
    return ordered




#####################################################################################
# 
#   (Unnecessary?) Kepler Mapper HTML Rendering Helper Functions 
#
#####################################################################################

#   NOTE: IF you would like to use this functionality, you will need to update  
#   the sys Path to import Tupper, and set the `root` variable from .env. 


# def config_plot_data(tupper: Tupper):
#     """ "Configure the data in a tupper to agree with KepplerMapper
#     visualizations.
#     NOTE: These visualizations are no longer maintained by KepplerMapper
#     and we do not reccomend using them.

#     Parameters
#     -----------
#     tupper: <tupper.Tupper>
#         A data container that holds raw, cleaned, and projected
#         versions of user data.

#     Returns
#     -----------
#     numeric_data: pd.DataFrame
#         Only numeric columns in the tupper.
#     labels : list
#         List of columns in `numeric_data`
#     """
#     temp_data = tupper.clean
#     string_cols = temp_data.select_dtypes(exclude="number").columns
#     numeric_data = temp_data.drop(string_cols, axis=1).dropna()
#     labels = list(numeric_data.columns)
#     return numeric_data, labels


# def mapper_plot_outfile(
#     hyper_parameters,
# ):
#     """Generate output filename for Kepler Mapper HTML Visualizations.
#     NOTE: These visualizations are no longer maintained by KepplerMapper
#     and we do not reccomend using them.

#     Parameters
#     -----------
#     hyper_parameters: list
#         A list of hyperparameters used to generate a particular Mapper.

#     Returns
#     -----------
#     output_file: str
#         A unique filename to identify JMapper Visualization.
#     """
#     root = env()
#     n, p, nbors, d, hdbscan_params, min_intersection = hyper_parameters
#     (
#         min_cluster_size,
#         max_cluster_size,
#     ) = hdbscan_params  # max_cluster size always set to zero, i.e. no upper bound
#     output_file = f"mapper_ncubes{n}_{int(p*100)}perc_hdbscan{min_cluster_size}_{max_cluster_size}UMAP_{nbors}Nbors_minD{d}_min_Intersection{min_intersection}.html"
#     output_dir = os.path.join(root, "data/visualizations/mapper_htmls/")

#     if os.path.isdir(output_dir):
#         output_file = os.path.join(output_dir, output_file)
#     else:
#         os.makedirs(output_dir, exist_ok=True)
#         output_file = os.path.join(output_dir, output_file)

#     return output_file
