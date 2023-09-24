import math
import ipywidgets as widgets
from ipywidgets import interact
from IPython.display import display, clear_output
import matplotlib.pyplot as plt


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

def interactive_visualization(token):
    # Create a dropdown widget for choosing the coloring parameter with custom style
    options_list = list(token.raw.columns)
    options_list.insert(0, None)
    col_dropdown = widgets.Dropdown(
        options=options_list,
        description='Color By:',
        disabled=False,
         style={'description_width': '120px', 'width': '200px', 'background-color': 'lightgray'}
    )

    node_size_dropdown = widgets.Dropdown(
        options=options_list,
        description='Size Nodes By:',
        disabled=False,
         style={'description_width': '120px', 'width': '200px', 'background-color': 'lightgray'}
    )

    # Create a dropdown widget for choosing the node_color_method with custom style
    color_method_dropdown = widgets.Dropdown(
        options=['average', 'sum', 'min', 'max', 'std'],
        description='Color Method:',
        disabled=False,
        style={'description_width': '120px', 'width': '200px', 'button_color': 'success'}
    )

    node_size_metric_dropdown = widgets.Dropdown(
        options=['sum', 'mean', 'median'],
        description='Node Size Method:',
        disabled=False,
        style={'description_width': '120px', 'width': '200px', 'button_color': 'success'},
        flex='0 0 auto', width='auto', min_width='100px'
    )

    # Create toggle buttons for various options with a red button style
    legend_toggle_button = widgets.ToggleButton(
        value=False,
        description='Legend Bar',
        disabled=False,
        style={'button_color': 'white', 'color': 'black'},
        tooltip='Toggle Legend Bar'  # Add a tooltip
    )

    group_label_toggle = widgets.ToggleButton(
        value=False,
        description='Group Labels',
        disabled=False,
        style={'button_color': 'white', 'color': 'black'},
        tooltip='Toggle Group Labels'  # Add a tooltip
    )

    node_label_toggle = widgets.ToggleButton(
        value=False,
        description='Node Labels',
        disabled=False,
        style={'button_color': 'white', 'color': 'black'},
        tooltip='Toggle Node Labels'  # Add a tooltip
    )

    show_edge_weights_toggle = widgets.ToggleButton(
        value=False,
        description='Show Edge Weights',
        disabled=False,
        style={'button_color': 'white', 'color': 'black'},
        tooltip='Toggle Edge Weights'  # Add a tooltip
    )

    # Create a slider widget for the k parameter
    k_slider = widgets.FloatSlider(
        value=0.1,
        min=0.05,
        max=0.5,
        step=0.01,
        description='k:',
        continuous_update=False,
    )

    # Create a slider widget for the spring_layout_seed parameter
    seed_slider = widgets.IntSlider(
        value=8,
        min=1,
        max=20,
        description='Seed:',
        continuous_update=False,
    )

    # Define an output widget for displaying the graph
    output = widgets.Output()

    left_top_box = widgets.VBox([legend_toggle_button, group_label_toggle, node_label_toggle, show_edge_weights_toggle])
    top_center_box = widgets.VBox([col_dropdown, color_method_dropdown, node_size_dropdown, node_size_metric_dropdown])
    top_right_box = widgets.VBox([k_slider, seed_slider], layout=widgets.Layout(justify_content='flex-end'))
    
    # Stack the layout containers horizontally in the main box
    main_box = widgets.HBox([top_right_box, top_center_box, left_top_box], layout=widgets.Layout(justify_content='flex-end'))
    
    container = widgets.VBox([main_box, output])

    # Define a function to visualize the graph based on the selected values
    def visualize_graph(change):
        with output:
            clear_output(wait=True)
            token.visualize_model(
                col=col_dropdown.value,
                node_color_method=color_method_dropdown.value,
                legend_bar=legend_toggle_button.value,
                group_labels=group_label_toggle.value,
                node_labels=node_label_toggle.value,
                show_edge_weights=show_edge_weights_toggle.value,
                k=k_slider.value,
                spring_layout_seed=seed_slider.value,
                node_size_col = node_size_dropdown.value,
                node_size_aggregation_method=node_size_metric_dropdown.value,
                node_size_multiplier=5
            )
            plt.show()

    # Use the observe method to connect the dropdowns, buttons, and sliders to the visualization function
    col_dropdown.observe(visualize_graph, names='value')
    color_method_dropdown.observe(visualize_graph, names='value')
    legend_toggle_button.observe(visualize_graph, names='value')
    group_label_toggle.observe(visualize_graph, names='value')
    node_label_toggle.observe(visualize_graph, names='value')
    show_edge_weights_toggle.observe(visualize_graph, names='value')
    k_slider.observe(visualize_graph, names='value')
    seed_slider.observe(visualize_graph, names='value')
    node_size_dropdown.observe(visualize_graph, names='value')
    node_size_metric_dropdown.observe(visualize_graph, names='value')

    # Display the widget container
    display(container)