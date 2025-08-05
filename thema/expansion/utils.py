def star_link(star_obj, node_features=None, group_features=None):
    """
    Connect a Star object to the Multiverse system by extracting its networkx graph
    for use with the Realtor class.

    This function extracts the networkx graph from a Star object's starGraph attribute
    and prepares it to be used with the Realtor class for spatial analysis.

    Parameters
    ----------
    star_obj : Star
        A Star object with an initialized starGraph attribute.
    node_features : array-like, optional
        Features for each node in the graph. If None, must be provided when creating
        the Realtor instance.
    group_features : array-like, optional
        Features for each group/component in the graph. If None, must be provided when
        creating the Realtor instance.

    Returns
    -------
    dict
        A dictionary containing:
        - 'graph': The networkx graph extracted from the Star object
        - 'node_features': The node features if provided
        - 'group_features': The group features if provided

    Example
    -------
    >>> from thema.multiverse.universe.star import Star
    >>> from thema.multiverse.universe.stars.jmapStar import JMapStar
    >>> from thema.expansion.realtor import Realtor
    >>> from thema.expansion.utils import star_link
    >>>
    >>> # Initialize and fit a JMapStar (a Star implementation)
    >>> jmap_star = JMapStar(data_path="path/to/data.csv",
    ...                       clean_path="path/to/clean.pkl",
    ...                       projection_path="path/to/projection.pkl")
    >>> jmap_star.fit()
    >>>
    >>> # Extract the graph and feature data for the Realtor
    >>> star_data = star_link(jmap_star,
    ...                        node_features=my_node_features,
    ...                        group_features=my_group_features)
    >>>
    >>> # Create a Realtor instance to find the best location for a target vector
    >>> target_vector = [0.1, 0.2, 0.3]  # Example target vector
    >>> realtor = Realtor(target_vector=target_vector,
    ...                    graph=star_data['graph'],
    ...                    node_features=star_data['node_features'],
    ...                    group_features=star_data['group_features'])
    >>>
    >>> # Find the best node for docking the target
    >>> best_node = realtor.node_docking()
    >>> print(f"The best node for the target is: {best_node}")
    >>>
    >>> # Alternatively, perform a random walk
    >>> samples = realtor.random_walk(n_samples=500, m_steps=800)
    >>> print(f"Random walk samples: {samples[:5]}...")
    """
    if not hasattr(star_obj, "starGraph") or star_obj.starGraph is None:
        raise ValueError(
            "Star object does not have an initialized starGraph attribute"
        )

    if not hasattr(star_obj.starGraph, "graph"):
        raise ValueError("starGraph does not contain a valid graph attribute")

    # Extract the networkx graph from the starGraph
    graph = star_obj.starGraph.graph

    return {
        "graph": graph,
        "node_features": node_features,
        "group_features": group_features,
    }
