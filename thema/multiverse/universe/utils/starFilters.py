from typing import Callable
import networkx as nx


def nofilterfunction(graphobject) -> int:
    """Default filter that accepts all graph objects."""
    return 1


def component_count_filter(target_components: int) -> Callable:
    """Filter for graphs with specific number of connected components.

    Args:
        target_components: Desired number of connected components

    Returns:
        Filter function that returns 1 for matching graphs, 0 otherwise

    Example:
        >>> filter_func = component_count_filter(4)
        >>> galaxy.collapse(filter_fn=filter_func)
    """

    def _filter(graphobject) -> int:
        if graphobject.starGraph is None:
            return 0
        return (
            1
            if nx.number_connected_components(graphobject.starGraph.graph)
            == target_components
            else 0
        )

    return _filter


def component_count_range_filter(
    min_components: int, max_components: int
) -> Callable:
    """Filter for graphs within component count range.

    Args:
        min_components: Minimum number of components (inclusive)
        max_components: Maximum number of components (inclusive)

    Returns:
        Filter function that returns 1 for graphs in range, 0 otherwise
    """

    def _filter(graphobject) -> int:
        if graphobject.starGraph is None:
            return 0
        n_components = nx.number_connected_components(
            graphobject.starGraph.graph
        )
        return 1 if min_components <= n_components <= max_components else 0

    return _filter


def minimum_nodes_filter(min_nodes: int) -> Callable:
    """Filter for graphs with minimum number of nodes.

    Args:
        min_nodes: Minimum number of nodes required

    Returns:
        Filter function that returns 1 for graphs meeting criteria, 0 otherwise
    """

    def _filter(graphobject) -> int:
        if graphobject.starGraph is None:
            return 0
        return (
            1
            if graphobject.starGraph.graph.number_of_nodes() >= min_nodes
            else 0
        )

    return _filter


def minimum_edges_filter(min_edges: int) -> Callable:
    """Filter for graphs with minimum number of edges.

    Args:
        min_edges: Minimum number of edges required

    Returns:
        Filter function that returns 1 for graphs meeting criteria, 0 otherwise
    """

    def _filter(graphobject) -> int:
        if graphobject.starGraph is None:
            return 0
        return (
            1
            if graphobject.starGraph.graph.number_of_edges() >= min_edges
            else 0
        )

    return _filter


def minimum_unique_items_filter(min_unique_items: int) -> Callable:
    """Filter for graphs with minimum number of unique items across all nodes.

    This filter counts the total number of unique data points present
    across all nodes in the Mapper graph, ensuring no double-counting
    of items that appear in multiple nodes.

    Args:
        min_unique_items: Minimum number of unique items required

    Returns:
        Filter function that returns 1 for graphs meeting criteria, 0 otherwise

    Example:
        >>> filter_func = minimum_unique_items_filter(100)
        >>> galaxy.collapse(filter_fn=filter_func)
    """

    def _filter(graphobject) -> int:
        if graphobject.starGraph is None:
            return 0

        # Collect all unique items from node membership lists
        unique_items = set()
        for node in graphobject.starGraph.graph.nodes():
            membership = graphobject.starGraph.graph.nodes[node]["membership"]
            unique_items.update(membership)

        return 1 if len(unique_items) >= min_unique_items else 0

    return _filter
