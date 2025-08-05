# File: multiverse/universe/starGraph.py
# Last Update: 05/15/24
# Updated by: JW

import networkx as nx


class starGraph:
    """
    A graph wrapper to guide you through the stars!

    Parameters
    ----------
    graph : nx.Graph
        The graph object representing the star graph.

    Attributes
    ----------
    graph : nx.Graph
        The graph object representing the star graph.

    Methods
    -------
    is_EdgeLess()
        Check if the graph is edgeless.
    components()
        Get a list of connected components in the graph.
    get_MST(k=0, components=None)
        Calculate a customizable Minimum Spanning Tree of the weighted graph.
    get_shortest_path(nodeID_1, nodeID_2)
        Calculate the shortest path between two nodes
        in the graph using Dijkstra's algorithm.
    """

    def __init__(self, graph):
        """
        Initialize the starGraph class.

        Parameters
        ----------
        graph : nx.Graph
            The graph object representing the star graph.
        """
        self.graph = graph

    @property
    def is_EdgeLess(self):
        """
        Check if the graph is edgeless.

        Returns
        -------
        bool
            True if the graph is edgeless, False otherwise.
        """
        return len(self.graph.edges) == 0

    @property
    def components(self):
        """
        Get a list of connected components in the graph.

        Returns
        -------
        dict
            A dictionary where the keys are component indices and the values
            are subgraphs representing the connected components.
        """
        return dict(
            [
                (i, self.graph.subgraph(c).copy())
                for i, c in enumerate(nx.connected_components(self.graph))
            ]
        )

    def get_MST(self, k=0, components=None):
        """
        Calculate a customizable Minimum Spanning Tree of the weighted graph.

        Default is to return a minimum spanning tree for each connected
        component. If a `k` value is supplied that is greater than the number
        of connected components, then a minimum spanning forest of k trees will
        be returned (in the case that k is less than the number of connected
        components, then the default MST is returned).

        In the case that only certain components should be considered for
        further edge removal, then they may be specified in `components` and
        the `k` value should be supplied as a list.

        Parameters
        ----------
        k : int or list, optional
            The number of trees in the minimum spanning forest. Note that k is ignored if it
            is less than the number of connected components. Default is 0.
        components : int or list, optional
            The connected components that are to be split. Default is None.

        Returns
        -------
        nx.Graph
            The minimum spanning tree or forest of the weighted graph.
        """
        # Calculate simple MST
        mst = nx.minimum_spanning_tree(self.graph, weight="weight")
        # Handle no components case
        if components is None:
            if k <= nx.number_connected_components(self.graph):
                return mst
            else:
                k = k - nx.number_connected_components(self.graph)
                # Sort edges by weight
                sorted_edges = sorted(
                    mst.edges(data=True), key=lambda x: x[2]["weight"]
                )

                for edge in sorted_edges[-k:]:
                    mst.remove_edge(edge[0], edge[1])

                return mst
        # Handle component specific
        else:
            # Cast ints to list
            if type(components) == int:
                components = [components]
            if type(k) == int:
                k = [k]

            # List k must be same length as component list
            assert len(k) == len(
                components
            ), "Length of k must be equal to length of components"

            mst = nx.Graph()

            for i in range(len(self._components)):
                cc_mst = nx.minimum_spanning_tree(
                    self._components[i], weight="weight"
                )

                # Component is to be split into specified number of groups
                if i in components:
                    j = components.index(i)
                    cc_sorted_edges = sorted(
                        cc_mst.edges(data=True), key=lambda x: x[2]["weight"]
                    )

                    # Check number of edges is greater than k value
                    assert k[j] < len(
                        cc_sorted_edges
                    ), f"k value for component {i} was greater than number of edges in component. "

                    # k value should split component if it is given
                    assert k[j] > 1, "Please supply k values greater than 1. "

                    # Remove k-1 edges to result in k components
                    for edge in cc_sorted_edges[-k[j] + 1 :]:
                        cc_mst.remove_edge(edge[0], edge[1])

                # Combine component MSTs
                mst = nx.union(cc_mst, mst)
            return mst

    def get_shortest_path(self, nodeID_1, nodeID_2):
        """
        Calculate the shortest path between two nodes in the graph using Dijkstra's algorithm.

        Parameters
        ----------
        nodeID_1 : int or str
            The identifier of the source node.
        nodeID_2 : int or str
            The identifier of the target node.

        Returns
        -------
        tuple
            A tuple containing:
            - A list representing the nodes in the shortest path from nodeID_1 to nodeID_2.
            - The length of the shortest path, considering edge weights.
            If no path exists between the nodes, returns (None, infinity).
        """
        try:
            # Calculate shortest path using Dijkstra's algorithm
            shortest_path = nx.shortest_path(
                self.graph, source=nodeID_1, target=nodeID_2, weight="weight"
            )

            # Calculate the length of the shortest path
            path_length = sum(
                self.graph[shortest_path[i]][shortest_path[i + 1]]["weight"]
                for i in range(len(shortest_path) - 1)
            )

            return shortest_path, path_length

        except nx.NetworkXNoPath:
            return None, float(
                "inf"
            )  # No path exists, return None and infinity length
