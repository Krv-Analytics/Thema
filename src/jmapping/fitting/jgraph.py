import networkx as nx
import kmapper as km
import numpy as np

from tupper import Tupper
from nerve import Nerve


from nammu.curvature import ollivier_ricci_curvature
from nammu.topology import PersistenceDiagram, calculate_persistence_diagrams
from nammu.utils import make_node_filtration


class JGraph:
    """
    A Graph Class to Handle all of your Curvature, Homology and Graph learning needs!

    """

    def __init__(self, nodes: dict(), weighted: bool = True, min_intersection=None):
        """
        Constructor for the JGraph Class.

        Parameters:
        -----------
        nodes: <dict()>
            The nodes of a simplicial complex

        weighted: bool
            If true, return a weighted graph based on node intersection.

        min-intersection:
            For the moment, min_intersection value as dictated by graph nerve

        """

        assert (
            len(nodes) > 0
        ), "You must first generate a non-empty Simplicial Complex \
        with `fit()` before you can convert to Networkx "

        self.weighted = weighted
        self.min_intersection = min_intersection
        self.graph = nx.Graph()
        self.nerve = Nerve(
            weighted=self.weighted, min_intersection=self.min_intersection
        )

        # Fit Nerve to generate edges
        edges = self.nerve.compute(nodes)

        if len(edges) == 0:
            self._is_Edgeless = True
        else:
            self._is_Edgeless = False
            self.graph.add_nodes_from(nodes)
            nx.set_node_attributes(self.graph, nodes, "membership")
            if self.weighted:
                self.graph.add_weighted_edges_from(edges)
            else:
                self.graph.add_edges_from(edges)

        self._components = dict(
            [
                (i, self.graph.subgraph(c).copy())
                for i, c in enumerate(nx.connected_components(self.graph))
            ]
        )

        self._curvature = np.array([])
        self._diagram = PersistenceDiagram()

    ####################################################################################################
    #
    #   Properties
    #
    ####################################################################################################

    @property
    def is_EdgeLess(self):
        return self._is_Edgeless

    @property
    def components(self):
        return self._components

    @property
    def curvature(self):
        """Return the curvature values for the graph of a JMapper object."""
        assert len(self._curvature) > 0, "You don't have any edge curvatures!"
        return self._curvature

    @curvature.setter
    def curvature(self, curvature_fn=ollivier_ricci_curvature):
        """Setter function for curvature.

        Parameters
        -----------
        curvature_fn: func
            The method for calculating discrete curvature of the graph.
            The default is set to Ollivier-Ricci Curvature.

        """
        weight = None
        if self.weighted:
            weight = "weight"

        try:
            curvature = curvature_fn(self.graph, weight=weight)
            assert len(curvature) == len(self.graph.edges())
            self._curvature = curvature
        except len(curvature) != len(self.graph.edges()):
            print("Invalid Curvature function")

    @property
    def diagram(self):
        """Return the persistence diagram based on curvature filtrations
        associated with JMapper graph."""
        if self._diagram is None:  # TODO: Change! diagram is never initialized to none
            try:
                self.calculate_homology()
            except self.complex == dict():
                print(
                    "Persistence Diagrams could not be obtained\
                    from this simplicial complex!"
                )
        return self._diagram

    ####################################################################################################
    #
    #   Member Functions
    #
    ####################################################################################################

    def calculate_homology(
        self,
        filter_fn=ollivier_ricci_curvature,
        use_min=True,
    ):
        """Compute Persistent Diagrams based on a curvature
        filtration of `self.graph`.

        Parameters
        -----------
        filter_fn: func
            The method for calculating discrete curvature of the graph.
            The default is set to Ollivier-Ricci.

        use_min: bool
            Sequence of edge values. Depending on the `use_min` parameter,
            either the minimum of all edge values or the maximum of all edge
            values is assigned to a vertex.


        Returns
        -----------
        persistence_diagram: src.jmapping.nammu.topology.PersistenceDiagram
            An array of tuples (b,d) that represent the birth and death of
            homological features in your graph according to the provided
            filtration function.


        """
        assert (
            len(self.graph.nodes()) > 0
        ), "First run `to_networkx` to generate a non-empty networkx graph."

        if len(self._curvature) > 0:
            self.curvature = filter_fn  # Set curvatures

        G = make_node_filtration(
            self.graph,
            self.curvature,
            attribute_name="curvature",
            use_min=use_min,
        )
        pd = calculate_persistence_diagrams(
            G,
            "curvature",
            "curvature",
        )
        self._diagram = pd
        return self.diagram
