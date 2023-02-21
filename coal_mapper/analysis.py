# TODO: Implement Persistence Homology based statistics here


from nammu.topology import PersistenceDiagram, calculate_persistence_diagrams
from nammu.curvature import ollivier_ricci_curvature, forman_curvature
from nammu.utils import make_node_filtration
from mapper import CoalMapper


class MapperAnalysis:
    """
    Analyzing different mapper graphs using discrete curvature and persistenet homology.

    """

    # TODO: Add full docstring and document functions
    def __init__(self, X):

        self.data = X
        self._curvature = None
        self._graph = None
        self._diagrams = None

    @property
    def graph(self):
        return self._graph

    @property
    def curvature(self):
        return self._curvature

    @property
    def diagrams(self):
        return self._diagrams

    # TODO: Test these methods and interface with CoalMapper
    @graph.setter
    def graph(self, cover, min_intersection: int = 1):

        # Check that a reasonable
        if (len(cover) == 2) and type(cover) is tuple:
            n_cubes, perc_overlap = cover
            if type(min_intersection) is not int or min_intersection < 1:
                print(
                    "Invalid Minimum Intersection Parameter for Mapper. \
                Defualt value (min_intersection=1) has been applied."
                )
                min_intersection = 1
            else:
                mapper = CoalMapper(X=self.data).compute_mapper(n_cubes, perc_overlap)
                self._graph = mapper.to_networkx(min_intersection)

                # Recompute Curvature and Persistence Diagrams
                ()

        else:
            print(
                "Please enter a valid Cover of the form \
                (n_cubes,perc_overlap)"
            )

    @curvature.setter
    def curvature(self, curvature_fn):

        assert curvature_fn in [
            ollivier_ricci_curvature,
            forman_curvature,
        ], "Invalid Curvature Function"

        if self._graph is None:
            print(
                "You must first define a graph representation for \
                `X` via using `kmapper` before you can compute edge curvatures"
            )
        else:
            self._curvature = curvature_fn(self._graph)

    @diagrams.setter
    def diagrams(self, filter_fn, use_min=True):
        if self._graph is None:
            print(
                "You must first define a graph representation for \
                `X` via using `kmapper` before you can compute persistent diagrams"
            )
        # Default to OR Curvature
        if filter_fn in [ollivier_ricci_curvature, forman_curvature]:
            if self._curvature is None:
                self._curvature = filter_fn(self._graph)  # Set curvatures

            G = make_node_filtration(
                self._graph, attribute="curvature", use_min=use_min
            )
            pd = calculate_persistence_diagrams(
                G,
                "curvature",
                "curvature",
            )
            self._diagrams = pd

        # User Defined Edge Filtration
        else:
            try:
                edge_values = filter_fn(self._graph)
                G = make_node_filtration(self._graph, edge_values, use_min=use_min)
                pd = calculate_persistence_diagrams(G)
                self._diagrams = pd
            except:
                print("Invalid filter function.")

    # TODO: Implement Visualization Methods. See starter code in ./nammu/
    def plot_curvature(self):
        # Look at filtration visualization script
        pass

    def plot_diagrams(self):
        pass


class CondensationAnalysis:
    """Handle Communications with PECAN"""

    # TODO:
    # check whether PECAN is installed
    # feature selection and convert to np.array, save as txt
    # Specify Callbacks
    # How to handle nested poetry??
