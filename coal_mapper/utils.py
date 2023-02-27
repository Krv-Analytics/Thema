import numpy as np
import pickle
from sklearn.cluster import KMeans


from coal_mapper.nammu.topology import calculate_persistence_diagrams
from coal_mapper.nammu.curvature import ollivier_ricci_curvature, forman_curvature
from coal_mapper.nammu.utils import make_node_filtration
from data_processing.accessMongo import mongo_pull

from coal_mapper.mapper import CoalMapper


class MapperTopology:
    """
    Analyzing different mapper graphs using discrete curvature and persistenet homology.

    """

    def __init__(self, X: np.ndarray):

        self.data = X
        self._curvature = None
        self._graph = None
        self._diagram = None

    @property
    def graph(self):
        return self._graph

    @property
    def curvature(self):
        if self._curvature is None:
            print(
                "Curvature has not been computed yet. \
                First generate a networkx Graph from the dataset via Mapper."
            )
        return self._curvature

    @property
    def diagram(self):
        if self._diagram is None:
            print(
                "Persistent Homology has not been computed yet. \
                First generate a networkx Graph from the dataset via Mapper."
            )
        return self._diagram

    @curvature.setter
    def curvature(self, curvature_fn):

        if self._graph is None:
            print(
                "You must first define a graph representation for \
                `X` via using `kmapper` before you can compute edge curvatures"
            )
        else:
            try:
                curvature = curvature_fn(self.graph)
                assert len(curvature) == len(self._graph.edges())
                self._curvature = curvature
            except:
                print("Invalid Curvature function")

    def set_graph(
        self,
        cover,
        clusterer=KMeans(5, n_init="auto", random_state=1618033),
        min_intersection: int = 1,
    ):
        """Generate a new networkX graph from Data via mapper. Recompute"""
        # Check that a reasonable Cover is provided
        if (len(cover) == 2) and type(cover) is tuple:
            n_cubes, perc_overlap = cover
            if type(min_intersection) is not int or min_intersection < 1:
                print(
                    "Invalid Minimum Intersection Parameter for Mapper. \
                Defualt value (min_intersection=1) has been applied."
                )
                min_intersection = 1
            else:
                print("Computing Mapper Algorithm...")
                mapper = CoalMapper(X=self.data)
                mapper.clusterer = clusterer
                mapper.compute_mapper(n_cubes, perc_overlap)
                print("Generating networkx Graph...")
                self._graph = mapper.to_networkx(min_intersection)

                # Automatically Compute OR Curvature and corresponding Diagrams when changing a graph
                print(
                    "Using Ollivier Ricci filtration to compute edge curvature values and persistence diagrams. "
                )
                self.curvature = ollivier_ricci_curvature
                self.calculate_homology(filter_fn=ollivier_ricci_curvature)

        else:
            print(
                "Please enter a valid Cover of the form \
                (n_cubes,perc_overlap)"
            )

    def calculate_homology(self, filter_fn, use_min=True):
        if self._graph is None:
            print(
                "You must first define a graph representation for \
                `X` via using `kmapper` before you can compute persistent diagrams"
            )
        # Default to OR Curvature
        if self._curvature is None:
            "Computing edge curvature values"
            self._curvature = filter_fn(self._graph)  # Set curvatures

        G = make_node_filtration(
            self._graph,
            self._curvature,
            attribute_name="curvature",
            use_min=use_min,
        )
        pd = calculate_persistence_diagrams(
            G,
            "curvature",
            "curvature",
        )
        self._diagram = pd

    # TODO: Implement Visualization Methods. See starter code in ./nammu/
    def plot_curvature(self):
        # Look at filtration visualization script
        pass

    def plot_diagrams(self):
        pass


def curvature_analysis(
    X,
    n_cubes,
    perc_overlap,
    K,
    min_intersection_vals,
):

    # Configure CoalMapper
    mapper = MapperTopology(X=X)
    clusterer = KMeans(n_clusters=K, n_init="auto")
    cover = (n_cubes, perc_overlap)
    # Generate Graphs
    results = {}

    for val in min_intersection_vals:
        mapper.set_graph(cover=cover, clusterer=clusterer, min_intersection=val)
        mapper.calculate_homology(filter_fn=ollivier_ricci_curvature, use_min=True)
        results[val] = (
            mapper.graph,
            mapper.curvature,
            mapper.diagram,
        )
    return results
