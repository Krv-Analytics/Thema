"""Object file for the JMapper Class."""


import kmapper as km
import networkx as nx
import numpy as np
from hdbscan import HDBSCAN
from kmapper import KeplerMapper

# Local Imports
from tupper import Tupper
from jgraph import JGraph
from fitting_utils import convert_keys_to_alphabet


class JMapper:
    """A wrapper and expansion for scikit-tda's `KMapper`.

    This class allows you to generate graph models of high dimensional data
    based on Singh et al.'s Mapper algorithm. Moreover, this class holds
    a JGraph object, equiped with the ability to compute graph curvature and
    homology, and a JBottle object, created for easily facilitating conventional
    analysis of the mapped data.

    Members
    -------

    mapper: <km.KeplerMapper>
        A kepler mapper object

    cover:
        A kepler mapper Cover

    clusterer: <dict()>
        A kepler mapper supported clusterer

    complex:
        A kepler mapper simplicial complex

    min_intersection:
        The minimum intersection required for an edge to arise. Set to `-1` for
        weighted graph representation.

    jgraph:
        A graph class with the usual suite of graph algorithms as well as
         homological and curvature analysis.


    Member Functions
    ----------------

    re_fit():
        A function for refitting a kmapper's `map` function.

    """

    ##################################################################################################
    #
    #   Initalization
    #
    ##################################################################################################

    def __init__(
        self,
        tupper: Tupper,  # Container for user Data
        n_cubes: int,
        perc_overlap: float,
        clusterer,
        verbose: int = 0,
    ):
        """Constructor for JMapper class.

        Parameters
        -----------
        Tupper: <tupper.Tupper>
            A data container that holds raw, cleaned, and projected
            versions of user data.

        n_cubes: int
            The number of cubes used to fit member kepler mapper

        perc_overlap: float
            The percent overlap used to fit member kepler mapper

        clusterer:
            The clusterer used in to fit member kepler mapper

        verbose: int, default is 0
        """

        # Intialize inherited Tupper
        self._tupper = tupper
        self._n_cubes = n_cubes
        self._perc_overlap = perc_overlap
        self._clusterer = clusterer
        self._cover = km.Cover(n_cubes, perc_overlap)
        # self._unclustered_items = None
        self._mapper = KeplerMapper(verbose=verbose)

        self._unclustered_items = None

        # Compute Simplicial Complex
        try:
            self._complex = self._mapper.map(
                lens=self._tupper.projection,
                X=self._tupper.projection,
                cover=self._cover,
                clusterer=self._clusterer,
            )
            self._nodes = convert_keys_to_alphabet(self._complex["nodes"])
        except:
            print("You have produced an Empty Simplicial Complex")
            self._complex = -1
            self._nodes == -1

        # These members will be set in model_helper and model_selector_helper for computational
        # efficienty purposes

        self._min_intersection = None

        # Public Member
        self.jgraph = None

    ##################################################################################################
    #
    #     Properties
    #
    ##################################################################################################

    @property
    def tupper(self):
        """
        Returns the tupper class data container
        """
        return self._tupper

    @property
    def nodes(self):
        """
        Returns the nodes of a jmapper
        """
        return self._nodes

    @property
    def n_cubes(self):
        """
        Returns the n_cubes member
        """
        return self._n_cubes

    @property
    def perc_overlap(self):
        """
        Returns the perc_overlap member
        """
        return self._perc_overlap

    @property
    def clusterer(self):
        """Return the clusterer used to fit JMapper."""
        return self._clusterer

    @property
    def cover(self):
        """Return the cover used to fit JMapper."""
        return self._cover

    @property
    def mapper(self):
        """Returns the scikit-tda object generated when executing
        the Mapper algorithm."""
        return self._mapper

    @property
    def complex(self):
        """Return the clusterer used to fit JMapper."""
        return self._complex

    @property
    def nodes(self):
        """Returns a alphabet list of node_ids."""
        return self._nodes

    @property
    def min_intersection(self):
        "Returns the min_intersection value used to initialize the JGraph object"
        if self._min_intersection is None:
            return -1
        else:
            return self._min_intersection

    @min_intersection.setter
    def min_intersection(self, min_intersection: int = 1):
        self._min_intersection = min_intersection

    ##################################################################################################
    #
    #  Member Functions
    #
    ##################################################################################################

    def get_unclustered_items(self):
        """Returns the list of items that were not clustered in the mapper fitting.

        Returns
        -------
        self._unclustered_item : list
           A list of unclustered item ids
        """
        if self._unclustered_items is None:

            N = len(self.tupper.clean)
            labels = dict()
            unclustered_items = []
            for idx in range(N):
                place_holder = []
                for node_id in self._nodes.keys():
                    if idx in self._nodes[node_id]:
                        place_holder.append(node_id)

                if len(place_holder) == 0:
                    place_holder = -1
                    unclustered_items.append(idx)
                labels[idx] = place_holder

            self._unclustered_items = unclustered_items

        return self._unclustered_items

    def re_fit(
        self,
        n_cubes: int = 6,
        perc_overlap: float = 0.4,
        clusterer=HDBSCAN(min_cluster_size=6),
    ):
        """
        Used to re-apply scikit-tda's implementation of the Mapper algorithm.
        Returns a dictionary that summarizes the fitted simplicial complex.

        Parameters
        -----------
        n_cubes: int, defualt 6
            Number of cubes used to cover of the latent space.
            Used to construct a kmapper.Cover object.

        perc_overlap: float, default 0.4
            Percentage of intersection between the cubes covering
            the latent space. Used to construct a kmapper.Cover object.

        clusterer: default is HDBSCAN
            Scikit-learn API compatible clustering algorithm.
            Must provide `fit` and `predict`.

        Returns
        -----------
        complex : dict
            A dictionary with "nodes", "links" and "meta"
            information of a simplicial complex.

        """
        # Log cover and clusterer from most recent fit
        self._n_cubes = n_cubes
        self._perc_overlap = perc_overlap
        self._cover = km.Cover(n_cubes, perc_overlap)
        self._clusterer = clusterer

        # Compute Simplicial Complex
        self._complex = self._mapper.map(
            lens=self._tupper.projection,
            X=self._tupper.projection,
            cover=self._cover,
            clusterer=self._clusterer,
        )

        return self._complex
