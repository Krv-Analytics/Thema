"""Object file for the JMapper Class."""


import kmapper as km
import networkx as nx
import numpy as np
from hdbscan import HDBSCAN
from kmapper import KeplerMapper

# Local Imports 
from tupper import Tupper
from jgraph import JGraph
from data_utils import convert_keys_to_alphabet

class JMapper():
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

    fit(): 
        A wrapper function for facilitating kmapper's `map` function. 
    
    """

##################################################################################################
#  
#   Initalization 
#
##################################################################################################
    
    def __init__(
        self, 
        tupper: Tupper = None,  # Container for user Data

                                # OR 
                                
        raw: str = "",          # path to raw data 
        clean: str = "",        # path to clean data   
        projection: str = "",   # path to projected data
            
        clusterer= dict(),
        cover = None,       
        complex = dict(),         
        min_intersection=None,  
        mapper: KeplerMapper = KeplerMapper(verbose=0),
    ):
        """Constructor for JMapper class.
        
        Parameters
        -----------
        Tupper: <tupper.Tupper>
            A data container that holds raw, cleaned, and projected
            versions of user data.

        OR

        raw: str
            Path (relative to root) to the users raw data (only pkl supported at this time) 
        
        clean: str
            Path (relative to root) to the users clean data (only pkl supported at this time)
        
        projection: str         
            Path (relative to root) to the users clean data (only pkl supported at this time)
        
    Optional Parameters
    -------------------

        verbose: int, default is 0
            Logging level passed through to `kmapper`. Levels (0,1,2)
            are supported.

        clusterer: 
            Scikit-learn API compatible clustering algorithm. Must provide fit and predict. 
            Default is `HDBSCAN`. 
        
        cover (kmapper.Cover): 
            Cover scheme for lens. Instance of kmapper.cover providing methods fit and transform.
        
        complex (kmapper.simplicial_complex): 
            A dictionary with “nodes”, “links” and “meta” information. 
        
        min_intersection: 
            The minimum intersection required for an edge to arise. Set to `-1` for 
            weighted graph representation.

        mapper: <kmapper.KeplerMapper>
            An instance of kmapper's KeplerMapper  
        """

    # Intialize inherited Tupper 
        if tupper is None: 
            self._tupper = Tupper(tupper.get_raw_path(), 
                           tupper.get_clean_path(), 
                           tupper.get_projection_path())
        else:
            self._tupper = tupper

    
        self._mapper = mapper
        self._clusterer = clusterer   
        self._cover = cover
        self._complex = complex 
        self._min_intersection = min_intersection

        self._jgraph = None

    
 
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
    def mapper(self):
        """Returns the scikit-tda object generated when executing
        the Mapper algorithm."""
        return self._mapper

    @property
    def cover(self):
        """Return the cover used to fit JMapper."""
        return self._cover

    @property
    def clusterer(self):
        """Return the clusterer used to fit JMapper."""
        return self._clusterer

    @property
    def complex(self):
        """Return the clusterer used to fit JMapper."""
        if len(self._complex["nodes"]) == 0:
            try:
                self.fit(clusterer=self.clusterer)
            except self._complex == dict():
                print("Your simplicial complex is empty!")
                print(
                    "Note: some parameters may produce a trivial\
                    mapper representation. \n"
                )
        return self._complex
    
    @property
    def min_intersection(self):
        "Returns the min_intersection value used to initialize the JGraph object"
        if self._min_intersection is None:
            print(
                "Please choose a minimum intersection \
                to generate a networkX graph!"
            )
        return self._min_intersection
    
    
    @property
    def jgraph(self):
        "Return the JGraph Object"
        if self._jgraph is None: 
            try: 
                self._jgraph = JGraph(convert_keys_to_alphabet(self._complex),
                                        self._min_intersection)
            except: 
                print("There was an error creating your JGraph Object")
        else:
            return self._jgraph

    
##################################################################################################
#  
#  KeplerMapper Fitting Function
#
##################################################################################################

    
    def fit(
        self,
        n_cubes: int = 6,
        perc_overlap: float = 0.4,
        clusterer=HDBSCAN(min_cluster_size=6),
    ):
        """
        Apply scikit-tda's implementation of the Mapper algorithm.
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
        self.n_cubes = n_cubes
        self.perc_overlap = perc_overlap
        self._cover = km.Cover(n_cubes, perc_overlap)
        self._clusterer = clusterer

        projection = self.tupper.projection
        # Compute Simplicial Complex
        self._complex = self._mapper.map(
            lens=projection,
            X=projection,
            cover=self.cover,
            clusterer=self.clusterer,
        )

        return self._complex
    