# File: multiverse/universe/star.py
# Last Update: 05/15/24
# Updated by: JW

import os
import pickle
from abc import abstractmethod

from ...core import Core


class Star(Core):
    """
    Simple Targeted Atlas Representation
    ----
    A STAR is a base class template for atlas (graph) construction algorithms.
    As a parent class, Star enforces structure on data management and graph
    generation, enabling a 'universal' procedure for generating these objects.

    For more information on implementing a realization of Star, please
    see docs/development/star.md.
    """

    def __init__(self, data_path, clean_path, projection_path):
        """
        Initalizes a Core managing raw, clean, and projected data.

        Parameters
        ----------
        data_path : str
            Path to a raw dataFrame saved as a pickle, csv, or xlsx file.
        clean_path : str
            Path to a Moon object File.
        projection_path : str
            Path to a projectile (child of a Comet) object file.
        """
        super().__init__(
            data_path=data_path,
            clean_path=clean_path,
            projection_path=projection_path,
        )
        self.starGraph = None

    @abstractmethod
    def fit(self):
        """
        An abstract method to be implemented by children of Star. This function
        must be realized by a graph construction algorithm able to
        initialize self.starGraph as a starGraph class.

        Note: All parameters necessary for the graph construction algorithm must
        be passed as arguments to the star child's constructor.
        """
        raise NotImplementedError

    def save(self, file_path, force=False):
        """
        Save the current object instance to a file using pickle serialization.

        Parameters
        ----------
        file_path : str
            The path to the file where the object will be saved.

        force : bool, default=False
            If True, saves object even if the starGraph is uninitialized or empty.

        Returns
        -------
        bool
            True if saved successfully, False otherwise.
        """
        try:
            save_ok = (
                force
                or hasattr(self, "starGraph")
                and hasattr(self.starGraph, "graph")
                and hasattr(self.starGraph.graph, "nodes")
                and len(self.starGraph.graph.nodes()) > 0
            )

            if save_ok:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "wb") as f:
                    pickle.dump(self, f)
                return True
            else:
                return False
        except Exception:
            return False
