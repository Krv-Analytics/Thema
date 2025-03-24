import numpy as np
from scipy.spatial.distance import cdist

class Realtor:
    """
    Find a the best location for an incoming target node in the cosmic neighborhood.
    """
    def __init__(self,target_vector,graph,node_features):
        self.target = target_vector
        self.node_features = node_features
        self.graph = graph
        
    
    def glauber_dynamics(self):
        # Placeholder for Glauber dynamics implementation @stuartwayland
        pass
    
    def node_docking(self,metric="euclidean"):
        distances = cdist([self.target], self.node_features, metric=metric)
        best_node_index = np.argmin(distances)
        return best_node_index
    
    
    def _star_link(self):
        # Connect into the Multiverse system. Take in a star and unpack appropriately
        pass