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
        # Placeholder for Glauber dynamics implementation
        pass
    
    def node_docking(self,metric="euclidean"):
        #calculate pairwise distance between target and all nodes in the graph
        distances = cdist([self.target], self.node_features, metric=metric)
        #find the index of the node with the minimum distance to the target
        best_node_index = np.argmin(distances)
        #return the index of the best node
        return best_node_index