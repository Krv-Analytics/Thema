import numpy as np
import pandas as pd

class Model:
    def __init__(self,tupper):

        self._tupper = tupper

        #Mapper Node Attributes
        self._node_id = None
        self._node_labels = None

        # Mapper Cluster
        self._cluster_ids = None
        self._cluster_description = None
        

    @property
    def cluster_ids(self):
        return self._cluster_ids
    
    @property.setter
    def cluster_ids(self,ids):
        self._cluster_ids = ids
        return self.cluster_ids 
    
    @property 
    def cluster_labels(self):
        return self._cluster_labels
    
    @property.setter
    def cluster_labels(self,labels):
        self._cluster_labels = labels
        return self.cluster_lables



    def cluter_density_analysis(self,cluter_id):
    



    def density_analysis(self):

        return 
