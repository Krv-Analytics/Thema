from os.path import isfile
import pickle
import numpy as np
import itertools
from coal_mapper import CoalMapper


class Model:
    def __init__(self, mapper:str, min_intersection=1):
        
        self._mapper = None
        self._tupper = None
        self._complex = None
        self._hyper_parameters = None

        self.min_intersection = min_intersection
        
        # Check is valid mapper path
        if isfile(mapper):
            self._mapper = mapper 
        
        # Mapper Node Attributes
        self._node_ids = None
        self._node_description = None

        # Mapper Cluster Attributes
        self._cluster_ids = None
        self._cluster_description = None
    
    @property
    def mapper(self):
        assert self._mapper, "Please Specify a valid path to a mapper object"
        with open(self._mapper, "rb") as mapper_file:
            reference = pickle.load(mapper_file)
            mapper = reference[self.min_intersection]
        return mapper

    @property
    def tupper(self):
        return self.mapper.tupper
    @property
    def hyper_parameters(self):
        if self._hyper_parameters:
            return self._hyper_parameters
        assert self._mapper, "Please Specify a valid path to a mapper object"
        with open(self._mapper, "rb") as mapper_file:
            reference = pickle.load(mapper_file)
            self._hyper_parameters = reference["hyperparameters"]
        return self._hyper_parameters
    @property
    def complex(self):
        return self.mapper.complex
    
    @property
    def node_ids(self):
        if self._node_ids is None:
            self.label_item_by_node()
        return self._node_ids
    @property
    def node_description(self):
        if self._node_description is None:
            self.compute_node_descriptions()
        return self._node_description
        
    @property
    def cluster_ids(self):
        if self._cluster_ids is None:
            self.label_item_by_cluster()
        return self._cluster_ids
    @property
    def cluster_description(self):
        if self._cluster_description is None:
            self.compute_cluster_descriptions()
        return self._cluster_description


    def label_item_by_node(self):
        """Function to set the node_id"""
        # Initialize Labels as -1
        print("Computing labels of items by node...")
        N = len(self.tupper.clean)
        labels = dict()
        nodes = self.complex["nodes"]
        for idx in range(N):
            place_holder = []
            for node_id in nodes.keys():
                if idx in nodes[node_id]:
                    place_holder.append(node_id)

            if len(place_holder) ==0:
                place_holder = -1
            labels[idx] = place_holder

        self._node_ids = labels
        return self._node_ids

    def label_item_by_cluster(self):
        assert (
            len(self.complex) > 0
        ), "You must first generate a Simplicial Complex with `fit()` before you perform clustering."

        # Initialize Labels as -1 (`unclustered`)
        labels = -np.ones(len(self.tupper.clean))
        count = 0
        components = self.mapper.components
        for component in components.keys():
            cluster_id = components[component]
            nodes = component.nodes()

            elements = []
            for node in nodes:
                elements.append(self.complex["nodes"][node])

            indices = set(itertools.chain(*elements))
            count += len(indices)
            labels[list(indices)] = cluster_id

        self._cluster_ids = labels
        return 
    
    def compute_node_descriptions():
        pass

    def compute_cluster_descriptions():
        pass
    
    
