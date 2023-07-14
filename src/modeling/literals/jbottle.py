"Object file for the JBottle class.  "



import os 
import sys 
import networkx as nx
import pandas as pd
import numpy as np

########################################################################################
# 
#   Handling Local Imports
# 
########################################################################################

from data_utils import (
    get_minimal_std,
    _define_zscore_df,
    std_zscore_threshold_filter
)

from dotenv import load_dotenv
load_dotenv()
root = os.getenv("root")
sys.path.append(root + "jmapping/fitting")

from tupper import Tupper
from jmapper import JMapper


########################################################################################
# 
#   JBottle class Implementation
# 
########################################################################################



class JBottle():
    """
    A bottle for all of your data analysis needs. 

    This class is designed to faciliate the interaction of a graph representation 
    arising from a JMapper fitting with the user's original data (whether it be raw, 
    clean, or projected). The hope is that this class will contain all necessary structures
    and functionality for any and all required data analysis, as well as provide utilities 
    to simplify the visualizations in Model.py. 

    Members
    ------- 
    
    raw: pd.DataFrame
        A data frame of the raw data used in jmapping and graph generation 

    clean: pd.DataFrame
        A data frame of the clean data used in jmapping and graph generation

    projection: 
        An np.array of the projected data used in jmapping and graph generation


    Member Functions 
    ----------------

    get_items_groupID(item):
        Returns the group of the selected item (-1 if unclustered)
    
    get_items_nodeID(item):
        Returns the list of of node_ids an item is a member of (-1 if unclustered)

        
    get_nodes_members(node_id):
        Returns the list of member items in a selected node 
    
    get_groups_members(group_id):
        Returns the list of member items in a selected group 

    get_groups_member_nodes(group_id):
        Returns the list of member nodes in a selected group 
    
    get_nodes_groupID(node_id)
        Returns the group of the selected node. 

    
    TODO: Complete remaining documentation  

    """
    def __init__(self, jmapper): 
        """
        Initialization of a JBottle object. 

        Parameters: 
        ----------

        jmapper: <jmapper.JMapper> 
            A jmapper object
        
        """
        self._raw = jmapper.tupper.raw
        self._clean = jmapper.tupper.clean
        self._projection = jmapper.tupper.projection
        self._unclustered = jmapper.get_unclustered_items()

    # Dictionaries to aid in group decomposition and item lookup 
        self._group_lookuptable = {key:[] for key in self._raw.index}
        self._node_lookuptable = {key:[] for key in self._raw.index}        
        self._group_directory = {}
        node_members = jmapper.nodes


        for i in jmapper.components: 
            cluster_members = {}
            for node in jmapper.components[i].nodes:
                cluster_members[node] = node_members[node] 
                for item in node_members[node]:
                    self._node_lookuptable[item] = self._node_lookuptable[item] + [node]
                    self._group_lookuptable[item] = list(set(self._group_lookuptable[item] + [i]))
            self._group_directory[i] = cluster_members
        
########################################################################################
# 
#   Properties
# 
########################################################################################


    @property 
    def raw(self):
        return self._raw 
    
    @property
    def clean(self):
        return self._clean 
    
    @property 
    def projection(self):
        return self._projection
    

########################################################################################
# 
#   Member Functions
# 
########################################################################################


    
    def get_items_groupID(self, item_id):
        """
        Look up function for finding an item's connected component (ie group)

        Parameters:
        -----------
        item_id: int 
            Index of desired look up item from user's raw data frame 
        
        Returns: 
        --------
        A list of group ids that the item is a member of (-1 if unclustered)

        """
        if item_id in self._unclustered:
            return -1
        else:
            return self._group_lookuptable[item_id]
    
    def get_items_nodeID(self, item_id: int): 
        """
        Look up function for finding item's member node 

        Parameters:
        -----------
        item_id: int 
            Index of desired look up item from user's raw data frame 
        
        Returns: 
        --------
        A list of node ids that the item is a member of 

        """
        if item_id in self._unclustered:
            return -1
        else:
            return self._node_lookuptable[item_id] 
    
    def get_nodes_members(self, node_id: str):
        """
        Look up function to get members of a node 

        Parameters:
        -----------
        node_id: str 
            String identifier of a node 
        
        Returns: 
        --------
        A list of member items 

        """
        return self._group_directory[self.get_nodes_groupID(node_id)][node_id]
    
    def get_groups_members(self, group_id: int):
        """
        Look up function to get items within a group 

        Parameters: 
        -----------
        group_id: int
            Group number of desired connected component 
        
        Returns:
        --------
        A list of the item members for the specified group

        """
        member_list = []
        for node in self._group_directory[group_id].keys():
            member_list = member_list + self._group_directory[group_id][node]   
        return list(set(member_list))
    
    def get_groups_member_nodes(self, group_id: int):
        """
        Look up Function to get nodes within a connected component

        Parameters: 
        -----------
        group_id: int
            Group number of desired connected component 
        
        Returns:
        --------
        A list of node members for the specified group

        """
        return [node for node in self._group_directory[group_id].keys()]

    def get_nodes_groupID(self, node_id):
        """
        Returns the node's group id. 

        Parameters:
        -----------
        node_id : str 
            A character ID specifying the node

        Returns:
        --------
        A group ID number.
        
        """
        for group in self._group_directory.keys():
            for node in self._group_directory[group].keys():
                if node == node_id:
                    return group
        return None
    

    def get_nodes_raw_df(self, node_id): 
        """
        TODO: Fill out Doc String
        """
        member_items = self.get_nodes_members(node_id)
        return self._raw[self._raw['index'].isin(member_items)]
    
    def get_nodes_clean_df(self, node_id):
        """
        TODO: Fill out Doc String
        """
        member_items = self.get_nodes_members(node_id)
        return self._clean[self._clean['index'].isin(member_items)]
    
    def get_nodes_projections(self, node_id): 
        """
        TODO: Fill out Doc String
        """
        member_items = self.get_nodes_members(node_id)
        projections = []
        for item in member_items:
            projections[item] = self._projection[item]
        return projections
    
    def get_groups_raw_df(self, group_id):
        """
        TODO: Fill out Doc String
        """
        member_items = self.get_groups_members(group_id)
        return self._raw[self._raw['index'].isin(member_items)]
    

    def get_groups_clean_df(self, group_id):
        """
        TODO: Fill out Doc String
        """
        member_items = self.get_groups_members(group_id)
        return self._clean[self._clean['index'].isin(member_items)]

    def get_groups_projections(self, group_id):
        """
        TODO: Fill out Doc String
        """
        member_items = self.get_groups_members(group_id)
        projections = []
        for item in member_items:
            projections[item] = self._projection[item]
        return projections

   
#   NOTE: Implementations of description functions can be found in data_utils.py 

    def compute_node_description(self, node_id, description_fn=get_minimal_std):
        """
        Compute a simple description of each node in the graph.

        This function labels each node based on a description function. The description
        function is used to select a defining column from the original dataset, which will 
        serve as a representative of the noes identity. Obviously there is a number of ways 
        to do this, but as a default this computes the most homogenous data column for a each 
        node. 

        Parameters:
        ----------
        node_id: 
            A node identifier (-1 for unclustered items)

        description_fn: function 
            A function that takes a data frame, mask, and density columns and returns 
            a column.

        Returns:
        --------
        A dictionary containing the representing column label and the number of items in 
        the node. 

        """

        cols = np.intersect1d(
            self._raw.select_dtypes(include=["number"]).columns,
            self._clean.columns,
        )

        if node_id == -1:
            mask = self._unclustered
        else:
            mask = self._get_nodes_items(node_id)
        
        label = description_fn(
            df=self.tupper.clean,
            mask=mask,
            density_cols=cols,
        )
        size = len(mask)
        return {"label": label, "size": size}
    


    def compute_group_description(self, group_id, description_fn=get_minimal_std):
        """
        Compute a simple description of a policy group.

        This function creates a density description based on its member nodes description 
        in get_node_description(). 

        Parameters:
        -----------
        group_id: int
            A group's identifier (-1 to get unclustered group)
        
        description_fn: function 
            A function to be passed to get_node_description()

        Returns
        -------
        A density description of the group.
        
        """
        tmp = {}
        group_size = 0
        if group_id == -1:
            unclustered_density = self.get_node_description(-1, description_fn=description_fn)
            return {unclustered_density['label']:1}    
        else: 
            member_nodes = self.get_groups_member_nodes(group_id)
            for node in member_nodes:
                node_density = self.get_node_description(node, description_fn=description_fn)
                label = node_density["label"]
                size = node_density["size"]
                group_size += size

                # If multiple nodes have same identifying column
                if label in tmp.keys():
                    size += tmp[label]
                tmp[label] = size
        
            return {label: np.round(size / group_size, 2) for label, size in tmp.items()}
        


    def compute_group_identity(self, group_id, eval_fn=std_zscore_threshold_filter):
        """
        TODO: Fill out Doc String
        """
        # STUB! 
        return self._group_identifiers