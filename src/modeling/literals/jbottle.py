"Object file for the JBottle class.  "

import networkx as nx
import pandas as pd
from tupper import Tupper

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


    Member Functions 
    ---------

    """
    def __init__(self,
                 tupper: Tupper(), 
                node_members: dict(), 
                connected_components: list
                 ): 
        """
        Initialization of a JBottle object 
        """
        self._raw = tupper.raw
        self._clean = tupper.clean
        self._projection = tupper.projection

    # Dictionaries to aid in group decomposition and item lookup 
        self._group_lookuptable = {key:[] for key in self._raw.index}
        self._node_lookuptable = {key:[] for key in self._raw.index}        
        self._group_directory = {}

        for i in connected_components: 
            cluster_members = {}
            for node in connected_components[i].nodes:
                cluster_members[node] = node_members[node] 
                for item in node_members[node]:
                    self._node_lookuptable[item] = self._node_lookuptable[item] + [node]
                    self._group_lookuptable[item] = list(set(self._group_lookuptable[item] + [i]))
            self._group_directory[i] = cluster_members
        
    
    @property 
    def raw(self):
        return self._raw 
    
    @property
    def clean(self):
        return self._clean 
    
    @property 
    def projection(self):
        return self._projection

    
    def get_items_groupID(self, item_id):
        """
        Look up function for finding an item's connected component (ie group)

        Parameters:
        -----------
        item_id: int 
            Index of desired look up item from user's raw data frame 
        
        Returns: 
        --------
        A list of group ids that the item is a member of 

        """
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
        return self._node_lookuptable[item_id] 
    
    def get_nodes_members(self, node_id: str):
        """
        Look up function to get members of a node 

        Parameters:
        -----------
        node_id: str 
            String identifier of node 
        
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
        Returns the group that a node is a member of 
    
        """
        for group in self._group_directory.keys():
            for node in self._group_directory[group].keys():
                if node == node_id:
                    return group
        return None
    

    def get_items_in_multiple_groups(self):
        # STUB! 
        return -1 
    
    