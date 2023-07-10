"Object file for the JBottle class.  "

import networkx as nx
import pandas as pd

class JBottle():
    """
    A bottle containing all of your data analysis needs. 

    This class is designed to faciliate the interaction of a graph representation 
    arising from a JMapper fitting with the user's original data (whether it be raw, 
    clean, or projected). The hope is that this class will contain all necessary structures
    and functionality for any and all required data analysis, as well as provide utilities 
    to simplify the visualizations in Model.py. 

    """
    def __init__(self,
                #raw: pd.DataFrame,
                #clean: pd.DataFrame, 
                #projection: pd.DataFrame, 
                node_members: dict(), 
                connected_components: list,
                 ): 
        #self._raw = raw 
        #self._clean = clean 
        #self._projection = projection

        self._phonebook = {} 

        for i in connected_components: 
            component_members = {}
            for node in connected_components[0].nodes:
                component_members[node] = node_members[node] 
                self._phonebook[i] = self._component_members
    
    @property
    def phonebook(self):
        return self._phonebook