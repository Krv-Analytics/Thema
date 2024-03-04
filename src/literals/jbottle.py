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

from .data_utils import (
    get_minimal_std,
    std_zscore_threshold_filter,
    get_best_std_filter,
    get_best_zscore_filter,
    error,
)

from ..fitting.tupper import Tupper
from ..fitting.jmapper import JMapper


########################################################################################
#
#   JBottle class Implementation
#
########################################################################################


class JBottle:
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

        # self._global_means = {col_name: [{'raw':self._raw[0]}] }
        self._unclustered = jmapper.get_unclustered_items()

        # Dictionaries to aid in group decomposition and item lookup
        self._group_lookuptable = {key: [] for key in self._raw.index}
        self._node_lookuptable = {key: [] for key in self._raw.index}
        self._group_directory = {}
        node_members = jmapper.nodes

        for i in jmapper.jgraph.components:
            cluster_members = {}
            for node in jmapper.jgraph.components[i].nodes:
                cluster_members[node] = node_members[node]
                for item in node_members[node]:
                    self._node_lookuptable[item] = self._node_lookuptable[item] + [node]
                    self._group_lookuptable[item] = list(
                        set(self._group_lookuptable[item] + [i])
                    )
            self._group_directory[i] = cluster_members

    ########################################################################################
    #
    #   Properties
    #
    ########################################################################################

    @property
    def raw(self):
        """Returns raw data."""
        return self._raw

    @property
    def clean(self):
        """Returns clean data."""
        return self._clean

    @property
    def projection(self):
        """Returns projected data."""
        return self._projection

    ########################################################################################
    #
    #   Member Functions
    #
    ########################################################################################

    def get_items_groupID(self, item_id: int):
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

    def get_nodes_groupID(self, node_id: str):
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

    def get_global_stats(self):
        """
        Calculates global mean and standard deviation statistics.

        Returns
        -------
        A dictionary containing statistics on both raw and clean df subsets for each group.
        """
        group_stats = {}
        raw_stats = pd.DataFrame()
        clean_stats = pd.DataFrame()
        dropped_columns = self.jmapper.tupper.get_dropped_columns()
        for id in self._group_directory.keys():

            numeric_columns = (
                self.get_groups_raw_df(id)
                .drop(columns=dropped_columns)
                .select_dtypes(include=np.number)
                .columns
            )
            raw_sub_df = self.get_groups_raw_df(id).select_dtypes(include=np.number)
            raw_stats["std"] = raw_sub_df.std()
            raw_stats["mean"] = raw_sub_df.mean()

            clean_sub_df = self.get_groups_clean_df(id)[numeric_columns]
            clean_stats["std"] = clean_sub_df.std()
            clean_stats["mean"] = clean_sub_df.mean()
            group_stats[id] = {"raw": raw_stats, "clean": clean_stats}

        return group_stats

    def get_nodes_raw_df(self, node_id: str):
        """
        Returns a subset of the raw dataframe only containing members of the specified node.

        Parameters
        ----------
        node_id : str
            A node's string identifier

        Returns
        --------
        A pandas data frame.
        """
        member_items = self.get_nodes_members(node_id)
        return self._raw.iloc[member_items]

    def get_nodes_clean_df(self, node_id: str):
        """
        Returns a subset of the clean dataframe only containing members of the specified node.

        Parameters
        ----------
        node_id : str
            A node's string identifier

        Returns
        --------
        A pandas data frame.
        """

        member_items = self.get_nodes_members(node_id)
        return self._clean.iloc[member_items]

    def get_nodes_projections(self, node_id: str):
        """
        Returns a subset of the projectinos array only containing members of the specified node.

        Parameters
        ----------
        node_id : str
            A node's string identifier

        Returns
        --------
        An np.array of projections.
        """
        member_items = self.get_nodes_members(node_id)
        projections = {}
        for item in member_items:
            projections[item] = self._projection[item]
        return projections

    def get_groups_raw_df(self, group_id: int):
        """
        Returns a subset of the raw dataframe only containing members of the specified group.

        Parameters
        ----------
        group_id : int
            A group's identifier

        Returns
        --------
        A pandas data frame.
        """
        member_items = self.get_groups_members(group_id)
        return self._raw.iloc[member_items]

    def get_groups_clean_df(self, group_id: int):
        """
        Returns a subset of the clean dataframe only containing members of the specified group.

        Parameters
        ----------
        group_id : int
            A group's identifier

        Returns
        --------
        A pandas data frame.
        """
        member_items = self.get_groups_members(group_id)
        return self._clean.iloc[member_items]

    def get_groups_projections(self, group_id: int):
        """
        Returns a subset of the projectinos array only containing members of the specified group.

        Parameters
        ----------
        node_id : str
            A groups's identifier

        Returns
        --------
        An np.array of projections.
        """
        member_items = self.get_groups_members(group_id)
        projections = {}
        for item in member_items:
            projections[item] = self._projection[item]
        return projections

    #   NOTE: Implementations of description functions can be found in data_utils.py

    def compute_node_description(self, node_id: str, description_fn=get_minimal_std):
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
            mask = self.get_nodes_members(node_id)

        label = description_fn(
            df=self.clean,
            mask=mask,
            density_cols=cols,
        )
        size = len(mask)
        return {"label": label, "size": size}

    def compute_group_description(self, group_id: int, description_fn=get_minimal_std):
        """
        Compute a simple description of a policy group.

        This function creates a density description based on its member nodes description
        in compute_node_description().

        Parameters:
        -----------
        group_id: int
            A group's identifier (-1 to get unclustered group)

        description_fn: function
            A function to be passed to compute_node_description()

        Returns
        -------
        A density description of the group.

        """
        tmp = {}
        group_size = 0
        if group_id == -1:
            unclustered_density = self.compute_node_description(
                -1, description_fn=description_fn
            )
            return {unclustered_density["label"]: 1}
        else:
            member_nodes = self.get_groups_member_nodes(group_id)
            for node in member_nodes:
                node_density = self.compute_node_description(
                    node, description_fn=description_fn
                )
                label = node_density["label"]
                size = node_density["size"]
                group_size += size

                # If multiple nodes have same identifying column
                if label in tmp.keys():
                    size += tmp[label]
                tmp[label] = size

            return {
                label: np.round(size / group_size, 2) for label, size in tmp.items()
            }

    def compute_group_identity(
        self, group_id: int, eval_fn=std_zscore_threshold_filter, *args, **kwargs
    ):
        """
        Computes the most important identifiers of a group as specified by the evalulation function.

        Parameters
        ----------

        group_id:
            A group's identifier.

        eval_fn:
            The function used score each column in the dataframe. The minimum scoring columns are chosen to represent the
            group's identity.

        kwargs:
            Any key word arguments that need to passed to the aliased evaluation functinon. If, for example,
            you wanted to pass a parameter `std_threshold` to your eval function, `std_zscore_threshold_filter`
            you could do as

            `compute_group_identity(id, eval_fn=std_zscore_threshold_filter, std_threshold=0.8)`
        """
        dropped_columns = self.jmapper.tupper.get_dropped_columns()
        global_stats = self.get_global_stats()[group_id]
        numeric_columns = (
            self.get_groups_raw_df(group_id)
            .drop(columns=dropped_columns)
            .select_dtypes(include=np.number)
            .columns
        )
        sub_df = self.get_groups_clean_df(group_id)[numeric_columns]
        id_table = sub_df.aggregate(eval_fn, global_stats=global_stats, *args, **kwargs)

        min_val = id_table.min()
        return id_table[id_table == min_val].index.tolist()

    def get_group_descriptions(self, description_fn=get_minimal_std):
        """
        Returns a dictionary of group descriptions for each group as specified by the passed
        description function.

        Parameter
        ---------
        description_fn: function
            A function that determines a representative column for each node in a group.

        Returns
        --------
        A density representing the composition of a group by its nodes' descriptions.
        """
        descriptions = {}
        for group_id in self._group_directory.keys():
            descriptions[group_id] = self.compute_group_description(
                group_id=group_id, description_fn=description_fn
            )

        return descriptions

    def get_group_identities(
        self,
        eval_fn=std_zscore_threshold_filter,
        *args,
        **kwargs,
    ):
        """
        Returns a dictionary of group identies as specified by compute_group_identity.

        Paramters
        ---------

        eval_fn:
            The function used score each column in the dataframe. The minimum scoring columns are chosen to represent the
            group's identity.

        kwargs:
            Any key word arguments that need to passed to the aliased evaluation functinon. If, for example,
            you wanted to pass a parameter `std_threshold` to your eval function, `std_zscore_threshold_filter`
            you could do so with

            `get_group_identities(eval_fn=std_zscore_threshold_filter, std_threshold=0.8)`
        """
        identities = {}
        for group_id in self._group_directory.keys():
            identities[group_id] = self.compute_group_identity(
                group_id=group_id, eval_fn=eval_fn, *args, **kwargs
            )

        return identities

    def target_matching(
        self,
        target: pd.DataFrame,
        col_filter: list = None,
    ):
        """
        Matches a target item into a generated group by calculating the minimum deviation
        from a groups mean over available numeric columns.

        Parameters
        ----------
        target: pd.DataFrame
            A data frame containing one row.

        col_filter:
            A list of columns to perform the mathcing on.
        """

        target_cols = target.select_dtypes(include=np.number).dropna(axis=1).columns
        if col_filter:
            raw_cols = col_filter
        else:
            raw_cols = self.raw.select_dtypes(include=np.number).columns

        scores = {}
        for group_id in self._group_directory.keys():
            group_data = self.get_groups_raw_df(group_id)
            score = 0
            for col in target_cols:
                if col in raw_cols:
                    x = target[col][0]
                    mu = group_data[col].mean()
                    score += error(x, mu)
            scores[group_id] = score

        min_index = min(scores, key=scores.get)
        return scores, min_index
