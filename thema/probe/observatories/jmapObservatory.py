# File: probe/visual_utils.py
# Last Update: 05/15/24
# Updated by: JW

import warnings

import numpy as np
import pandas as pd

from ..data_utils import (
    custom_Zscore,
    error,
    get_minimal_std,
    select_highestZscoreCols,
    std_zscore_threshold_filter,
)
from ..observatory import Observatory


def initialize():
    return jmapObservatory


class jmapObservatory(Observatory):
    """
    Custom observatory for viewing JMAP Stars.

    This class extends the `Observatory` class and provides additional
    functionality specific to the graph models outputted by JMAP Star.

    Parameters:
    -----------
    star_file : str
        The file path to the star file.

    Attributes:
    -----------
    _unclustered : list
        A list of unclustered items.
    _group_lookuptable : dict
        A dictionary to aid in group decomposition and item lookup.
    _node_lookuptable : dict
        A dictionary to aid in group decomposition and item lookup.
    _group_directory : dict
        A dictionary containing cluster members for each group.

    Methods:
    --------
    get_items_groupID(item_id)
        Look up function for finding an item's connected component (i.e., group).
    get_items_nodeID(item_id)
        Look up function for finding an item's member node.
    get_nodes_members(node_id)
        Look up function to get members of a node.
    get_groups_members(group_id)
        Look up function to get items within a group.
    get_groups_member_nodes(group_id)
        Look up function to get nodes within a connected component.
    get_nodes_groupID(node_id)
        Returns the node's group id.
    get_global_stats()
        Calculates global mean and standard deviation statistics.
    get_nodes_raw_df(node_id)
        Returns a subset of the raw dataframe only containing members of the
        specified node.
    get_nodes_clean_df(node_id)
        Returns a subset of the clean dataframe only containing members of the
        specified node.
    get_nodes_projections(node_id)
        Returns a subset of the projections array only containing members of
        the specified node.
    get_groups_raw_df(group_id)
        Returns a subset of the raw dataframe only containing members of the
        specified group.
    get_groups_clean_df(group_id)
        Returns a subset of the clean dataframe only containing members of the
        specified group.
    get_groups_projections(group_id)
        Returns a subset of the projections array only containing members of
        the specified group.
    compute_node_description(node_id, description_fn=get_minimal_std)
        Compute a simple description of each node in the graph.
    compute_group_description(group_id, description_fn=get_minimal_std)
        Compute a simple description of a policy group.

    Example
    ---------
    >>> from thema.probe.observatories import jmapObservatory
    >>> star_file = "path/to/star_file"
    >>> obs = jmapObservatory(star_file)
    >>> obs.get_items_groupID(1)
    """

    def __init__(self, star_file):
        """
        Initialize a JmapObservatory object.

        Parameters
        ----------
        star_file : str
            The path to the star file.

        Returns
        -------
        None

        Notes
        -----
        This constructor initializes a JmapObservatory object by calling the superclass's
        constructor and setting up various data structures for group decomposition and item lookup.

        The following instance variables are initialized:
        - self._unclustered : list
            A list of unclustered items obtained from the star file.
        - self._group_lookuptable : dict
            A dictionary that maps each item to the list of groups it belongs to.
        - self._node_lookuptable : dict
            A dictionary that maps each item to the list of nodes it belongs to.
        - self._group_directory : dict
            A dictionary that maps each group to its cluster members.

        """
        super().__init__(star_file=star_file)

        self._unclustered = self.star.get_unclustered_items()

        self._group_lookuptable = {key: [] for key in self.data.index}
        self._node_lookuptable = {key: [] for key in self.data.index}
        self._group_directory = {}
        node_members = self.star.nodes

        for i in self.star.starGraph.components:
            cluster_members = {}
            for node in self.star.starGraph.components[i].nodes:
                cluster_members[node] = node_members[node]
                for item in node_members[node]:
                    self._node_lookuptable[item] = self._node_lookuptable[item] + [node]
                    self._group_lookuptable[item] = list(
                        set(self._group_lookuptable[item] + [i])
                    )
            self._group_directory[i] = cluster_members

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
            return self._group_lookuptable[item_id][0]

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
        return self.data.iloc[member_items]

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
        return self.clean.iloc[member_items]

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
        return self.data.iloc[member_items]

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
        return self.clean.iloc[member_items]

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
            self.data.select_dtypes(include=["number"]).columns,
            self.clean.columns,
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

    def get_group_numbers(self) -> list:
        """
        Return a list of all group #s in a jmapStar graph
        """
        return list(self.star.starGraph.components.keys())

    def get_aggregatedGroupDf(
        self, aggregation_func=None, clean: bool = True
    ) -> pd.DataFrame:
        """
        Aggregate each group of the DataFrame using a custom aggregation function.

        Parameters:
        - aggregation_func: function, the aggregation function to apply

        Returns:
        - DataFrame with the aggregation function applied to each group
        """
        if aggregation_func is None:
            aggregation_func = np.mean

        if clean:
            df_func = self.get_groups_clean_df
        else:
            df_func = self.get_groups_raw_df

        combined_df = pd.DataFrame()
        for num in self.get_group_numbers():
            temp = df_func(num)
            temp["Group"] = num
            combined_df = pd.concat([combined_df, temp], ignore_index=True)

        grouped_df = combined_df.groupby("Group")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aggregated_df = grouped_df.agg(aggregation_func).reset_index()

        if not clean:
            for col in aggregated_df.select_dtypes(include=["object"]):
                if col in aggregated_df.columns:
                    most_common = grouped_df[col].agg(
                        lambda x: x.value_counts().idxmax()
                    )
                    aggregated_df[col] = aggregated_df.apply(
                        lambda row: most_common[row["Group"]], axis=1
                    )

        return aggregated_df

    def define_nodeValueDict(
        self, group_number: int, col: str, aggregation_func=None
    ) -> dict:
        """
        Creates a dict where each node is assiged a value based on an
        aggregation of items in that node,used to create a path graph
        target/sink nodes.

        Parameters
        ---------
        group_number : int
            Group/Connected component number

        col : str
            Column from the clean dataframe

        aggregation_func : np.<function>, defaults to calculating mean
            method by which to aggregate values within a node for coloring
                - color node by the median value or by the sum of values for
                example supports all numpy aggregation functions such as
                np.mean, np.median, np.sum, etc

        Returns
        -------
        results_dict : dict
            A dictionarty with node IDs as keys and their corresponding
            numeric value as values
        """
        if aggregation_func is None:
            aggregation_func = np.mean

        results = []
        for node in self.get_groups_member_nodes(group_number):
            df = self.get_nodes_clean_df(node)
            results.append((node, df[col].agg(aggregation_func)))

        results.sort(key=lambda x: x[1], reverse=True)
        results_dict = {}
        for _, (node, ret) in enumerate(results, start=1):
            results_dict[node] = ret

        return results_dict

    def dataset_zscores_df(self, n_cols=10):
        """
        STUB
        """
        zscore_df = pd.DataFrame()

        for group in self.get_group_numbers():
            subset_df = self.get_groups_clean_df(group)
            group_dict = {"Group": group}

            for col in self.clean:
                t = custom_Zscore(self.clean, subset_df=subset_df, column_name=col)
                group_dict[col] = t

            zscore_df = pd.concat(
                [zscore_df, pd.DataFrame([group_dict])], ignore_index=True
            )
        zscores = zscore_df.set_index("Group")
        return select_highestZscoreCols(zscores, n_cols=n_cols)
