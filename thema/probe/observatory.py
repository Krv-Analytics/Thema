# File: probe/observatory.py
# Last Update: 05/16/24
# Updated by: JW

import pickle
from abc import abstractmethod

from ..core import Core


class Observatory(Core):
    """
    A bottle for all of your data analysis needs.

    This class is designed to facilitate the interaction of a graph
    representation arising fitted Stars on the user's original data
    (whether it be raw,
    clean, or projected).

    The hope is that this class will contain all necessary structures
    and functionality for any and all required data analysis, as well as
    provide utilities to simplify the visualizations.

    Parameters
    ----------
    Core : class
        The base class for the Observatory.

    Attributes
    ----------
    raw : pd.DataFrame
        A data frame of the raw data used in jmapping and graph generation.

    clean : pd.DataFrame
        A data frame of the clean data used in jmapping and graph generation.

    projection : np.array
        An array of the projected data used in jmapping and graph generation.

    Methods
    -------
    get_items_groupID(item)
        Returns the group of the selected item (-1 if unclustered).

    get_items_nodeID(item)
        Returns the list of node_ids an item is a member of (-1 if unclustered).

    get_nodes_members(node_id)
        Returns the list of member items in a selected node.

    get_groups_members(group_id)
        Returns the list of member items in a selected group.

    get_groups_member_nodes(group_id)
        Returns the list of member nodes in a selected group.

    get_nodes_groupID(node_id)
        Returns the group of the selected node.

    __init__(self, star_file)
        Initialization of an Observatory object.

    Parameters
    ----------
    star_file : str
        The path to the star file.

    Raises
    ------
    AssertionError
        If the starGraph attribute of the star object is None.

    """

    def __init__(self, star_file):
        """
        Initialization of an Observatory object.

        Parameters
        ----------
        star_file : str
            The path to the star file.

        Raises
        ------
        AssertionError
            If the starGraph attribute of the star object is None.

        """
        with open(star_file, "rb") as f:
            self.star = pickle.load(f)

        assert self.star.starGraph is not None

        super().__init__(
            data_path=self.star.get_data_path(),
            clean_path=self.star.get_clean_path(),
            projection_path=self.star.get_projection_path(),
        )

    @abstractmethod
    def get_items_groupID(self, item):
        """
        Returns the group of the selected item (-1 if unclustered).

        Parameters
        ----------
        item : str
            The selected item.

        Returns
        -------
        int
            The group ID of the selected item (-1 if unclustered).

        """
        raise NotImplementedError

    @abstractmethod
    def get_items_nodeID(self, item):
        """
        Returns the list of node_ids an item is a member of (-1 if unclustered).

        Parameters
        ----------
        item : str
            The selected item.

        Returns
        -------
        list
            The list of node IDs the item is a member of (-1 if unclustered).

        """
        raise NotImplementedError

    @abstractmethod
    def get_nodes_members(self, node_id):
        """
        Returns the list of member items in a selected node.

        Parameters
        ----------
        node_id : int
            The ID of the selected node.

        Returns
        -------
        list
            The list of member items in the selected node.

        """
        raise NotImplementedError

    @abstractmethod
    def get_groups_members(self, group_id):
        """
        Returns the list of member items in a selected group.

        Parameters
        ----------
        group_id : int
            The ID of the selected group.

        Returns
        -------
        list
            The list of member items in the selected group.

        """
        raise NotImplementedError

    @abstractmethod
    def get_groups_member_nodes(self, group_id):
        """
        Returns the list of member nodes in a selected group.

        Parameters
        ----------
        group_id : int
            The ID of the selected group.

        Returns
        -------
        list
            The list of member nodes in the selected group.

        """
        raise NotImplementedError

    @abstractmethod
    def get_nodes_groupID(self, node_id):
        """
        Returns the group of the selected node.

        Parameters
        ----------
        node_id : int
            The ID of the selected node.

        Returns
        -------
        int
            The group ID of the selected node.

        """
        raise NotImplementedError
