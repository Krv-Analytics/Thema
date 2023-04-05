class Model:
    def __init__(self, tupper):

        self._tupper = tupper

        # Mapper Node Attributes
        self._node_id = None
        self._node_description = None

        # Mapper Cluster
        self._cluster_ids = None
        self._cluster_description = None

    @property
    def cluster_ids(self):
        return self._cluster_ids

    @cluster_ids.setter
    def cluster_ids(self, ids):
        self._cluster_ids = ids
        return self.cluster_ids

    @property
    def cluster_description(self):
        return self._cluster_description

    @cluster_description.setter
    def cluster_description(self, description):
        self._cluster_description = description

    @property
    def node_ids(self):
        return self._node_ids

    @node_ids.setter
    def node_ids(self, ids):
        self._node_ids = ids
        return self.node_ids

    @property
    def node_description(self):
        return self._node_description

    @node_description.setter
    def node_description(self, description):
        self._node_description = description
