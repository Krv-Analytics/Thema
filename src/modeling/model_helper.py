def label_item_by_node(self):
    # Initialize Labels as -1
    N = len(self.tupper.clean)
    place_holder = [list() for _ in range(N)]
    labels = dict()
    nodes = self.complex["nodes"]
    for idx in range(N):
        for node_label in nodes.keys():
            if idx in nodes[node_label]:
                place_holder[idx].append(node_label)

        labels[idx] = place_holder

    self._model._node_id = labels
    # TODO: Compute node density description


def label_item_by_component(self):
    """
    Execute a mapper-based clutering based on connected components.
    Append a column to `self.data` labeling each item.

    Returns
    -----------
    data: pd.Dataframe
        An updated dataframe with a column titled `cluster_labels`

    """
    assert (
        len(self.complex) > 0
    ), "You must first generate a Simplicial Complex with `fit()` before you perform clustering."

    # Initialize Labels as -1 (`unclustered`)
    labels = -np.ones(len(self.tupper.clean))
    count = 0
    for component in self.components.keys():
        cluster_label = self.components[component]
        nodes = component.nodes()

        elements = []
        for node in nodes:
            elements.append(self.complex["nodes"][node])

        indices = set(itertools.chain(*elements))
        count += len(indices)
        labels[list(indices)] = cluster_label

    self._model.cluster_ids = labels
    # TODO: Compute node density description
