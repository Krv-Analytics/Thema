.. _graphing:

==================
Graphs & Selection
==================

Galaxy builds Mapper graphs (Stars) from embeddings, computes graph distances, clusters, and selects representatives.

Configuration
-------------

In ``params.yaml``:

.. code-block:: yaml

   Galaxy:
     metric: stellar_curvature_distance   # Graph-to-graph distance
     selector: max_nodes                  # Representative selection strategy
     nReps: 3                             # Or use distance_threshold
     stars: [jmap]
     jmap:
       nCubes: [8, 12]
       percOverlap: [0.2, 0.4]
       minIntersection: [-1]              # -1 => weighted edges
       clusterer:
         - [HDBSCAN, {min_cluster_size: 5}]

Parameters
----------

Galaxy
^^^^^^

**metric** : str
    Distance metric between graphs. Options:
    
    - ``"stellar_curvature_distance"`` (recommended)

**selector** : str
    Representative selection rule within each cluster. Options:
    
    - ``"max_nodes"`` (default)
    - ``"max_edges"``
    - ``"min_nodes"``
    - ``"random"``

**nReps** : int or null
    Number of representatives to select. If null, use ``distance_threshold`` in ``collapse()``.

JMAP (Mapper)
^^^^^^^^^^^^^

**nCubes** : list of int
    Number of intervals per dimension in the cover. Higher -> finer resolution.

**percOverlap** : list of float
    Overlap fraction between adjacent intervals (0-1).

**minIntersection** : list of int
    Minimum overlap size to form an edge. ``-1`` uses weighted edges without a minimum.

**clusterer** : list of [str, dict]
    Clustering algorithm and parameters used within each cube.

Build and Compare
-----------------

.. code-block:: python

   from thema.multiverse import Galaxy

   galaxy = Galaxy(YAML_PATH=yaml)
   galaxy.fit()  # Writes Star objects to models/

   selection = galaxy.collapse(
       metric="stellar_curvature_distance",
       curvature="balanced_forman_curvature",
       selector="max_nodes"
   )

   # selection: {cluster_id: {"star": StarGraph, "file": Path, ...}}

Visualization (Optional)
------------------------

.. code-block:: python

   import matplotlib.pyplot as plt

   coords = galaxy.get_galaxy_coordinates()  # MDS on the distance matrix
   plt.scatter(coords[:, 0], coords[:, 1], s=12, alpha=0.6)
   plt.title("Galaxy of graph models")
   plt.show()

Curvature Choices
-----------------

- ``forman_curvature``: Fast default
- ``balanced_forman_curvature``: More sensitive structure
- ``resistance_curvature``: Emphasizes connectivity
- ``ollivier_ricci_curvature``: Most detailed, slowest
