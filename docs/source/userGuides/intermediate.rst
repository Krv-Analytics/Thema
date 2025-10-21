.. _tuning:

====================
Tuning and Selection
====================

Fine-tune parameter grids, apply filters, and select representatives.

Parameter Grid Strategy
-----------------------

Start small, expand based on results:

1. **Planet**: 2-3 seeds, 1-2 samples each
2. **Oort**: 3 values per parameter
3. **Galaxy**: 2-3 values for nCubes and percOverlap

Oort: Embedding Parameters
--------------------------

t-SNE Grid
^^^^^^^^^^

.. code-block:: yaml

   Oort:
     tsne:
       perplexity: [15, 30, 50]
       dimensions: [2]
       seed: [42]
     projectiles: [tsne]

**perplexity**
    - 5-15: Local structure (small clusters)
    - 30-50: Global structure (large-scale patterns)
    - Rule: perplexity ≈ sqrt(n_samples)

PCA Grid
^^^^^^^^

.. code-block:: yaml

   Oort:
     pca:
       dimensions: [2, 3, 5]
       seed: [42]
     projectiles: [pca]

**dimensions**
    - 2: Fast, good for visualization
    - 3-5: Captures more variance, slower graph construction

Galaxy: Mapper Parameters
-------------------------

Mapper Configuration
^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   Galaxy:
     metric: stellar_curvature_distance
     selector: max_nodes
     nReps: 3
     stars: [jmap]
     jmap:
       nCubes: [5, 10, 20]
       percOverlap: [0.5, 0.6, 0.7]
       minIntersection: [-1]
       clusterer:
         - [HDBSCAN, {min_cluster_size: 3}]
         - [HDBSCAN, {min_cluster_size: 5}]

**nCubes**
    Number of intervals covering the projection space.
    
    - 5: Coarse resolution, few large clusters
    - 10-20: Moderate (recommended starting point)
    - 50+: Fine-grained, many small clusters

**percOverlap**
    Fraction of overlap between adjacent cubes (0-1).
    
    - 0.3-0.5: Less connectivity, more components
    - 0.6-0.7: Moderate connectivity (recommended)
    - 0.8+: High connectivity, fewer components

**minIntersection**
    Minimum shared items to form an edge.
    
    - -1: Weighted edges (recommended)
    - Positive: Stricter edge requirements

**clusterer**
    Algorithm and parameters for clustering within cubes.

Clustering Algorithms
^^^^^^^^^^^^^^^^^^^^^

**HDBSCAN** (recommended)

.. code-block:: yaml

   clusterer:
     - [HDBSCAN, {min_cluster_size: 3}]
     - [HDBSCAN, {min_cluster_size: 5, min_samples: 3}]

- ``min_cluster_size``: Minimum items per cluster (2-10 typical)
- ``min_samples``: Core point requirement (optional)

**DBSCAN**

.. code-block:: yaml

   clusterer:
     - [DBSCAN, {eps: 0.5, min_samples: 5}]

**KMeans**

.. code-block:: yaml

   clusterer:
     - [KMeans, {n_clusters: 8}]

Filtering Graphs
----------------

Apply filters to remove unwanted graphs before distance computation.

Built-in Filters
^^^^^^^^^^^^^^^^

**component_count(k)**
    Keep graphs with exactly k components

**component_count_range(min_k, max_k)**
    Keep graphs with component count in [min_k, max_k]

**minimum_nodes_filter(n)**
    Keep graphs with at least n nodes

**minimum_edges_filter(n)**
    Keep graphs with at least n edges

**minimum_unique_items_filter(n)**
    Keep graphs covering at least n unique data points

Programmatic Filtering
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from thema.multiverse import Galaxy
   from thema.multiverse.universe.utils.starFilters import (
       component_count_filter,
       minimum_unique_items_filter
   )

   galaxy = Galaxy(YAML_PATH="params.yaml")
   galaxy.fit()

   # Filter for 3-component graphs with 80%+ coverage
   filter_3comp = component_count_filter(3)
   selection = galaxy.collapse(
       filter_fn=filter_3comp,
       selector="max_nodes",
       nReps=2
   )

Selection Strategies
--------------------

**selector** options in ``collapse()``:

**max_nodes** (recommended)
    Largest graph per cluster. Good for interpretability.

**max_edges**
    Most connected graph per cluster.

**min_nodes**
    Smallest graph per cluster. Minimal representatives.

**random**
    Random selection per cluster.

Collapse Methods
----------------

Two ways to control representative count:

**By Count**

.. code-block:: python

   selection = galaxy.collapse(
       nReps=5,
       selector="max_nodes"
   )

**By Distance Threshold**

.. code-block:: python

   selection = galaxy.collapse(
       distance_threshold=250,
       selector="max_nodes"
   )

Curvature Metrics
-----------------

Choose curvature metric in ``collapse()``:

.. code-block:: python

   selection = galaxy.collapse(
       curvature="balanced_forman_curvature",
       nReps=3
   )

**forman_curvature**
    Fast, default choice

**balanced_forman_curvature**
    More sensitive to structural differences

**resistance_curvature**
    Emphasizes connectivity patterns

**ollivier_ricci_curvature**
    Most detailed, slowest computation

Complete Example
----------------

.. code-block:: python

   from thema.multiverse import Galaxy
   from thema.multiverse.universe.utils.starFilters import (
       component_count_range_filter,
       minimum_unique_items_filter
   )

   galaxy = Galaxy(YAML_PATH="params.yaml")
   galaxy.fit()

   # Filter for 3-5 components with 85%+ coverage
   comp_filter = component_count_range_filter(3, 5)
   cov_filter = minimum_unique_items_filter(int(0.85 * total_items))

   def combined_filter(star):
       return comp_filter(star) and cov_filter(star)

   selection = galaxy.collapse(
       filter_fn=combined_filter,
       curvature="balanced_forman_curvature",
       nReps=4,
       selector="max_nodes"
   )

   print(f"Selected {len(selection)} representatives")
