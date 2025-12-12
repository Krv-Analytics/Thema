.. _advanced:

=============================
Customizing Thema
=============================

Extend Thema with custom filters, graph builders, and large-scale workflows.

Custom Filter Functions
-----------------------

Filters return 1 (keep) or 0 (discard) for each graph.

Filter Structure
^^^^^^^^^^^^^^^^

.. code-block:: python

   def custom_filter(star_object) -> int:
       if star_object.starGraph is None:
           return 0
       # Your logic here
       return 1 if condition else 0

Example: Component and Coverage Filter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import networkx as nx

   def multi_criteria_filter(star_object, min_components=3, min_coverage=50):
       """Keep graphs with at least min_components and min_coverage unique items."""
       if star_object.starGraph is None:
           return 0

       graph = star_object.starGraph.graph
       n_components = nx.number_connected_components(graph)

       unique_items = set()
       for node in graph.nodes():
           unique_items.update(graph.nodes[node]["membership"])

       return 1 if (n_components >= min_components and len(unique_items) >= min_coverage) else 0

Usage:

.. code-block:: python

   from functools import partial
   from thema.multiverse import Galaxy

   galaxy = Galaxy(YAML_PATH="params.yaml")
   galaxy.fit()

   my_filter = partial(multi_criteria_filter, min_components=4, min_coverage=100)
   selection = galaxy.collapse(filter_fn=my_filter, nReps=3)

Example: Density Filter
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def density_filter(star_object, min_density=0.1):
       """Keep graphs with edge density above threshold."""
       if star_object.starGraph is None:
           return 0

       graph = star_object.starGraph.graph
       n_nodes = graph.number_of_nodes()
       n_edges = graph.number_of_edges()

       if n_nodes < 2:
           return 0

       max_edges = n_nodes * (n_nodes - 1) / 2
       density = n_edges / max_edges if max_edges > 0 else 0

       return 1 if density >= min_density else 0

Custom Star Implementations
---------------------------

Create custom graph construction methods by subclassing ``Star``.

Minimal Star Template
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from thema.multiverse.universe.star import Star
   from thema.multiverse.universe.utils.starGraph import starGraph
   import networkx as nx

   class CustomStar(Star):
       def __init__(self, data_path, clean_path, projection_path, **kwargs):
           super().__init__(data_path, clean_path, projection_path)
           self.custom_param = kwargs.get("custom_param", default_value)

       def fit(self):
           # Access data
           projection = self.projection  # np.ndarray from Oort
           clean_data = self.clean       # pd.DataFrame from Planet

           # Build graph
           G = nx.Graph()
           # ... your graph construction logic ...

           # Wrap in starGraph
           self.starGraph = starGraph(G)

Example: k-NN Graph
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sklearn.neighbors import kneighbors_graph
   import numpy as np

   class kNNStar(Star):
       def __init__(self, data_path, clean_path, projection_path, k=5):
           super().__init__(data_path, clean_path, projection_path)
           self.k = k

       def fit(self):
           projection = self.projection

           # Build k-NN graph
           adjacency = kneighbors_graph(
               projection,
               n_neighbors=self.k,
               mode="connectivity"
           )

           G = nx.from_scipy_sparse_array(adjacency)

           # Add node attributes (membership = single item per node)
           for i, node in enumerate(G.nodes()):
               G.nodes[node]["membership"] = [i]

           self.starGraph = starGraph(G)

Registering Custom Stars
^^^^^^^^^^^^^^^^^^^^^^^^

To use custom stars in the pipeline:

1. Add your star class to ``thema/multiverse/universe/stars/``
2. Register it in ``thema/config.py`` via the ``star_name_to_config`` mapping
3. Reference the class name directly in YAML or parameter dictionaries

Scaling to Large Datasets
-------------------------

Performance Considerations
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Parameter Grid Size**
    Combinatorial explosion occurs quickly. A grid with 5 values per parameter across 4 parameters yields 625 combinations.

**Memory Management**
    - Filter aggressively before ``collapse()``
    - Process component counts separately to reduce memory footprint
    - Use ``distance_threshold`` instead of ``nReps`` for adaptive selection

**Computational Cost**
    - ``forman_curvature``: Fast (∼1s per graph pair)
    - ``balanced_forman_curvature``: Moderate (∼3s per graph pair)
    - ``ollivier_ricci_curvature``: Slow (∼10s+ per graph pair)

Optimization Strategies
^^^^^^^^^^^^^^^^^^^^^^^

1. **Start Small**: Validate pipeline with 10-20 graphs before expanding
2. **Progressive Refinement**: Coarse grid -> filter -> fine grid on promising regions
3. **Parallel Stages**: Planet, Oort, and Galaxy parallelize internally
4. **Disk I/O**: Use SSD for output directories; avoid network drives

Example: Large-Scale Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from thema.multiverse import Planet, Oort, Galaxy
   from thema.multiverse.universe.utils.starFilters import (
       minimum_unique_items_filter,
       component_count_range_filter
   )
   from pathlib import Path

   # Stage 1: Preprocessing (few seeds, multiple samples for robustness)
   planet = Planet(
       data="/data/large_dataset.pkl",
       outDir="./outputs/large_run/clean",
       seeds=[42, 13, 99],
       numSamples=3,
       scaler="standard",
       encoding="one_hot",
       imputeColumns="auto",
       imputeMethods="auto"
   )
   planet.fit()

   # Stage 2: Embeddings (sparse grid initially)
   oort = Oort(
       data="/data/large_dataset.pkl",
       cleanDir="./outputs/large_run/clean",
       outDir="./outputs/large_run/projections",
       params={
           "tsne": {
               "perplexity": [30, 66],
               "dimensions": [2],
               "seed": [42]
           },
           "pca": {
               "dimensions": [2],
               "seed": [42]
           }
       }
   )
   oort.fit()

   # Stage 3: Graphs (moderate grid)
   galaxy = Galaxy(
       data="/data/large_dataset.pkl",
       cleanDir="./outputs/large_run/clean",
       projDir="./outputs/large_run/projections",
       outDir="./outputs/large_run/models",
      params={
          "jmapStar": {
               "nCubes": [5, 10, 20],
               "percOverlap": [0.5, 0.7],
               "minIntersection": [-1],
               "clusterer": [
                   ["HDBSCAN", {"min_cluster_size": 5}],
                   ["HDBSCAN", {"min_cluster_size": 10}]
               ]
           }
       }
   )
   galaxy.fit()

   # Stage 4: Filter and select (use fast curvature metric)
   total_items = len(planet.data)
   coverage_filter = minimum_unique_items_filter(int(0.9 * total_items))
   comp_filter = component_count_range_filter(3, 8)

   def combined(star):
       return coverage_filter(star) and comp_filter(star)

   selection = galaxy.collapse(
       filter_fn=combined,
       curvature="forman_curvature",  # Fast option
       distance_threshold=200,
       selector="max_nodes"
   )

   print(f"Selected {len(selection)} representatives from {len(list(Path('./outputs/large_run/models').glob('*.pkl')))} graphs")

Debugging Tips
--------------

**Check intermediate outputs**
    Verify Moon, Comet, and Star files exist and load correctly

**Inspect graph properties**
    Load Star objects and check nodes, edges, components before filtering

**Validate filters**
    Test custom filters on a single graph before applying to all

**Monitor memory**
    Use ``top`` or ``htop`` during ``collapse()`` to catch memory issues

**Log parameter combinations**
    Track which parameter sets produce usable graphs
