.. _programmatic:

Manual Configuration Guide
===========================

This guide explains how to configure and run the Thema pipeline programmatically without YAML files. It covers preprocessing, dimensionality reduction, graph construction, filtering, and model selection.

Overview
--------

The Thema pipeline consists of four main stages:

1. **Planet** - Data preprocessing and cleaning
2. **Oort** - Dimensionality reduction
3. **Galaxy** - TDA graph construction using Mapper
4. **Collapse** - Model selection and filtering

Each stage produces outputs consumed by the next, creating a structured workflow from raw data to representative graphs.

Directory Structure
-------------------

Thema organizes outputs into three directories:

- ``clean/`` - Preprocessed datasets
- ``projections/`` - Dimensionality-reduced data
- ``graphs/`` - TDA Mapper graphs

**Important:** Thema requires absolute paths for all directory arguments.

.. code-block:: python

    from pathlib import Path
    
    base_dir = Path("/absolute/path/to/thema_outputs")
    clean_dir = base_dir / "clean"
    projections_dir = base_dir / "projections"
    graphs_dir = base_dir / "graphs"
    
    for d in [clean_dir, projections_dir, graphs_dir]:
        d.mkdir(parents=True, exist_ok=True)

Stage 1: Preprocessing with Planet
-----------------------------------

``Planet`` handles data cleaning and generates multiple preprocessed versions for robustness analysis.

Class Initialization
--------------------

.. code-block:: python

    from thema.multiverse import Planet
    
    planet = Planet(
        data=input_path,
        dropColumns=columns_to_drop,
        imputeColumns=columns_to_impute,
        imputeMethods=imputation_methods,
        scaler=scaling_method,
        seeds=random_seeds,
        numSamples=samples_per_seed
    )
    planet.outDir = clean_dir
    planet.fit()

Parameters
----------

``data`` : str or Path
    Absolute path to input data file (CSV, pickle, or parquet)

``dropColumns`` : list of str
    Column names to remove before analysis. Typically includes identifiers, dates, or non-numeric features.

``imputeColumns`` : list of str
    Column names requiring imputation for missing values. Must align with ``imputeMethods``.

``imputeMethods`` : list of str
    Imputation strategy for each column in ``imputeColumns``. Options:
    
    - ``"mode"`` - Most frequent value
    - ``"mean"`` - Column mean
    - ``"median"`` - Column median
    - ``"sampleNormal"`` - Sample from normal distribution fitted to column
    - ``"zeros"`` - Fill with zeros

``scaler`` : str
    Feature scaling method. Options:
    
    - ``"standard"`` - Zero mean, unit variance (recommended)
    - ``"minmax"`` - Scale to [0, 1] range
    - ``"robust"`` - Robust to outliers using IQR
    - ``None`` - No scaling

``seeds`` : list of int
    Random seeds for reproducible sampling. Each seed generates ``numSamples`` datasets.

``numSamples`` : int
    Number of imputed datasets per seed. Creates multiple "universes" for robustness.

Output
------

Planet generates preprocessed pickle files in ``outDir``:

- ``moon_<seed>_<sample>.pkl`` - Each contains cleaned, imputed, and scaled data

These files are automatically discovered by Oort and Galaxy.

Example
-------

.. code-block:: python

    planet = Planet(
        data="/data/raw_dataset.pkl",
        dropColumns=["id", "name", "timestamp"],
        imputeColumns=["age", "category", "value"],
        imputeMethods=["sampleNormal", "mode", "median"],
        scaler="standard",
        seeds=[42, 13, 99],
        numSamples=2
    )
    planet.outDir = clean_dir
    planet.fit()
    # Produces: 6 files (3 seeds × 2 samples)

Stage 2: Dimensionality Reduction with Oort
--------------------------------------------

``Oort`` projects high-dimensional data to lower dimensions for graph construction.

Class Initialization
--------------------

.. code-block:: python

    from thema.multiverse import Oort
    
    oort = Oort(
        data=input_path,
        cleanDir=clean_dir,
        outDir=projections_dir,
        params=projection_config
    )
    oort.fit()

Parameters
----------

``data`` : str or Path
    Path to original raw data file (same as Planet input)

``cleanDir`` : str or Path
    Absolute path to Planet output directory (``clean/``)

``outDir`` : str or Path
    Absolute path for projection outputs

``params`` : dict
    Nested dictionary specifying projection methods and hyperparameters

Projection Configuration
-------------------------

The ``params`` dictionary structure:

.. code-block:: python

    params = {
        "method_name": {
            "param1": [value1, value2, ...],
            "param2": [value3, value4, ...],
            "dimensions": [2],  # Output dimensionality
            "seed": [42]        # Random seed
        }
    }

Supported Methods
~~~~~~~~~~~~~~~~~

**t-SNE** (``"tsne"``)

.. code-block:: python

    "tsne": {
        "perplexity": [15, 30, 50],  # Balance local vs global structure
        "dimensions": [2],            # Typically 2 for Mapper
        "seed": [42]
    }

- ``perplexity``: Lower values (5-15) emphasize local structure, higher values (30-50) preserve global patterns

**PCA** (``"pca"``)

.. code-block:: python

    "pca": {
        "dimensions": [2, 3, 5],
        "seed": [42]  # Not used but required
    }

**UMAP** (``"umap"``)

.. code-block:: python

    "umap": {
        "n_neighbors": [15, 30, 50],
        "min_dist": [0.1, 0.3, 0.5],
        "dimensions": [2],
        "seed": [42]
    }

Output
------

Oort generates projection files in ``outDir``:

- ``<method>_<params>_moon_<seed>_<sample>.pkl`` - Reduced data for each parameter combination and Moon

Example
-------

.. code-block:: python

    projection_config = {
        "tsne": {
            "perplexity": [15, 30, 66],
            "dimensions": [2],
            "seed": [42]
        },
        "pca": {
            "dimensions": [2, 5],
            "seed": [42]
        }
    }
    
    oort = Oort(
        data="/data/raw_dataset.pkl",
        cleanDir=clean_dir,
        outDir=projections_dir,
        params=projection_config
    )
    oort.fit()

Stage 3: Graph Construction with Galaxy
----------------------------------------

``Galaxy`` constructs TDA Mapper graphs from projections using clustering and cover schemes.

Class Initialization
--------------------

.. code-block:: python

    from thema.multiverse import Galaxy
    
    galaxy = Galaxy(
        data=input_path,
        cleanDir=clean_dir,
        projDir=projections_dir,
        outDir=graphs_dir,
        params=mapper_config
    )
    galaxy.fit()

Parameters
----------

``data`` : str or Path
    Path to original raw data file

``cleanDir`` : str or Path
    Absolute path to Planet outputs (``clean/``)

``projDir`` : str or Path
    Absolute path to Oort outputs (``projections/``)

``outDir`` : str or Path
    Absolute path for graph outputs

``params`` : dict
    Mapper algorithm configuration

Mapper Configuration
--------------------

The ``params`` dictionary uses the ``"jmap"`` key:

.. code-block:: python

    params = {
        "jmap": {
            "nCubes": [5, 10, 20],
            "percOverlap": [0.5, 0.6, 0.7],
            "minIntersection": [-1],
            "clusterer": [
                ["HDBSCAN", {"min_cluster_size": 3}],
                ["HDBSCAN", {"min_cluster_size": 10}]
            ]
        }
    }

Mapper Parameters
~~~~~~~~~~~~~~~~~

``nCubes`` : list of int
    Number of hypercubes (intervals) covering the projection space. More cubes = finer resolution.
    
    - **3-5**: Coarse, few large clusters
    - **10-20**: Moderate resolution (recommended starting point)
    - **50+**: Fine-grained, many small clusters

``percOverlap`` : list of float
    Percentage overlap between adjacent hypercubes (0-1 range).
    
    - **0.3-0.5**: Less overlap, more disconnected components
    - **0.6-0.7**: Moderate overlap (recommended)
    - **0.8+**: High overlap, highly connected graphs

``minIntersection`` : list of int
    Minimum items required in cube overlap to form an edge.
    
    - **-1**: No minimum (default, recommended)
    - **Positive values**: Stricter edge formation

``clusterer`` : list of [str, dict] pairs
    Clustering algorithms and their parameters. Each entry is ``[algorithm_name, param_dict]``.

Clustering Options
~~~~~~~~~~~~~~~~~~

**HDBSCAN** (recommended)

.. code-block:: python

    ["HDBSCAN", {"min_cluster_size": 5, "min_samples": 3}]

- ``min_cluster_size``: Minimum items to form a cluster (2-10 typical)
- ``min_samples``: Core point requirement (optional)

**DBSCAN**

.. code-block:: python

    ["DBSCAN", {"eps": 0.5, "min_samples": 5}]

**KMeans**

.. code-block:: python

    ["KMeans", {"n_clusters": 8}]

Graph Interpretation
--------------------

Mapper graphs contain:

- **Nodes**: Clusters of data points
- **Edges**: Overlap between clusters (shared items)
- **Connected components**: Groups of connected nodes representing distinct patterns or "archetypes"

Output
------

Galaxy generates graph files in ``outDir``:

- ``star_<projection>_<mapper_params>.pkl`` - Each contains a Mapper graph model

Example
-------

.. code-block:: python

    mapper_config = {
        "jmap": {
            "nCubes": [5, 10, 20],
            "percOverlap": [0.55, 0.65, 0.75],
            "minIntersection": [-1],
            "clusterer": [
                ["HDBSCAN", {"min_cluster_size": 2}],
                ["HDBSCAN", {"min_cluster_size": 5}],
                ["HDBSCAN", {"min_cluster_size": 10}]
            ]
        }
    }
    
    galaxy = Galaxy(
        data="/data/raw_dataset.pkl",
        cleanDir=clean_dir,
        projDir=projections_dir,
        outDir=graphs_dir,
        params=mapper_config
    )
    galaxy.fit()

Stage 4: Filtering and Model Selection
---------------------------------------

After generating graphs, filter and select representative models using built-in or custom filters.

Graph Filtering
---------------

Built-in Filter Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from thema.multiverse.universe.utils.starFilters import (
        minimum_unique_items_filter,
        component_count_filter,
        component_count_range_filter,
        minimum_nodes_filter,
        minimum_edges_filter,
        nofilterfunction
    )

``minimum_unique_items_filter(n)``
    Keep graphs covering at least ``n`` unique data items
    
    .. code-block:: python
    
        coverage_filter = minimum_unique_items_filter(1000)

``component_count_filter(k)``
    Keep graphs with exactly ``k`` connected components
    
    .. code-block:: python
    
        three_component_filter = component_count_filter(3)

``component_count_range_filter(min_k, max_k)``
    Keep graphs with component count in range [min_k, max_k]
    
    .. code-block:: python
    
        mid_range_filter = component_count_range_filter(3, 8)

``minimum_nodes_filter(n)``
    Keep graphs with at least ``n`` nodes

``minimum_edges_filter(n)``
    Keep graphs with at least ``n`` edges

``nofilterfunction``
    No filtering, keep all graphs

Loading Filtered Graphs
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from thema.multiverse.universe.geodesics import _load_starGraphs
    
    filtered_graphs = _load_starGraphs(
        dir=graphs_dir,
        graph_filter=filter_function
    )

Example: Coverage-Based Filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pandas as pd
    from pathlib import Path
    
    # Get total item count from cleaned data
    sample_file = next(Path(clean_dir).glob("*.pkl"))
    total_items = len(pd.read_pickle(sample_file).imputeData)
    
    # Filter for 85% coverage
    coverage_filter = minimum_unique_items_filter(int(total_items * 0.85))
    high_coverage_graphs = _load_starGraphs(
        dir=graphs_dir,
        graph_filter=coverage_filter
    )

Model Collapse (Representative Selection)
------------------------------------------

The ``collapse()`` method clusters similar graphs and selects representatives.

Method Signature
~~~~~~~~~~~~~~~~

.. code-block:: python

    representatives = galaxy.collapse(
        metric="stellar_curvature_distance",
        curvature="forman_curvature",
        distance_threshold=250,
        nReps=None,
        selector="max_nodes",
        filter_fn=filter_function,
        files=list_of_graph_files
    )

Parameters
~~~~~~~~~~

``metric`` : str
    Distance metric for graph comparison
    
    - ``"stellar_curvature_distance"`` - Curvature-based (recommended)
    - Other metrics may be available depending on implementation

``curvature`` : str
    Curvature calculation method
    
    - ``"forman_curvature"`` - Forman-Ricci curvature (recommended)
    - ``"ollivier_curvature"`` - Ollivier-Ricci curvature (slower)

``distance_threshold`` : float
    Maximum distance for graphs to be considered similar. Lower = stricter clustering.

``nReps`` : int or None
    Number of representatives to select. If None, uses ``distance_threshold`` instead.

``selector`` : str
    How to choose representatives from each cluster
    
    - ``"max_nodes"`` - Graph with most nodes
    - ``"max_edges"`` - Graph with most edges
    - ``"min_nodes"`` - Graph with fewest nodes
    - ``"random"`` - Random selection

``filter_fn`` : callable or None
    Filter function to apply before clustering

``files`` : list of Path or None
    Specific graph files to consider. If None, uses all files in ``outDir``.

Return Value
~~~~~~~~~~~~

Dictionary mapping cluster IDs to representative graph information:

.. code-block:: python

    {
        0: {"star": StarGraph_object, "file": Path, ...},
        1: {"star": StarGraph_object, "file": Path, ...},
        ...
    }

Example: Component-Based Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from thema.multiverse.universe.utils.starFilters import component_count_filter
    
    # Select representatives for graphs with exactly 5 components
    filter_5_components = component_count_filter(5)
    
    representatives = galaxy.collapse(
        metric="stellar_curvature_distance",
        curvature="forman_curvature",
        distance_threshold=200,
        selector="max_nodes",
        filter_fn=filter_5_components,
        files=list(high_coverage_graphs)
    )
    
    # Extract StarGraph objects
    selected_graphs = [v["star"] for v in representatives.values()]

Example: Selecting Across Component Counts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from collections import defaultdict
    
    # Group by component count
    component_groups = defaultdict(list)
    for graph_file in high_coverage_graphs:
        star = pd.read_pickle(graph_file)
        n_components = star.starGraph.nComponents
        component_groups[n_components].append(graph_file)
    
    # Select representatives for each component count
    all_representatives = {}
    for n_components, files in component_groups.items():
        filter_fn = component_count_filter(n_components)
        reps = galaxy.collapse(
            metric="stellar_curvature_distance",
            curvature="forman_curvature",
            distance_threshold=250,
            selector="max_nodes",
            filter_fn=filter_fn,
            files=files
        )
        all_representatives[n_components] = [v["star"] for v in reps.values()]

Complete Workflow Example
--------------------------

.. code-block:: python

    from pathlib import Path
    from thema.multiverse import Planet, Oort, Galaxy
    from thema.multiverse.universe.geodesics import _load_starGraphs
    from thema.multiverse.universe.utils.starFilters import (
        minimum_unique_items_filter,
        component_count_filter
    )
    import pandas as pd
    
    # Setup
    base_dir = Path("/absolute/path/to/outputs")
    clean_dir = base_dir / "clean"
    projections_dir = base_dir / "projections"
    graphs_dir = base_dir / "graphs"
    
    for d in [clean_dir, projections_dir, graphs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 1. Preprocessing
    planet = Planet(
        data="/data/dataset.pkl",
        dropColumns=["id", "name"],
        imputeColumns=["age", "category"],
        imputeMethods=["sampleNormal", "mode"],
        scaler="standard",
        seeds=[42, 13],
        numSamples=2
    )
    planet.outDir = clean_dir
    planet.fit()
    
    # 2. Dimensionality Reduction
    oort = Oort(
        data="/data/dataset.pkl",
        cleanDir=clean_dir,
        outDir=projections_dir,
        params={
            "tsne": {
                "perplexity": [15, 30, 50],
                "dimensions": [2],
                "seed": [42]
            }
        }
    )
    oort.fit()
    
    # 3. Graph Construction
    galaxy = Galaxy(
        data="/data/dataset.pkl",
        cleanDir=clean_dir,
        projDir=projections_dir,
        outDir=graphs_dir,
        params={
            "jmap": {
                "nCubes": [5, 10, 20],
                "percOverlap": [0.6, 0.7],
                "minIntersection": [-1],
                "clusterer": [
                    ["HDBSCAN", {"min_cluster_size": 3}],
                    ["HDBSCAN", {"min_cluster_size": 8}]
                ]
            }
        }
    )
    galaxy.fit()
    
    # 4. Filter for High Coverage
    sample_file = next(Path(clean_dir).glob("*.pkl"))
    total_items = len(pd.read_pickle(sample_file).imputeData)
    coverage_filter = minimum_unique_items_filter(int(total_items * 0.85))
    
    high_coverage = _load_starGraphs(
        dir=graphs_dir,
        graph_filter=coverage_filter
    )
    
    # 5. Select Representatives for 3-Component Graphs
    filter_3_comp = component_count_filter(3)
    reps = galaxy.collapse(
        metric="stellar_curvature_distance",
        curvature="forman_curvature",
        distance_threshold=200,
        selector="max_nodes",
        filter_fn=filter_3_comp,
        files=list(high_coverage)
    )
    
    selected = [v["star"] for v in reps.values()]
    print(f"Selected {len(selected)} representative graphs")

Tips and Best Practices
-----------------------

Parameter Selection
-------------------

1. **Start Simple**: Begin with small parameter grids and expand based on results
2. **Preprocessing Seeds**: 2-3 seeds with 2-3 samples each provides good robustness
3. **Projection Methods**: t-SNE with perplexities [15, 30, 50] covers local to global structure
4. **Mapper Resolution**: Start with nCubes=[5, 10, 20] and percOverlap=[0.6, 0.7]
5. **Clustering**: HDBSCAN with min_cluster_size=[3, 5, 10] is robust

Performance Optimization
------------------------

- **Parallelization**: Planet, Oort, and Galaxy automatically parallelize across parameter combinations
- **Incremental Analysis**: Process subsets of parameters first to validate pipeline
- **File Management**: Large parameter grids generate many files; monitor disk usage
- **Memory**: Galaxy.collapse() loads graphs into memory; filter aggressively for large datasets

Common Pitfalls
---------------

1. **Relative Paths**: Always use absolute paths for directory arguments
2. **Mismatched Parameters**: Ensure ``imputeColumns`` and ``imputeMethods`` lists align
3. **Over-Parameterization**: Combinatorial explosion occurs quickly; be selective
4. **Coverage vs Resolution**: Balance coverage filtering with parameter exploration
5. **Component Count**: Some parameter combinations may produce zero components

Troubleshooting
---------------

**No graphs pass coverage filter**
    - Reduce coverage threshold
    - Increase percOverlap in Mapper config
    - Check data quality and imputation

**Too many similar graphs**
    - Decrease distance_threshold in collapse()
    - Use stricter filter_fn
    - Reduce parameter grid size

**Empty components**
    - Increase percOverlap
    - Decrease min_cluster_size
    - Use fewer nCubes

**Out of memory during collapse**
    - Filter more aggressively before collapse
    - Process component counts separately
    - Reduce number of graphs