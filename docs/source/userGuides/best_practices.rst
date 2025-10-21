.. _best_practices:

==============
Best Practices
==============

Workflow Strategy
-----------------

**Start Small, Scale Up**
    Begin with minimal parameter grids (2-3 values per parameter) to validate the pipeline. Expand grids based on initial results.

**Incremental Validation**
    Run Planet -> Oort -> Galaxy separately to inspect intermediate outputs before full automation.

**Parameter Exploration Order**
    1. Fix preprocessing (Planet)
    2. Explore embeddings (Oort)
    3. Tune graph construction (Galaxy)
    4. Apply filters and selection

Data Management
---------------

**File Formats**
    Save raw data as pickle (``.pkl``) to preserve dtypes and avoid parsing issues.

**Absolute Paths**
    Always use absolute paths for ``data``, ``cleanDir``, ``projDir``, ``outDir`` parameters.

**Output Organization**
    Use consistent naming: ``{outDir}/{runName}/{clean,projections,models}/``

**Clean Between Runs**
    Remove previous outputs to avoid confusion:

    .. code-block:: python

        T = Thema("params.yaml")
        T.spaghettify()  # Deletes entire {outDir}/{runName} tree

Preprocessing (Planet)
----------------------

**Auto-Detection**
    Use ``imputeColumns="auto"`` and ``imputeMethods="auto"`` for initial runs, then inspect with ``get_missingData_summary()``.

**Scaling**
    Use ``scaler="standard"`` (zero mean, unit variance) for most cases. Use ``"robust"`` only if outliers are problematic.

**Encoding**
    Use ``encoding="one_hot"`` for categorical variables. Avoid with high-cardinality features (>50 categories).

**Imputation Sampling**
    Set ``numSamples=1`` unless using randomized methods (``sampleNormal``, ``sampleCategorical``). Multiple samples only help capture imputation uncertainty.

**Seeds**
    Use 2-3 explicit seeds (e.g., ``[42, 13, 99]``) for reproducibility. Avoid ``"auto"`` in production runs.

Embeddings (Oort)
-----------------

**t-SNE Perplexities**
    Start with ``[15, 30, 50]`` to cover local and global structure. Adjust based on dataset size (perplexity ≈ sqrt(n_samples)).

**PCA Dimensions**
    Use 2D for speed and visualization. Higher dimensions capture more variance but slow down graph construction.

**Method Combination**
    Run both t-SNE and PCA to compare linear vs nonlinear projections.

**Reproducibility**
    Fix seeds for t-SNE: ``seed: [42]``. PCA is deterministic.

Graph Construction (Galaxy)
---------------------------

**Mapper Parameters**
    Start with ``nCubes: [5, 10, 20]`` and ``percOverlap: [0.5, 0.7]``. Adjust based on graph connectivity:
    
    - Too many disconnected components? Increase ``percOverlap`` or decrease ``nCubes``
    - Graphs too dense? Decrease ``percOverlap`` or increase ``nCubes``

**Clustering**
    Use HDBSCAN with ``min_cluster_size: [3, 5, 10]``. Start with smaller values for finer clusters.

**Edge Formation**
    Use ``minIntersection: [-1]`` for weighted edges (recommended). Positive values enforce stricter connectivity.

Filtering and Selection
-----------------------

**Filter Before Distance**
    Apply filters before ``collapse()`` to reduce computational cost. Use ``minimum_unique_items_filter`` to ensure coverage.

**Representative Selection**
    - ``selector="max_nodes"``: Most interpretable (default)
    - ``selector="max_edges"``: Most connected
    - ``selector="min_nodes"``: Minimal examples

**Component Count Strategy**
    Filter by component count to focus on specific graph topologies. Process different component counts separately for targeted selection.

Performance
-----------

**Curvature Metrics**
    - ``forman_curvature``: Fast, good for large grids
    - ``balanced_forman_curvature``: Better sensitivity, moderate speed
    - ``ollivier_ricci_curvature``: Use only when geometry is critical (slow)

**Grid Size Management**
    Parameter grids grow combinatorially. A grid with 4 parameters × 3 values each = 81 combinations per Moon file. Monitor disk space.

**Memory Optimization**
    - Filter aggressively before ``collapse()``
    - Use ``distance_threshold`` instead of ``nReps`` for adaptive selection
    - Process large datasets in batches by component count

**Parallel Execution**
    Planet, Oort, and Galaxy parallelize automatically. No manual configuration needed.

Reproducibility
---------------

**Version Control**
    Track ``params.yaml`` in git. Include Thema version in commit messages.

**Seed Management**
    Use explicit seeds for Planet and Oort: ``seeds: [42, 13, 99]``

**Environment Management**
    Use ``uv`` for dependency management:

    .. code-block:: bash

        uv sync --extra dev
        uv run python script.py

**Documentation**
    Save parameter configurations and filter criteria alongside outputs.

Troubleshooting
---------------

**No Graphs Pass Filters**
    - Reduce coverage threshold in ``minimum_unique_items_filter``
    - Increase ``percOverlap`` in Mapper config
    - Check for imputation issues in Planet

**Too Many Similar Graphs**
    - Decrease ``distance_threshold`` in ``collapse()``
    - Use stricter filters
    - Reduce parameter grid size

**Disconnected Graphs**
    - Increase ``percOverlap`` (try 0.7-0.8)
    - Decrease ``min_cluster_size`` in HDBSCAN
    - Use fewer ``nCubes``

**Out of Memory**
    - Filter more aggressively before ``collapse()``
    - Process component counts separately
    - Reduce parameter grid size
    - Use ``forman_curvature`` instead of slower metrics

**Slow Collapse**
    - Switch to ``forman_curvature``
    - Filter to fewer graphs before distance computation
    - Use ``distance_threshold`` instead of ``nReps``

Common Pitfalls
---------------

**Relative Paths**
    Always use absolute paths. Relative paths may break depending on execution context.

**Mismatched Parameters**
    Ensure ``imputeColumns`` and ``imputeMethods`` lists have the same length.

**Over-Parameterization**
    Resist the urge to test every possible parameter value. Start small, expand strategically.

**Ignoring Coverage**
    Graphs with low coverage miss large portions of the dataset. Always filter by ``minimum_unique_items``.

**Component Count Blindness**
    Different component counts represent fundamentally different topologies. Process them separately for better selection.
