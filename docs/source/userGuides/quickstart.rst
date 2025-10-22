.. _quickstart:

==========
Quickstart
==========

Run the full pipeline from raw data to representative graphs.

.. mermaid::
   :align: center

   flowchart LR
     A[params.yaml] --> B[Planet: clean]
     B --> C[Oort: embed]
     C --> D[Galaxy: graphs]
     D --> E[Representatives]

One-Shot Pipeline
-----------------

.. code-block:: python

   from thema.thema import Thema

   T = Thema("params.yaml")
   T.genesis()
   print(T.selected_model_files)  # Paths to representative graphs

Minimal Configuration
---------------------

**params.yaml**

.. code-block:: yaml

   runName: my_run
   data: /path/to/data.pkl
   outDir: ./outputs

   Planet:
     scaler: standard           # Zero-mean, unit-variance scaling
     encoding: one_hot          # Categorical encoding
     imputeColumns: auto        # Auto-detect columns with missing values
     imputeMethods: auto        # Auto-select imputation method per column
     numSamples: 1              # Number of imputed datasets per seed
     seeds: auto                # Auto-generate random seeds

   Oort:
     tsne:
       perplexity: [30]         # t-SNE neighborhood size
       dimensions: [2]          # Output dimensions
       seed: [42]
     pca:
       dimensions: [2]
       seed: [42]
     projectiles: [tsne, pca]   # Methods to run

   Galaxy:
     metric: stellar_curvature_distance  # Graph distance metric
     selector: max_nodes                 # Selection strategy
     nReps: 2                            # Number of representatives
     stars: [jmap]                       # Graph construction method
     jmap:
       nCubes: [8]              # Cover resolution
       percOverlap: [0.3]       # Cube overlap fraction
       minIntersection: [-1]    # Edge formation threshold (-1 = weighted)
       clusterer:
         - [HDBSCAN, {min_cluster_size: 5}]

Key Parameters
--------------

**runName** : str
    Output subdirectory name

**data** : str
    Absolute path to input data (CSV, pickle, parquet)

**outDir** : str
    Base directory for all outputs. Creates: ``{outDir}/{runName}/{clean,projections,models}/``

**Planet.scaler** : str
    Options: ``standard`` (recommended), ``minmax``, ``robust``, ``None``

**Oort.projectiles** : list of str
    Projection methods to use: ``tsne``, ``pca``

**Galaxy.nReps** : int
    Number of representative graphs to select

**Galaxy.selector** : str
    Options: ``max_nodes`` (largest graph), ``max_edges``, ``random``

Step-by-Step Control
--------------------

Run stages independently:

.. code-block:: python

   from thema.multiverse import Planet, Oort, Galaxy

   planet = Planet(YAML_PATH="params.yaml")
   planet.fit()  # Outputs: {outDir}/{runName}/clean/*.pkl

   oort = Oort(YAML_PATH="params.yaml")
   oort.fit()    # Outputs: {outDir}/{runName}/projections/*.pkl

   galaxy = Galaxy(YAML_PATH="params.yaml")
   galaxy.fit()  # Outputs: {outDir}/{runName}/models/*.pkl

   reps = galaxy.collapse()  # Dict: {cluster_id: {"star": StarGraph, "file": Path}}

Cleaning Outputs
----------------

Remove previous run outputs:

.. code-block:: python

   T = Thema("params.yaml")
   T.spaghettify()  # Deletes {outDir}/{runName}/ directory tree

Logging
-------

Enable detailed logging:

.. code-block:: python

   import thema
   thema.enable_logging('DEBUG')  # Verbose output
   # or
   thema.enable_logging('INFO')   # Progress messages
