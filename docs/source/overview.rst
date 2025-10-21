========
Overview
========

What is Thema?
--------------

Thema systematically explores hyperparameter spaces for unsupervised learning through three stages:

1. **Planet** (Preprocessing): Generates multiple clean data versions with different imputation, scaling, and encoding strategies
2. **Oort** (Embeddings): Creates low-dimensional projections across parameter grids (t-SNE, PCA)
3. **Galaxy** (Graph Construction & Selection): Builds Mapper graphs, computes topological distances, and selects representatives

Instead of manually tuning preprocessing and embedding parameters, Thema generates candidate models systematically and uses curvature-based graph distances to identify diverse, high-quality representatives.

Pipeline Flow
-------------

.. mermaid::

   flowchart LR
     A[Raw Data] --> B[Planet]
     B --> C[Moon Files]
     C --> D[Oort]
     D --> E[Comet Files]
     E --> F[Galaxy]
     F --> G[Star Graphs]
     G --> H[Curvature Distance]
     H --> I[Representatives]
     
     style A fill:#e1f5ff
     style C fill:#fff4e1
     style E fill:#ffe1f5
     style G fill:#e1ffe1
     style I fill:#ffe1e1

Core Concepts
-------------

**Moon**
    Preprocessed dataset (cleaned, encoded, scaled, imputed). Saved as ``.pkl`` files in ``clean/`` directory. Each Moon represents a specific combination of preprocessing choices.

**Comet**
    Low-dimensional embedding from a Moon. Contains projection array and metadata. Saved as ``.pkl`` files in ``projections/`` directory. Multiple Comets per Moon (one per projection method/parameter combo).

**Star**
    Mapper graph built from a Comet using Kepler Mapper algorithm. Contains nodes (clusters), edges (overlaps), and topology. Saved as ``.pkl`` files in ``models/`` directory.

**Galaxy**
    Orchestrator that generates Stars across parameter grids, computes pairwise graph distances using curvature filtrations, clusters similar graphs, and selects representatives.

Key Parameters
--------------

**Planet**
    - ``scaler``: ``standard``, ``minmax``, ``robust``
    - ``encoding``: ``one_hot``, ``label``, ``ordinal``
    - ``imputeMethods``: ``mean``, ``median``, ``mode``, ``sampleNormal``
    - ``seeds``: Random seeds for reproducible sampling

**Oort**
    - ``perplexity``: t-SNE neighborhood size (5-50)
    - ``dimensions``: Output dimensionality (typically 2)
    - ``projectiles``: List of methods to use (``tsne``, ``pca``)

**Galaxy**
    - ``nCubes``: Cover resolution (5-50)
    - ``percOverlap``: Cube overlap fraction (0-1)
    - ``clusterer``: Algorithm for within-cube clustering (HDBSCAN, DBSCAN, KMeans)
    - ``metric``: Graph distance (``stellar_curvature_distance``)
    - ``selector``: Representative selection (``max_nodes``, ``max_edges``, ``random``)

Output Structure
----------------

Thema organizes outputs hierarchically:

.. code-block:: text

   {outDir}/{runName}/
   ├── clean/
   │   ├── moon_42_0.pkl
   │   ├── moon_42_1.pkl
   │   └── ...
   ├── projections/
   │   ├── tsne_perplexity30_dims2_seed42_moon_42_0.pkl
   │   ├── pca_dims2_seed42_moon_42_0.pkl
   │   └── ...
   └── models/
       ├── star_tsne_perplexity30_nCubes10_overlap0.6.pkl
       ├── star_pca_dims2_nCubes10_overlap0.6.pkl
       └── ...

When to Use Thema
-----------------

**Good Use Cases**

- Exploring preprocessing choices for unsupervised learning
- Comparing embedding methods (t-SNE vs PCA) systematically
- Finding robust data representations across hyperparameter grids
- Identifying diverse graph topologies in your data
- Validating clustering stability across multiple configurations

**Not Ideal For**

- Supervised learning (Thema focuses on unsupervised tasks)
- Single fixed preprocessing pipeline (use sklearn directly)
- Real-time inference (Thema generates models offline)
- Small datasets (<100 samples; topological methods need sufficient data)

Next Steps
----------

**New Users**
    Start with :ref:`Quickstart <quickstart>` for a 5-minute walkthrough.

**YAML Workflows**
    See :ref:`Getting Started <getting_started>` for complete tutorial.

**Programmatic Control**
    Read :ref:`Programmatic Pipeline <programmatic>` for Python-only workflows.

**Parameter Tuning**
    Check :ref:`Tuning and Selection <tuning>` for grid strategies and filtering.

**Advanced Customization**
    Explore :ref:`Customizing Thema <advanced>` to write custom filters and graph builders.
