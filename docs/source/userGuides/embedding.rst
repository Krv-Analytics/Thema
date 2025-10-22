.. _embeddings:

==========
Embeddings
==========

Oort projects high-dimensional data to lower dimensions for graph construction. Supports t-SNE and PCA.

.. note::
   Requires Moon files from Planet. See :ref:`preprocessing` first.

Basic Usage
-----------

.. code-block:: python

   from thema.multiverse import Oort

   oort = Oort(
       data="/path/to/data.pkl",
       cleanDir="./outputs/my_run/clean",
       outDir="./outputs/my_run/projections",
       params={
           "tsne": {
               "perplexity": [30],
               "dimensions": [2],
               "seed": [42]
           }
       }
   )
   oort.fit()

Parameters
----------

**data** : str or Path
    Absolute path to original raw data file (same as Planet input)

**cleanDir** : str or Path
    Absolute path to Planet output directory containing Moon files

**outDir** : str or Path
    Absolute path for projection outputs

**params** : dict
    Nested dictionary of projection methods and hyperparameters

Projection Methods
------------------

t-SNE
^^^^^

Nonlinear dimensionality reduction preserving local structure.

.. code-block:: python

   params = {
       "tsne": {
           "perplexity": [15, 30, 50],
           "dimensions": [2],
           "seed": [42]
       }
   }

**perplexity** : list of int or float
    Balances local vs global structure. Typical range: 5-50.

    - 5-15: Emphasizes local neighborhoods
    - 30-50: Preserves global patterns
    - Rule of thumb: perplexity ≈ sqrt(n_samples)

**dimensions** : list of int
    Output dimensionality. Typically 2 for Mapper graphs.

**seed** : list of int
    Random seed for reproducibility

PCA
^^^

Linear dimensionality reduction via principal components.

.. code-block:: python

   params = {
       "pca": {
           "dimensions": [2, 3, 5],
           "seed": [42]  # Not used, but required
       }
   }

**dimensions** : list of int
    Number of principal components to retain

**seed** : list of int
    Placeholder (PCA is deterministic)

Parameter Grids
---------------

Oort generates embeddings for all parameter combinations:

.. code-block:: python

   params = {
       "tsne": {
           "perplexity": [15, 30, 50],
           "dimensions": [2],
           "seed": [42, 13]
       }
   }
   # Produces: 3 perplexities × 1 dimension × 2 seeds = 6 embeddings per Moon file

Output
------

Comet files saved as ``<method>_<params>_moon_<seed>_<sample>.pkl`` in ``outDir``:

- ``tsne_perplexity30_dims2_seed42_moon_42_0.pkl``
- ``pca_dims2_seed42_moon_42_0.pkl``

Each contains the reduced-dimension array and metadata.

Examples
--------

**Single Method**

.. code-block:: python

   oort = Oort(
       data="/data/survey.pkl",
       cleanDir="./outputs/analysis/clean",
       outDir="./outputs/analysis/projections",
       params={
           "tsne": {
               "perplexity": [30],
               "dimensions": [2],
               "seed": [42]
           }
       }
   )
   oort.fit()

**Multiple Methods**

.. code-block:: python

   oort = Oort(
       data="/data/survey.pkl",
       cleanDir="./outputs/analysis/clean",
       outDir="./outputs/analysis/projections",
       params={
           "tsne": {
               "perplexity": [15, 30, 50],
               "dimensions": [2],
               "seed": [42]
           },
           "pca": {
               "dimensions": [2, 5],
               "seed": [42]
           }
       }
   )
   oort.fit()

**YAML Configuration**

In ``params.yaml``:

.. code-block:: yaml

   Oort:
     tsne:
       perplexity: [30, 50]
       dimensions: [2]
       seed: [42]
     pca:
       dimensions: [2]
       seed: [42]
     projectiles: [tsne, pca]

Then:

.. code-block:: python

   oort = Oort(YAML_PATH="params.yaml")
   oort.fit()

Best Practices
--------------

- Start with t-SNE perplexities [15, 30, 50] to capture local and global structure
- Use 2D embeddings for Mapper (higher dimensions increase computational cost)
- PCA is fast and deterministic; use for baseline comparisons
- t-SNE is stochastic; fix seeds for reproducibility
- Grid explosion: 5 parameters × 3 Moon files = 15 embeddings
