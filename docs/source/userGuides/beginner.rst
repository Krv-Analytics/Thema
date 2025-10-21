.. _getting_started:

==========================
Getting Started
==========================

Complete end-to-end workflow using uv for environment management.

Prepare Your Data
-----------------

Thema accepts CSV, pickle, or parquet files:

.. code-block:: python

   import pandas as pd

   df = pd.read_csv("raw_data.csv")
   df.to_pickle("data.pkl")  # Recommended for mixed types

Create Configuration
--------------------

Save as **params.yaml** in your project directory:

.. code-block:: yaml

   runName: my_analysis
   data: /absolute/path/to/data.pkl
   outDir: ./outputs

   Planet:
     scaler: standard
     encoding: one_hot
     imputeColumns: auto
     imputeMethods: auto
     numSamples: 1
     seeds: auto

   Oort:
     tsne:
       perplexity: [30]
       dimensions: [2]
       seed: [42]
     pca:
       dimensions: [2]
       seed: [42]
     projectiles: [tsne, pca]

   Galaxy:
     metric: stellar_curvature_distance
     selector: max_nodes
     nReps: 2
     stars: [jmap]
     jmap:
       nCubes: [8]
       percOverlap: [0.3]
       minIntersection: [-1]
       clusterer:
         - [HDBSCAN, {min_cluster_size: 5}]

Run the Pipeline
----------------

.. code-block:: python

   from thema.thema import Thema

   T = Thema("params.yaml")
   T.genesis()

   print(f"Representatives: {T.selected_model_files}")

Inspect Outputs
---------------

Pipeline creates:

.. code-block:: text

   outputs/
   └── my_analysis/
       ├── clean/          # Preprocessed data (Moon files)
       ├── projections/    # Embeddings (Comet files)
       └── models/         # Graphs (Star files)

Load representative graphs:

.. code-block:: python

   import pandas as pd

   for file in T.selected_model_files:
       star = pd.read_pickle(file)
       print(f"Nodes: {star.starGraph.nNodes}, Components: {star.starGraph.nComponents}")

Visualize Graph Landscape (Optional)
------------------------------------

View relationships between all generated graphs:

.. code-block:: python

   import matplotlib.pyplot as plt

   coords = T.galaxy.get_galaxy_coordinates()  # MDS projection of distance matrix
   plt.scatter(coords[:, 0], coords[:, 1], alpha=0.6)
   plt.title("Model Space")
   plt.show()
