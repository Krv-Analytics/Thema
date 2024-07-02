.. _embeddings :

Embeddings
=====================================

This guide walks through using ``thema`` for Low Dimensional Embeddings and Dimensionality Reduction of clean datasets.

.. note::
    Embeddings are dependant on the creation of a ``Moon`` object. Please see the `Data Preprocessing Guide <preprocessing.html#preprocessing-guide>`_ for more info on preprocessing steps and the :ref:`outerSystem` API documentation for info on ``thema.multiverse.inner.Moon`` objects containing cleaned, encoded, and scaled data.


.. _projHandling:

Projection Handling with Oort Class
-----------------------------------

The Oort class in Thema provides functionality for handling projections. When using the Oort class for projections:

1. **Output Data**: Upon execution, the Oort class will create a directory as specified in the ``params.yaml`` configuration file. This directory serves as the designated location for storing all projection-related data. The Oort class does not store the projection data within its own instance. Instead, it manages and organizes the data within the designated directory. This approach ensures that all generated data, such as output files or processed results, are stored systematically for further analysis or usage.

2. **Configuration**: The ``params.yaml`` file serves as the central configuration point for specifying parameters for generating multiple projections. This file provides comprehensive control over hyperparameter selection, facilitating grid search exploration to optimize UMAP embeddings. Users can customize within the ``params.yaml`` to precisely tailor the generated data storage and algorithm behavior to their project requirements.

.. note::

    Projections are `embeddings` of your data.

.. _umap: 

UMAP Embeddings
---------------

The ``params.yaml`` file allows you to configure various parameters for generating UMAP embeddings using the ``Oort`` class. UMAP (Uniform Manifold Approximation and Projection) is a dimensionality reduction technique that can be customized through several parameters.

YAML Configuration for UMAP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- **Oort**: The top-level key indicating the configuration for the ``Oort`` class.
- **umap**: Contains the parameters for configuring the UMAP embeddings.
- **nn**: Specifies the number of nearest neighbors to consider for each point. Two values are provided (2 and 4), indicating a range to explore during grid search.
- **minDist**: Specifies the minimum distance between points in the embedding space. Multiple values are provided below, indicating a grid-search over multiple hyperparameters.
- **dimensions**: Specifies the number of dimensions for the UMAP embedding. A value of 2 indicates a 2D embedding.
- **seed**: Specifies the random seed for reproducibility. A value of 32 is provided.
- **projectiles**: Lists the dimensionality reduction methods to be used. Here, ``umap`` is specified.

Here is an example of the relevant section in the ``params.yaml`` file:

.. _firstYaml:

.. code-block:: yaml

    Oort:
        umap:
            nn:
            - 2
            - 4
            minDist:
            - 0.1
            - 0.25
            dimensions:
            - 2
            seed:
            - 32
        projectiles:
            - umap

The parameters listed under the ``umap`` key are used in a grid search to generate multiple UMAP embeddings. Each grid search iteration will create a UMAP embedding based on different combinations of the provided parameter values. This allows you to explore a variety of configurations to identify the best one for your data.

For example, the grid search will consider the following combinations of parameters:

- nn: 2, minDist: 0.05, dimensions: 2, seed: 32
- nn: 4, minDist: 0.05, dimensions: 2, seed: 32
- nn: 2, minDist: 0.1, dimensions: 2, seed: 32
- nn: 4, minDist: 0.1, dimensions: 2, seed: 32

By adding more values to each parameter, you can expand the grid search to include more combinations. 

By performing a grid search with these parameters, you can systematically evaluate different UMAP configurations to find the one that best suits your data and objectives. Feel free to add more parameters to the grid search to explore additional configurations and improve the quality of your UMAP embeddings.

Instantiate Oort with UMAP Embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, import the correct module:

.. code-block:: python

   from thema.multiverse import Oort

Here, we instantiate a simple Oort using the :ref:`yaml configs<firstYaml>` shown above:

.. code-block:: python

   yaml = "<PATH TO YOUR params.yaml>"
   oort = Oort(YAML_PATH=yaml)
   oort.fit()

The ``outDir`` member variable of an ``Oort`` object stores the location of newly created projection files:

Other Embeddings
----------------

Thema also supports PCA and T-SNE embeddings. The `params.yaml` file allows you to configure parameters for generating these embeddings alongside, or instead of, UMAP.

TSNE Parameters
^^^^^^^^^^^^^^^

.. hint::
    - An interactive overview of the key differences between UMAP and t-SNE projections: `UMAP vs. t-SNE <https://pair-code.github.io/understanding-umap/#:~:text=UMAP%20vs%20t%2DSNE%2C%20revisited,structure%20in%20the%20final%20projection.>`_

- **perplexity**: Controls the number of effective neighbors used in TSNE. Example value: 2.

- **dimensions**: Specifies the number of dimensions for TSNE embeddings. Example value: 2.

- **seed**: Specifies the random seed for reproducibility in TSNE. Example value: 32.

.. code-block:: yaml
    
    Oort:
        tsne: 
            perplexity: 
            - 2 
            - 4
            dimensions:
            - 2
            seed:
            - 32
            - 42
        projectiles:
            - tsne

PCA Parameters
^^^^^^^^^^^^^^^

- **dimensions**: Specifies the number of dimensions for PCA embeddings. Example value: 2.

- **seed**: Specifies the random seed for reproducibility in PCA. Example value: 32.

.. code-block:: yaml
    
    Oort:
        pca: 
            dimensions:
            - 2
            - 3
            - 4
            seed:
            - 32
        projectiles:
            - pca

Example YAML with Multiple Embedding Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To include multiple projections in your Oort object crearion, modify the `projectiles` section in your ``params.yaml`` file to include multiple embeddings (and dont forget to define the parameter). 

.. note::
    
    This is `just` the ``Oort`` section of the yaml, and does not affect any other keys (such as the ``Planet`` yaml key controlling data preprocessing, for example).

.. code-block:: yaml

    Oort:
        umap:
            nn:
            - 2
            minDist:
            - 0.1
            dimensions:
            - 2
            seed:
            - 32
        tsne: 
            perplexity: 
            - 2 
            dimensions:
            - 2
            seed:
            - 32
        pca: 
            dimensions:
            - 2
            seed:
            - 32
        projectiles:
            - umap
            - tsne
            - pca 