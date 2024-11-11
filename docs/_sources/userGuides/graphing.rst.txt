.. _graphing:

Graph Construction
=================================

THEMA manages and explores a diverse range of data representations by organizing them into a unified framework. Graph-based pairwise distances quantify similarity between data points in graph models, facilitating:

- **Validation:** Evaluate model effectiveness by measuring similarities in data structures.
- **Objective Insights:** Gain quantitative metrics for community detection, anomaly detection, and more.
- **Democracy in Analysis:** Ensure unbiased analysis with multiple perspectives on data complexities.

Managed by the :ref:`galaxy` and :ref:`jmap` classes, graph construction currently supports using the `Kepler Mapper Algorithm <https://kepler-mapper.scikit-tda.org/en/latest/>`_ to construct graph representations of your data. This creates multiple Jmap Star objects using the parameters seen in the YAML above.

.. -graphConstruction:

Choices in Graph Construction
------------------------------

.. _fullYamlTest:

.. code-block:: yaml

    Galaxy:
        metric: stellar_kernel_distance
        selector: random
        nReps: 3
        stars:
            - jmap 
        jmap:
            nCubes:
            - 5
            - 10
            percOverlap:
            - 0.2
            - 0.5
            minIntersection:
            - -1
            clusterer:
            - [HDBSCAN, {min_cluster_size: 2}] 
            - [HDBSCAN, {min_cluster_size: 10}]

.. _graphComparison:

Graph Comparison
----------------

Managed by the :ref:`geodesics` class, graph comparison uses what we call `Stellar Kernel Distance` to quantify similarity between and across multiple graph-based representations of your data.

Thema uses the below to compute and quantify similarity across multiple graph models:

**Stellar Kernel Distance**

The "stellar kernel distance" refers to the measurement of similarity or dissimilarity between graphs based on their structural properties. It calculates how closely related or different two graphs are, considering their connectivity and node attributes.

**Grakel Kernels**

Grakel kernels are mathematical functions used to compute pairwise distances between graphs. These kernels transform graph structures into numerical representations, enabling comparison and analysis across a dataset. 

Example
-----------------------------

Here, we run through a simplified version of the Thema pipeline to create multiple graph representations of our dataset! In this example, we use the `scikit-learn breast cancer dataset <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html>`_ to demonstrate Thema's functionality.

Step 1: YAML Setup
^^^^^^^^^^^^^^^^^^^^

For this example, we construct a very simple hyperameter space to example the breast cancer dataset. Here is what our YAML looks like:

.. code-block:: yaml

    runName: demoFiles
    data: <Path to your Raw Data>
    outDir: <Path to your Out Directory>
    Planet:
        scaler: standard
        encoding: one_hot
        dropColumns: None
        imputeColumns: None
        imputeMethods: None
        numSamples: 1
        seeds: auto

    Oort:
        umap:
            nn:
            - 5
            - 10
            - 50
            minDist:
            - 0.05
            - 0.1
            - 0.15
            - 0.5
            dimensions:
            - 2
            seed:
            - 32
            - 50
        projectiles:
            - umap

    Galaxy:
        metric: stellar_kernel_distance
        selector: random
        nReps: 2
        stars:
            - jmap 
        jmap:
            nCubes:
            - 2
            - 5
            - 15
            - 30
            percOverlap:
            - 0.05
            - 0.1
            - 0.5
            minIntersection:
            - -1
            clusterer:
            - [HDBSCAN, {min_cluster_size: 2}] 
            - [HDBSCAN, {min_cluster_size: 5}] 
            - [HDBSCAN, {min_cluster_size: 10}] 

Step 2: Preprocessing
^^^^^^^^^^^^^^^^^^^^^

Handle filepaths -- not necessary when running locally!

.. ipython:: python

    import sys, os
    sys.path.insert(0, os.path.abspath('.'))
    yaml = os.path.join(os.path.abspath('.'),'source', 'userGuides', 'demoFiles', 'params.yaml')

    if not os.path.isfile(yaml):
        # Print the current working directory
        cwd = os.getcwd()
        raise FileNotFoundError(f"YAML parameter file could not be found: {yaml}\nCurrent working directory: {cwd}")

See the :ref:`preprocessing` for a detailed look at the steps involved in data preprocessing. In this specific example, the breast cancer dataset being used has no missing values and most pre-processing steps required for more `organic`, real world datasets have already been taken by scikit-learn.

.. ipython:: python

    from thema.multiverse import Planet

    planet = Planet(YAML_PATH=yaml)
    planet.fit()

Step 3: Embedding
^^^^^^^^^^^^^^^^^^^^^

See the :ref:`embeddings` User Guide for more information on embedding selections and hyperameter selection.

.. ipython:: python

    from thema.multiverse import Oort

    oort = Oort(YAML_PATH=yaml)
    oort.fit()

Step 4: Graph Construction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now we get to the real meat and potatoes! This generates graph based on the parameters shown in the :ref:`demo YAML <fullYamlTest>` above.

.. ipython:: python

    from thema.multiverse import Galaxy

    galaxy = Galaxy(YAML_PATH=yaml)
    galaxy.fit()

Step 5: Graph Model Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _mds:

Plotting MDS
~~~~~~~~~~~~~~~

MDS (`Multidimensional Scaling <https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html>`_) is a statistical technique to visualize data similarities or dissimilarities in a lower-dimensional space while preserving relative distances as accurately as possible. It transforms complex distance data into a visual representation, revealing relationships and patterns that are hard to discern in high-dimensional spaces.

Key Benefits:

- **Visual Insight**: Simplifies complex data relationships for easier interpretation and analysis.
- **Comparison**: Facilitates comparing datasets or models based on distance metrics.
- **Communication**: Communicates findings visually, aiding in decision-making and stakeholder engagement across disciplines.

.. ipython:: python

    model_representatives = galaxy.collapse()
    galaxy.show_mds()

.. raw:: html

   <div style="width: 100%; overflow: hidden;">
       <iframe src="/Users/gathrid/Repos/thema_light/Thema/docs/source/userGuides/demoFiles/MDS.html"
               style="border: none; width: 100%; height: 80vh;">
       </iframe>
   </div>

.. hint::

    Each point on the above plot represents an entire graph model. Background coloring represents the density of graph models.

Based on the plot, we can decide how many graph representatives we would like to select and analyze. In this case, we will look at 2 graph models - 1 from the highest density region and 1 from the 2nd highest region of model density.

Based on the plot, we would *not* like to select models that are outside the density coloring, as these models are unrepresentative of the dataset and the hyperameters used to produce them represent a poor selection based on the dataset.

.. _selectingModels:

Selecting Models
~~~~~~~~~~~~~~~~~~

You can either use built-in functionality in the params.yaml to select representative models, based on the ``selector``  and ``nReps`` arguments passed in the yaml:

.. ipython:: python

    model_representatives = galaxy.collapse()
    model_representatives

Or you can select your own representatives by index from the ``show_mds()`` plot. Here we select index 311 (the bottom-most point) and 98 (in the middle of a high-density region):

.. ipython:: python

    import os
    
    selection1, selection2 = os.listdir(galaxy.outDir)[311], os.listdir(galaxy.outDir)[98]
    print(f"\nSelection 1: {selection1}\n\nSelection 2: {selection2}")

And that is it! Now you have searched the hyperameter space to create a landscape (galaxy!) of graph models (stars!) and selected representative models from the distribution. More advanced similarity metrics can be used here to select graphs - contact us at `Krv Analytics <https://krv-analytics.us>`_ for more info!

See the next guide for information on analyzing selected graph models.