.. _thema-universe:

THEMA: Universe
===============

The ``thema.multiverse.universe`` package builds graphs from embeddings, compares them, and picks representatives.

.. grid:: 1 2 2 2
   :gutter: 4
   :padding: 2 2 0 0

   .. grid-item:: **Stars**

      Turn projections into graphs. See :ref:`jmapStar` for Kepler Mapper.

   .. grid-item:: **Galaxy**

      Generate many Stars, compare, cluster, and select.

   .. grid-item:: **Geodesics**

      Curvature-based graph distances for comparisons.

Architecture
------------

.. code-block:: bash

    thema.multiverse.universe
    ├── galaxy.py
    ├── geodesics.py
    ├── star.py
    ├── utils
    │   ├── starFilters.py
    │   ├── starGraph.py
    │   ├── starHelpers.py
    │   └── starSelectors.py
    └── stars
        ├── gudhiStar.py
        ├── jmapStar.py
        └── pyballStar.py


Module Contents
---------------

.. toctree::
   :maxdepth: 2

   multiverse.stars
   multiverse.galaxy
   multiverse.geodesics
   universe/utils/starFilters
   universe/utils/starSelectors
