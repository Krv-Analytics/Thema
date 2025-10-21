.. _systems:

THEMA: System
=============

The ``thema.multiverse.system`` package forms the first half of the pipeline: cleaning/imputation (Inner) and projection (Outer).

.. grid:: 1 2 2 2
   :gutter: 4
   :padding: 2 2 0 0

   .. grid-item:: **Inner System**

      Clean, encode, scale, and impute raw data into Moon files.

   .. grid-item:: **Outer System**

      Run a small grid over projectors to generate embeddings.

Architecture
------------

.. code-block:: bash
    
    thema.multiverse.system
    ├── inner
    │   ├── inner_utils.py
    │   ├── moon.py
    │   └── planet.py
    └── outer
        ├── comet.py
        ├── oort.py
        └── projectiles
            ├── pcaProj.py
            └── tsneProj.py

Module Contents
---------------

.. toctree::
   :maxdepth: 2

   innerSystem
   outerSystem
