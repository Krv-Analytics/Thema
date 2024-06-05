.. _systems:

THEMA: Systems
==========================

The ``thema.multiverse.systems`` module comprises the first segment of the Thema pipeline, focusing on data cleaning/processing and data projection.

.. grid:: 2
   :gutter: 4

   .. grid-item::

      .. _thema-stars:

      **Inner System**

      The Inner System encompasses functionality for cleaning, processing, encoding, scaling, and imputing data. It facilitates the transition from tabular data to internally-managed Python-friendly files, enhancing ease of use.

   .. grid-item::

      .. _thema-galaxy:

      **Outer System**

      The Outer System includes functionality for comprehending the projection space of your data. It enables grid searching of a hyperparameter space for multiple unsupervised projection and dimensionality reduction methods.

.. raw:: html

   <hr style="border: none; border-top: 2px solid #007bff; margin: 20px 0;">


Architecture
----------------

.. code-block:: bash
    
    thema.multiverse.systems
    ├── inner
    │   ├── inner_utils.py
    │   ├── moon.py
    │   └── planet.py
    └── outer
        ├── comet.py
        ├── oort.py
        └── projectiles
            ├── pcaProj.py
            ├── tsneProj.py
            └── umapProj.py


Module Contents
---------------

.. toctree::
   :maxdepth: 2

   innerSystem
   outerSystem
