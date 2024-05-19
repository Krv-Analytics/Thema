THEMA: Systems
==========================

The ``thema.multiverse.systems`` module comprises the first segment of the Thema pipeline, focusing on data cleaning/processing and data projection.

**Inner System**
    The Inner System encompasses functionality for cleaning, processing, encoding, scaling, and imputing data. It facilitates the transition from tabular data to internally-managed Python-friendly files, enhancing ease of use.

**Outer System**
    The Outer System includes functionality for comprehending the projection space of your data. It enables grid searching of a hyperparameter space for multiple unsupervised projection and dimensionality reduction methods.

Module Contents
^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 2

   innerSystem
   outerSystem