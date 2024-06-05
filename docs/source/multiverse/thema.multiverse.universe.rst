.. _thema-universe:

THEMA: Universe
===============

The ``thema.multiverse.universe`` module provides functionality for creating and managing a universe of different unsupervised projections of data. At the core of THEMA lies the concept of Stars, which are base class templates for atlas (graph) construction algorithms.

.. grid:: 2
   :gutter: 4

   .. grid-item::

      .. _thema-stars:

      **Stars**

      The Star class provides a structured approach to creating atlas representations of data. At the core of the Star class is the concept of a Simple Targeted Atlas Representation (STAR), which serves as a base template for constructing atlas representations of data. See :ref:`JMAP Star Class<jmapStar>` for more info on a specific Star instance.

   .. grid-item::

      .. _thema-galaxy:

      **Galaxy**

      The Galaxy class is a core component of the THEMA package, designed to manage and explore a vast space of data representations. It serves as a container for star objects. By organizing these projections into a galaxy, users are able to democratize model selection and quantify agreement between models.

   .. grid-item::

      .. _thema-geodesics:

      **Geodesics**

      The Geodesics module provides functionality to compute pairwise distance matrices between multiple graph representations of your data. It is used by the Galaxy class to construct a galaxy space of stars, specifically to derive metrics of agreement between models and democratize model selection.

.. raw:: html

   <hr style="border: none; border-top: 2px solid #007bff; margin: 20px 0;">

Architecture
----------------

.. code-block:: bash

    thema.multiverse.universe
    ├── galaxy.py
    ├── geodesics.py
    ├── star.py
    ├── starGraph.py
    ├── starSelectors.py
    └── stars
        └── jmapStar.py


Module Contents
---------------

.. toctree::
   :maxdepth: 2

   multiverse.stars
   multiverse.galaxy
   multiverse.geodesics