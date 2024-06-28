.. _stars:

Stars
==========================

THEMA is a Python package designed to provide a structured approach to creating and managing a universe of different unsupervised projections of data. At the core of THEMA lies the concept of ``Stars`` which are base class templates for atlas (graph) construction algorithms.

**Simple Targeted Atlas Representation (STAR)**
    The starGraph class serves as a base template for constructing atlas representations of data. By enforcing structure on data management and graph generation, starGraph enables a universal procedure for generating these objects. This base class provides a foundation for implementing various graph generation algorithms.

**JMAP Star Class**
    The :ref:`JMAP Star Class<jmapStar>` is a custom implementation of a Kepler Mapper (K-Mapper) algorithm into a Star object. This implementation allows users to explore the topological structure of their data using the Mapper algorithm, which is a powerful tool for visualizing high-dimensional data. The JMAP Star Class generates a graph representation of projections using Kepler Mapper, offering insights into the complex relationships within the data.

Star Functionality
-----------------

Star Class
^^^^^^^^^^^^^
.. automodule:: thema.multiverse.universe.star
   :members:
   :undoc-members:
   :show-inheritance:

StarGraph Class
^^^^^^^^^^^^^^^^
.. automodule:: thema.multiverse.universe.starGraph
   :members:
   :undoc-members:
   :show-inheritance:

.. _jmapStar:

jmapStar Class
^^^^^^^^^^^^^
.. automodule:: thema.multiverse.universe.stars.jmapStar
   :members:
   :undoc-members:
   :show-inheritance: