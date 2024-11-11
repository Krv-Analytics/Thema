.. _outerSystem:

Outer System
==========================

The ``thema.multiverse.system.outer`` module provides essential functionality for managing and exploring high-dimensional data through projection algorithms. At the core of this module is the ``COMET`` class, which serves as a base template for projection algorithms, enforcing a structured approach to data management and projection. This enables a universal procedure for generating projection objects.

The ``Oort`` class, a key component of the ``system.outer`` module, generates a space of projected representations of an original, high-dimensional dataset. While navigating this space of projections can be challenging, our tools facilitate easy exploration and interpretation of the data.


:ref:`Comet Class: <comet-class>`
    A base class template for projection algorithms, enforcing structure on data management and projection.
    
:ref:`Projectiles: <projectiles>`
    Support for creating ``Comet`` subclasses. Thema currently supports three projection methods:
        - Uniform Manifold Approximation and Projection for Dimension Reduction (:ref:`UniformMAP`)
        - T-distributed Stochastic Neighbor Embedding (:ref:`TSNE`)
        - Principle Component Analysis (:ref:`principCA`)

:ref:`Oort Class: <oort-class>`
    Generates a space of projected representations of high-dimensional datasets, aiding in data exploration.

Usage
^^^^^^^^^^^^^^^^^^

- **Creating and Managing Projections:** Use ``Projectiles`` class to create and manage a universe of different unsupervised projections of your data.
- **Unlocking the Multiverse of Representations** Use ``Oort`` to handle the space of multiple data projections

.. _comet-class:

Comet Base Class
-----------------

.. automodule:: thema.multiverse.system.outer.comet
   :members:
   :undoc-members:
   :show-inheritance:


.. _projectiles:

Projectiles
-----------------
Create and manage the universe of different unsupervised projections of your data. We have decided to support three standard dimensionality reduction methods:

- Uniform Manifold Approximation and Projection for Dimension Reduction: `UMAP <https://umap-learn.readthedocs.io/en/latest/>`_
- T-distributed Stochastic Neighbor Embedding: `t-SNE <https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#:~:text=t%2DSNE%20%5B1%5D%20is,and%20the%20high%2Ddimensional%20data.>`_
- Principle Component Analysis: `PCA <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_

.. hint::
    - An interactive overview of the key differences between UMAP and t-SNE projections: `UMAP vs. t-SNE <https://pair-code.github.io/understanding-umap/#:~:text=UMAP%20vs%20t%2DSNE%2C%20revisited,structure%20in%20the%20final%20projection.>`_
    
.. _UniformMAP:
UMAP
^^^^^^^^^^^^^^^^^^
.. automodule:: thema.multiverse.system.outer.projectiles.umapProj
   :members:
   :undoc-members:
   :show-inheritance:

.. _TSNE:
t-SNE
^^^^^^^^^^^^^^^^^^
.. automodule:: thema.multiverse.system.outer.projectiles.tsneProj
   :members:
   :undoc-members:
   :show-inheritance:

.. _principCA:
PCA
^^^^^^^^^^^^^^^^^^
.. automodule:: thema.multiverse.system.outer.projectiles.pcaProj
   :members:
   :undoc-members:
   :show-inheritance:


.. _oort-class:

Oort Class
-----------------

.. automodule:: thema.multiverse.system.outer.oort
   :members:
   :undoc-members:
   :show-inheritance:



