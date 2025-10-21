.. _outerSystem:

Outer System
============

The ``thema.multiverse.system.outer`` module handles dimension reduction.
- ``Comet`` is the abstract base for projectors.
- ``Oort`` runs a grid over projectors and writes results to disk.

.. _comet-class:

Comet Base Class
----------------

.. automodule:: thema.multiverse.system.outer.comet
   :members:
   :undoc-members:
   :show-inheritance:

.. _projectiles:

Projectors
----------

Currently supported projectors:

- t-SNE (tsneProj)
- PCA (pcaProj)

.. _TSNE:

t-SNE
^^^^^
.. automodule:: thema.multiverse.system.outer.projectiles.tsneProj
   :members:
   :undoc-members:
   :show-inheritance:

.. _principCA:

PCA
^^^
.. automodule:: thema.multiverse.system.outer.projectiles.pcaProj
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. _oort-class:

Oort Class
----------

.. mermaid::

   classDiagram
     Core <|-- Comet
     Comet <|-- tsneProj
     Comet <|-- pcaProj
     Core <|-- Oort
     class Comet {
       <<Abstract>>
       +projectionArray
       +fit()
       +save()
     }
     class tsneProj {
       +perplexity
       +dimensions
       +seed
       +fit()
     }
     class pcaProj {
       +dimensions
       +seed
       +fit()
     }
     class Oort {
       +params
       +cleanDir
       +outDir
       +fit()
       +writeParams_toYaml()
     }
     Oort o--> Comet : instantiate projectiles

.. automodule:: thema.multiverse.system.outer.oort
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
