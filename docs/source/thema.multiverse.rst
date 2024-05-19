THEMA: Multiverse
==========================
   
Overview 
------------------------------
The ``thema.multiverse`` module provides an advanced framework for analyzing the multiverse of unsupervised representations of your data. It leverages topology and machine learning to distill metrics of agreement and trustworthiness across the space of models. This helps you understand how different representations align and vary, offering insights into the reliability and consistency of your models.

By evaluating hyperparameters and their impact on model performance and alignment, ``thema.multiverse`` provides a robust, data-driven approach to selecting and validating the best representations for your specific needs. With THEMA, you can confidently navigate the complex landscape of unsupervised learning, ensuring that your models are both reliable and insightful.

The *multiverse* contains the ``system`` and ``universe`` submodules.

.. code-block:: bash

   multiverse
      ├── system
      │   ├── inner
      │   │
      │   └── outer
      └── universe
         ├── galaxy.py
         ├── geodesics.py
         ├── star.py
         ├── starGraph.py
         ├── starSelectors.py
         └── stars


Core Subpackages
------------------------------


.. toctree::
   :maxdepth: 2

   thema.multiverse.system
   thema.multiverse.universe
