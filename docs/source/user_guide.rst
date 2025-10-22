.. _user_guide:

==========
User Guide
==========

Thema provides a topological data analysis pipeline that transforms raw tabular data into representative graph models through preprocessing, dimensionality reduction, and Mapper graph construction.

How to Use This Guide
---------------------

This guide is organized for both learning and reference:

- **New users**: Start with :ref:`Installation <installation>` → :ref:`Quickstart <quickstart>` → :ref:`Getting Started <getting_started>`
- **YAML-driven workflows**: See :ref:`Quickstart <quickstart>` and :ref:`Getting Started <getting_started>`
- **Programmatic control**: See :ref:`Programmatic Pipeline <programmatic>` and component guides (:ref:`Preprocessing <preprocessing>`, :ref:`Embeddings <embeddings>`, :ref:`Graphs & Selection <graphing>`)
- **Parameter tuning**: See :ref:`Tuning and Selection <tuning>`
- **Advanced customization**: See :ref:`Customizing Thema <advanced>`

Guides by Task
--------------

**Getting Started**

- :doc:`userGuides/installation` - Install Thema via pip or set up development environment
- :doc:`userGuides/quickstart` - Run the full pipeline with a minimal YAML configuration
- :doc:`userGuides/beginner` - Complete walkthrough from setup to results with uv

**Pipeline Components**

- :doc:`userGuides/preprocessing` - Clean, encode, scale, and impute data with Planet
- :doc:`userGuides/embedding` - Generate low-dimensional projections with Oort (t-SNE, PCA)
- :doc:`userGuides/graphing` - Build Mapper graphs and select representatives with Galaxy

**Workflows**

- :doc:`userGuides/programmatic` - Build pipelines programmatically without YAML, with complete parameter reference
- :doc:`userGuides/intermediate` - Fine-tune parameter grids, apply filters, and optimize selection strategies
- :doc:`userGuides/advanced` - Write custom filters and graph builders, scale to large datasets

**Reference**

- :doc:`userGuides/best_practices` - Recommended workflows, parameter choices, and troubleshooting
- :doc:`userGuides/testing` - Test suite information
- :doc:`overview` - High-level architecture and terminology

Quick Navigation
----------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Task
     - Guide
   * - Install Thema
     - :doc:`userGuides/installation`
   * - Run first pipeline
     - :doc:`userGuides/quickstart`
   * - Understand parameters
     - :doc:`userGuides/preprocessing`, :doc:`userGuides/embedding`, :doc:`userGuides/graphing`
   * - Build without YAML
     - :doc:`userGuides/programmatic`
   * - Tune hyperparameters
     - :doc:`userGuides/intermediate`
   * - Filter and select models
     - :doc:`userGuides/intermediate`
   * - Write custom filters
     - :doc:`userGuides/advanced`
   * - Optimize performance
     - :doc:`userGuides/best_practices`
   * - Troubleshoot issues
     - :doc:`userGuides/best_practices`

.. toctree::
   :maxdepth: 1
   :hidden:

   overview
   userGuides/installation
   userGuides/quickstart
   userGuides/beginner
   userGuides/programmatic
   userGuides/intermediate
   userGuides/advanced
   userGuides/preprocessing
   userGuides/embedding
   userGuides/graphing
   userGuides/best_practices
   userGuides/testing
