.. _index:

=====
Thema
=====

**Topological Hyperparameter Evaluation and Mapping Algorithm**

Thema explores model space through systematic preprocessing, dimensionality reduction, and graph construction, then selects representative models using topological distances.

Quick Links
-----------

.. grid:: 1 2 3 3
   :gutter: 3
   :padding: 2 2 0 0

   .. grid-item-card:: :octicon:`rocket` Quickstart
      :link: quickstart
      :link-type: ref
      :class-card: intro-card
      :shadow: md

      Run the full pipeline in 5 minutes with minimal YAML configuration.

   .. grid-item-card:: :octicon:`book` User Guide
      :link: user_guide
      :link-type: ref
      :class-card: intro-card
      :shadow: md

      Complete guides from beginner to advanced customization.

   .. grid-item-card:: :octicon:`code` API Reference
      :link: thema
      :link-type: ref
      :class-card: intro-card
      :shadow: md

      Full API documentation for Planet, Oort, Galaxy classes.

What is Thema?
--------------

Thema generates multiple data representations through systematic hyperparameter grids, then identifies representative models using curvature-based graph distances. Instead of guessing at preprocessing choices or embedding parameters, explore the space of possibilities and let topological methods guide selection.

**Pipeline Stages**

1. **Planet**: Clean, encode, scale, and impute tabular data → Moon files
2. **Oort**: Generate embeddings (t-SNE, PCA) across parameter grids → Comet files
3. **Galaxy**: Build Mapper graphs, compute distances, cluster, and select representatives → Star files


Typical Workflow
----------------


.. mermaid::
   
   graph LR
      subgraph Input
         A[params.yaml + raw dataset]
      end
      
      subgraph "Stage 1: Preprocess"
         B["Planet: clean + impute"]
         M1["Moon 1"]
         M2["Moon 2"]
         M3["Moon N"]
      end
      
      subgraph "Stage 2: Project"
         C["Oort: dimensionality reduction"]
         P1["t-SNE Comet"]
         P2["PCA Comet"]
      end
      
      subgraph "Stage 3: Graph"
         D["Galaxy: mapper graphs"]
         S1["StarGraph 1"]
         S2["StarGraph 2"]
      end
      
      subgraph "Stage 4: Select"
         F["Filters"]
         R["Representatives"]
      end
      

      A --> B
      B --> M1
      B --> M2
      B --> M3
      M1 --> C
      M2 --> C
      M3 --> C
      C --> P1
      C --> P2
      P1 --> D
      P2 --> D
      D --> S1
      D --> S2
      S1 --> F
      S2 --> F
      F --> R

      style Input fill:#f9f9f9,stroke:#999
      style B fill:#D9EDF7,stroke:#31708F,stroke-width:2px
      style C fill:#D9EDF7,stroke:#31708F,stroke-width:2px
      style D fill:#D9EDF7,stroke:#31708F,stroke-width:2px
      style F fill:#D9EDF7,stroke:#31708F,stroke-width:2px



**Option 1: YAML-Driven (Recommended for most users)**

.. code-block:: python

   from thema.thema import Thema

   T = Thema("params.yaml")
   T.genesis()  # Runs Planet → Oort → Galaxy
   print(T.selected_model_files)

**Option 2: Programmatic**

.. code-block:: python

   from thema.multiverse import Planet, Oort, Galaxy

   planet = Planet(data="data.pkl", scaler="standard", ...)
   planet.fit()

   oort = Oort(cleanDir="./clean", params={"tsne": {...}})
   oort.fit()

   galaxy = Galaxy(projDir="./projections", params={"jmap": {...}})
   galaxy.fit()
   representatives = galaxy.collapse()

Key Features
------------

**Robust Preprocessing**
   Multiple imputation strategies, encoding options, and scaling methods with reproducible seeds.

**Grid Search Over Embeddings**
   Systematic exploration of t-SNE perplexities, PCA dimensions, and projection parameters.

**Topological Graph Construction**
   Kepler Mapper implementation with configurable covers, clustering algorithms, and overlap parameters.

**Curvature-Based Selection**
   Graph distance metrics using Forman-Ricci and Ollivier-Ricci curvatures for model comparison.

**Flexible Filtering**
   Built-in filters (coverage, component count, graph size) plus custom filter support.

Installation
------------

.. code-block:: bash

   pip install thema

For development:

.. code-block:: bash

   git clone https://github.com/Krv-Analytics/thema.git
   cd thema
   uv sync --extra dev --extra docs

Supports Python 3.10, 3.11, 3.12.

Logging
-------

Enable detailed logging for debugging:

.. code-block:: python

   import thema
   thema.enable_logging('DEBUG')  # or 'INFO'

Use ``'DEBUG'`` for verbose output or ``'INFO'`` for standard progress messages.

Next Steps
----------

- :ref:`Quickstart <quickstart>` - Run your first pipeline
- :ref:`Getting Started <getting_started>` - Complete tutorial
- :ref:`Preprocessing <preprocessing>` - Data cleaning with Planet
- :ref:`Embeddings <embeddings>` - Dimensionality reduction with Oort
- :ref:`Graphs & Selection <graphing>` - Mapper graphs with Galaxy
- :ref:`Best Practices <best_practices>` - Recommended workflows and troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   thema

.. toctree::
   :maxdepth: 2
   :caption: Guides
   :hidden:

   user_guide
   userGuides/cosmic_graph

References
----------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
