.. _probe:

THEMA: Probe
===================

The Probe module is designed to "probe" your data, much like a space probe explores and gathers information in space. It provides tools and functionality to analyze, visualize, and understand complex data structures within the Thema framework. The Probe module includes classes and utilities for exploring data relationships, visualizing data structures, and gaining insights into the underlying patterns and structures of your data.

.. grid:: 1 2 2 2
   :gutter: 4
   :padding: 2 2 0 0

   .. grid-item:: **Telescope**

      The ``Telescope`` class provides visualization capabilities for democratically selected star instances. It is designed to meet various visualization needs for analyzing and understanding data relationships within the Thema framework.

   .. grid-item:: **Observatory**

      The ``jmapObservatory`` class is a custom observatory designed specifically for viewing JMAP Stars. It extends the functionality of the base ``Observatory`` class, providing additional methods and attributes tailored to the graph models outputted by JMAP Star.

   .. grid-item:: **Probe Utils**

      Utility functionality to help with Probe functionality (to probe your data!).

.. raw:: html

   <hr style="border: none; border-top: 2px solid #007bff; margin: 20px 0;">

Architecture
----------------

.. code-block:: bash

   thema.probe
   ├── data_utils.py
   ├── observatories
   │   └── jmapObservatory.py
   ├── observatory.py
   ├── telescope.py
   └── visual_utils.py


Submodules
----------

.. toctree::
   :maxdepth: 2

   thema.probe.telescope
   thema.probe.observatory
   thema.probe.utils
