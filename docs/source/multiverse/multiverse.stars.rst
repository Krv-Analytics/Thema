.. _stars:

Stars
=====

A Star turns your projection into a **graph** that reveals the shape of your data. Think of it 
like a map: rows that are similar get grouped into nodes, and nodes that share rows get connected 
by edges. This lets you see clusters, branches, and outliers at a glance—without needing to 
understand the math under the hood.

All three options produce the same type of output (a ``starGraph``). They differ in how much 
tuning they require and how they decide which rows belong together:

+---------------------+---------------------------+-----------------------------------------------+
| Star Type           | Best For                  | Key Parameters                                |
+=====================+===========================+===============================================+
| :ref:`jmapStar`     | Fine-grained control.     | ``n_cubes``, ``perc_overlap``, ``clusterer``  |
|                     | Tune cubes & overlap.     |                                               |
+---------------------+---------------------------+-----------------------------------------------+
| :ref:`gudhiStar`    | Auto-tuned defaults.      | ``N``, ``beta``, ``C``, ``clusterer``         |
|                     | Less manual tuning.       |                                               |
+---------------------+---------------------------+-----------------------------------------------+
| :ref:`pyballStar`   | Simplest option.          | ``EPS`` (ball radius)                         |
|                     | Just set one parameter.   |                                               |
+---------------------+---------------------------+-----------------------------------------------+

**Which should I use?**

- **New to this?** Start with :ref:`pyballStar`—set ``EPS`` and go.
- **Want sensible defaults?** Try :ref:`gudhiStar`—it estimates parameters for you.
- **Need precise control?** Use :ref:`jmapStar`—full control over resolution and clustering.


Star API
--------

Star Class
^^^^^^^^^^
.. automodule:: thema.multiverse.universe.star
   :members:
   :undoc-members:
   :show-inheritance:

starGraph Class
^^^^^^^^^^^^^^^
.. automodule:: thema.multiverse.universe.utils.starGraph
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. _jmapStar:

jmapStar Class
^^^^^^^^^^^^^^
.. automodule:: thema.multiverse.universe.stars.jmapStar
   :members:
   :undoc-members:
   :show-inheritance:

.. _gudhiStar:

gudhiStar Class
^^^^^^^^^^^^^^^
.. automodule:: thema.multiverse.universe.stars.gudhiStar
   :members:
   :undoc-members:
   :show-inheritance:

.. _pyballStar:

pyballStar Class
^^^^^^^^^^^^^^^^
.. automodule:: thema.multiverse.universe.stars.pyballStar
   :members:
   :undoc-members:
   :show-inheritance:
