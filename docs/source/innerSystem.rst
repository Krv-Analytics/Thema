Inner System
==========================

Overview
-----------------
Thema's ``multiverse.system.inner`` submodule offers two key classes: **Moon** and **Planet**. These classes facilitate preprocessing of tabular data, easing the transition from raw datasets to cleaned, imputed, and ready-for-analysis dataframes.

:ref:`Moon Class: <moon-class>`
    This class sits close to the original dataset, focusing on preprocessing steps crucial for downstream analysis. It streamlines data cleaning and aids in the creation of an *imputeData* dataframe, which is a formatted version of the data suitable for in-depth exploration. The Moon class supports standard sklearn.preprocessing operations such as scaling and encoding, with an emphasis on imputation methods for handling missing values.


:ref:`Planet Class: <planet-class>`
    Operating within the inner system, the Planet class manages the transformation of raw tabular data into processed, scaled, encoded, and complete datasets. Its key function is handling datasets with missing values, utilizing a ``Moon`` *imputeData* dataframe using random sampling to fill in these gaps while exploring the distribution of possible missing values.

Both classes are integral to Thema's data preprocessing capabilities, providing efficient solutions for common data cleaning and imputation tasks.

Moon Class
-----------------
.. _moon-class:

.. note::
    ``thema.multiverse.system.inner.Moon`` handles data **preprocessing**, moving from raw, tabular datasets to cleaned, scaled, and encoded python-friendly formats.

.. automodule:: thema.multiverse.system.inner.moon
   :members:
   :undoc-members:
   :show-inheritance:

Planet Class
-----------------
.. _planet-class:
.. note::
    ``thema.multiverse.system.inner.Planet`` handles data **transformation**, managing and exploring the distribution of possible missing values.

.. automodule:: thema.multiverse.system.inner.planet
   :members:
   :undoc-members:
   :show-inheritance:

Inner System Utils
-----------------
.. automodule:: thema.multiverse.system.inner.inner_utils
   :members:
   :undoc-members:
   :show-inheritance: