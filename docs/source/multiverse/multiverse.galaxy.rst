Galaxy
==========================

The ``Galaxy`` class is a core component of the THEMA package, designed to manage and explore a vast space of data representations. It serves as a container for star objects, each representing a unique unsupervised projection of data. By organizing these projections into a galaxy, users are able to democratize model selection and quantify agreement between models. THEMA helps ensure that the most reliable representations of your data are chosen.

A galaxy is generated from distributions of inner and outer systems, which define the parameters for the creation of star objects. This structure allows for a flexible and scalable approach to managing complex data representations.

**Key Features**

- **Star Management**: The ``Galaxy`` class maintains a collection of star objects, enabling easy access and manipulation of various data projections.
- **Search Functionality**: Users can search for specific stars within the galaxy based on defined criteria, facilitating targeted exploration of data.
- **Distribution-Based Generation**: Stars are generated from specified inner and outer system distributions, providing a customizable and systematic approach to data representation.

**Use Cases**

- **Data Exploration**: Providing a structured way to explore different unsupervised projections of data.
- **Visualization**: Enabling visualization of complex data relationships through star objects.
- **Algorithm Implementation**: Serving as a foundation for implementing and managing various graph generation algorithms, such as the Kepler Mapper algorithm used in the ``jmapStar Class``.

By leveraging the ``Galaxy`` class, users can create a comprehensive and organized space of data representations, facilitating deeper insights and more effective analysis of complex datasets.

Galaxy Class
--------------
.. automodule:: thema.multiverse.universe.galaxy
   :members:
   :undoc-members:
   :show-inheritance:
