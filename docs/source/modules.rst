.. _user_guide:

==========
User Guide
==========

The User Guide covers all of pandas by topic area. Each of the subsections
introduces a topic (such as "working with missing data"), and discusses how
pandas approaches the problem, with many examples throughout.

Users brand-new to pandas should start with 

For a high level summary of the pandas fundamentals, see 

Further information on any specific method can be obtained in the


How to read these guides
------------------------
In these guides you will see input code inside code blocks such as:

::

    import pandas as pd
    pd.DataFrame({'A': [1, 2, 3]})


or:

.. ipython:: python

    import pandas as pd
    pd.DataFrame({'A': [1, 2, 3]})

The first block is a standard python input, while in the second the ``In [1]:`` indicates the input is inside a `notebook <https://jupyter.org>`__. In Jupyter Notebooks the last line is printed and plots are shown inline.

For example:

.. ipython:: python

    a = 1
    a
is equivalent to:

::

    a = 1
    print(a)



Guides
-------

.. If you update this toctree, also update the manual toctree in the
   main index.rst.template

.. toctree::
    :maxdepth: 2
