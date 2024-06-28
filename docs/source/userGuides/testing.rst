Testing
=========

.. ipython:: python

   import numpy as np
   import pandas as pd

Fundamentally, **data alignment is intrinsic**. The link
between labels and data will not be broken unless done so explicitly by you.

We'll give a brief intro to the data structures, then consider all of the broad
categories of functionality and methods in separate sections.

.. _basics.series:

Series
------

:class:`Series` is a one-dimensional labeled array capable of holding any data
type (integers, strings, floating point numbers, Python objects, etc.). The axis
labels are collectively referred to as the **index**. The basic method to create a :class:`Series` is to call: