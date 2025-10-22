.. _installation:

============
Installation
============

PyPI (End Users)
----------------

Install via pip:

.. code-block:: bash

   pip install thema

Verify:

.. code-block:: bash

   pip show thema

Development Setup (uv)
----------------------

For editable installs, testing, or building docs:

.. code-block:: bash

   # Install uv (macOS/Linux)
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # or: brew install uv

   git clone https://github.com/Krv-Analytics/thema.git
   cd thema

   # Install with development dependencies
   uv sync --extra dev --extra docs

Run Tests
---------

.. code-block:: bash

   uv run pytest -q

Build Documentation
-------------------

.. code-block:: bash

   uv run sphinx-build -b html docs/source docs/_build/html

Python Support
--------------

Requires Python 3.10, 3.11, or 3.12.
