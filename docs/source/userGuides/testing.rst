Testing
=======

Run tests
---------

- All tests use pytest.
- No linters or formatters are configured in this repo.

Commands
^^^^^^^^

.. code-block:: bash

   uv run pytest -q

.. code-block:: bash

   uv run pytest --cov=thema --cov-report=term-missing -q
