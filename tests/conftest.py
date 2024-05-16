# File:tests/conftest.py
# Last Updated: 04-04-24
# Updated By: SW

import os
import sys


def pytest_configure(config):
    """
    Called before test collection starts.
    """
    # Add the root directory of the project to the Python path
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(root_path)
