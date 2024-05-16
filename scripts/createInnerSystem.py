# File: scripts/createInnerSystem.py
# Last Update: 05/15/24
# Updated By: JW

import os
import sys

import utils as ut

path_to_root = ut.get_root()
sys.path.append(path_to_root)

from thema.multiverse import Planet


def run_planet():
    """
    Run the planet simulation.

    This function creates and runs a simulation of the Inner Solar System
    using the parameters specified in `params.yaml`. It is called when using
    `invoke i`

    NOTE: Your `params.yaml` file must be in the root directory to
    work for invoke commands.

    Examples:
    >>> run_planet()
    -------------------------------------------------------------------------------------------------

     Successfully created the Inner Solar System!

    -------------------------------------------------------------------------------------------------
    """
    path_to_root = ut.get_root()
    path_to_yaml = os.path.join(path_to_root, "params.yaml")
    planet = Planet(YAML_PATH=path_to_yaml)
    planet.fit()

    print(
        "-------------------------------------------------------------------------------------------------"
    )
    print("\n\n Successfully created the Inner Solar System!  \n\n")
    print(
        "-------------------------------------------------------------------------------------------------"
    )


if __name__ == "__main__":
    run_planet()
