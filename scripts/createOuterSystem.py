# File: scripts/createOuterSystem.py
# Last Update: 05/15/24
# Updated By: JW

import os
import sys

import utils as ut

path_to_root = ut.get_root()
sys.path.append(path_to_root)

from thema.multiverse import Oort


def run_oort():
    """
    Run the Oort function to create the Outer Solar System.

    This function creates and runs a simulation of the Outer Solar System
    using the parameters specified in `params.yaml`. It is called when using
    `invoke o`

    NOTE: Your `params.yaml` file must be in the root directory to
    work for invoke commands.

    Examples:
    >>> run_oort()
    -------------------------------------------------------------------------------------------------

     Successfully created the Outer Solar System!

    -------------------------------------------------------------------------------------------------
    """
    path_to_root = ut.get_root()
    path_to_yaml = os.path.join(path_to_root, "params.yaml")
    oort = Oort(YAML_PATH=path_to_yaml)
    oort.fit()

    print(
        "-------------------------------------------------------------------------------------------------"
    )
    print("\n\n Successfully created the Outer Solar System!  \n\n")
    print(
        "-------------------------------------------------------------------------------------------------"
    )


if __name__ == "__main__":
    run_oort()
