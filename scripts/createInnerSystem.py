# File: scripts/createInnerSystem.py
# Last Update: 04-19-24
# Updated By: SW

import os
import sys
import utils as ut


path_to_root = ut.get_root()
sys.path.append(path_to_root)

from thema.multiverse import Planet


def run_planet():
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
