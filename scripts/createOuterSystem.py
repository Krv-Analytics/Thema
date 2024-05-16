# File: scripts/createOuterSystem.py
# Last Update: 04-19-24
# Updated By: SW

import os
import sys
import utils as ut


path_to_root = ut.get_root()
sys.path.append(path_to_root)

from thema.multiverse import Oort


def run_oort():
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
