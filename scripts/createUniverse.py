# File: scripts/createUniverse.py
# Last Update: 05/15/24
# Updated By: JW

import os
import sys

import utils as ut
from omegaconf import OmegaConf

path_to_root = ut.get_root()
sys.path.append(path_to_root)

from thema.multiverse import Galaxy


def run_galaxy():
    path_to_root = ut.get_root()
    path_to_yaml = os.path.join(path_to_root, "params.yaml")
    try:
        with open(path_to_yaml, "r") as f:
            yamlParams = OmegaConf.load(f)
    except Exception as e:
        print(e)

    path_to_galaxy = os.path.join(
        yamlParams.outDir, yamlParams.runName + f"/{yamlParams.runName}_Galaxy.pkl"
    )

    galaxy = Galaxy(YAML_PATH=path_to_yaml)
    galaxy.fit()
    galaxy.collapse()
    galaxy.save(path_to_galaxy)

    print(
        "-------------------------------------------------------------------------------------------------"
    )
    print("\n\n Successfully created the Universe! (and unlocked its secrets).  \n\n")
    print(
        "-------------------------------------------------------------------------------------------------"
    )


if __name__ == "__main__":
    run_galaxy()
