# File: scripts/sweeper.py
# Last Update: 04-19-24
# Updated By: SW

import os
import sys
import shutil
import utils as ut

from omegaconf import OmegaConf


def get_path_to_data():
    """Returns path outDir as specified in params.yaml"""
    path_to_root = ut.get_root()
    path_to_yaml = os.path.join(path_to_root, "params.yaml")
    with open(path_to_yaml, "r") as f:
        yamlParams = OmegaConf.load(f)
    return os.path.join(yamlParams.outDir, yamlParams.runName)
    re


def sweep():
    broom = get_path_to_data()
    if os.path.exists(broom):
        try:
            shutil.rmtree(broom)
        except OSError as e:
            print(f"Error: {broom} : {e.strerror}")


def sweep_inner():
    broom = os.path.join(get_path_to_data(), "clean")
    print(f"rm -r {broom}")
    if os.path.exists(broom):
        try:
            shutil.rmtree(broom)
        except OSError as e:
            print(f"Error: {broom} : {e.strerror}")


def sweep_outer():
    broom = os.path.join(get_path_to_data(), "projections")
    print(f"rm -r {broom}")
    if os.path.exists(broom):
        try:
            shutil.rmtree(broom)
        except OSError as e:
            print(f"Error: {broom} : {e.strerror}")


def sweep_universe():
    broom = os.path.join(get_path_to_data(), "models")
    try:
        shutil.rmtree(broom)
    except OSError as e:
        print(f"Error: {broom} : {e.strerror}")


_COMMANDS_ = {
    "inner": sweep_inner,
    "outer": sweep_outer,
    "universe": sweep_universe,
}
if __name__ == "__main__":
    if len(sys.argv) < 2:
        args = None
    else:
        args = ut.parse_args(_COMMANDS_)

    if args is None:
        sweep()
    else:
        command = args.command[0]
        if command in _COMMANDS_.keys():
            command_handler = _COMMANDS_[command]
            command_handler()
