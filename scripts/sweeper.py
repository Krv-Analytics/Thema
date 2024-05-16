# File: scripts/sweeper.py
# Last Update: 05/15/24
# Updated By: JW

import os
import shutil
import sys

import utils as ut
from omegaconf import OmegaConf


def get_path_to_data():
    """
    Returns
    -------
    str
        The path to the data directory specified in params.yaml.
    """

    path_to_root = ut.get_root()
    path_to_yaml = os.path.join(path_to_root, "params.yaml")
    with open(path_to_yaml, "r") as f:
        yamlParams = OmegaConf.load(f)
    return os.path.join(yamlParams.outDir, yamlParams.runName)


def sweep():
    """
    Remove the directory associated with the current run in `params.yaml`.

    This function deletes the directory specified by the path returned
    by the `get_path_to_data()` function.
    If the directory exists, it is recursively removed using
    `shutil.rmtree()`. If an error occurs during the removal process,
    an `OSError` is raised.

    Raises
    ------
    OSError
        If an error occurs during the removal process.
    """

    broom = get_path_to_data()
    if os.path.exists(broom):
        try:
            shutil.rmtree(broom)
        except OSError as e:
            print(f"Error: {broom} : {e.strerror}")


def sweep_inner():
    """
    Remove the `clean` directory associated with the
    current run in `params.yaml`.

    This function deletes the directory specified by the path returned
    by the `get_path_to_data()` function. If the directory exists, it is
    recursively removed using `shutil.rmtree()`. If an error occurs during
    the removal process, an `OSError` is raised.

    Raises
    ------
    OSError
        If an error occurs during the removal process.
    """

    broom = os.path.join(get_path_to_data(), "clean")
    print(f"rm -r {broom}")
    if os.path.exists(broom):
        try:
            shutil.rmtree(broom)
        except OSError as e:
            print(f"Error: {broom} : {e.strerror}")


def sweep_outer():
    """
    Remove the `projections` directory associated with the
    current run in `params.yaml`.

    This function deletes the directory specified by the path returned
    by the `get_path_to_data()` function. If the directory exists, it is
    recursively removed using `shutil.rmtree()`. If an error occurs during
    the removal process, an `OSError` is raised.

    Raises
    ------
    OSError
        If an error occurs during the removal process.
    """

    broom = os.path.join(get_path_to_data(), "projections")
    print(f"rm -r {broom}")
    if os.path.exists(broom):
        try:
            shutil.rmtree(broom)
        except OSError as e:
            print(f"Error: {broom} : {e.strerror}")


def sweep_universe():
    """
    Remove the `models` directory associated with the
    current run in `params.yaml`.

    This function deletes the directory specified by the path returned
    by the `get_path_to_data()` function. If the directory exists, it is
    recursively removed using `shutil.rmtree()`. If an error occurs during
    the removal process, an `OSError` is raised.

    Raises
    ------
    OSError
        If an error occurs during the removal process.
    """
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
