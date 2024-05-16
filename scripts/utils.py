# File: scripts/utils.py
# Lasted Updated: 04-19-24
# Updated By: SW

import argparse
import subprocess


def get_root():
    """
    Function to return the absolute filepath to the root of your directory.

    This function finds the highest dir in your current location that is a git repository
    and returns the absolute path to this directory as the root. Used to deal with pathing.
    """

    try:

        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Error finding git root of dir: {e}")


def parse_args(_COMMANDS_):
    """
    Custom arg parser to support _COMMANDS_
    """
    parser = argparse.ArgumentParser(description="Parse arguments")
    parser.add_argument(
        "command",
        nargs=1,
        help=f"Please select a command from the following arguments: {set(_COMMANDS_.keys())}",
    )
    return parser.parse_args()
