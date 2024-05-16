# File: scripts/utils.py
# Lasted Updated: 05/15/24
# Updated By: JW

import argparse
import subprocess


def get_root():
    """
    Return the absolute filepath to the root of the directory.

    This function finds the highest directory in the current location
    that is a git repository and returns the absolute path to this
    directory as the root.

    Returns
    -------
    str
        The absolute filepath to the root directory.

    Raises
    ------
    ValueError
        If there is an error finding the git root of the directory.

    Examples
    --------
    >>> get_root()
    '/user/path/to/Thema'
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

    Parameters
    ----------
    _COMMANDS_ : dict
        A dictionary containing the available commands as keys.

    Returns
    -------
    argparse.Namespace
        An object containing the parsed command-line arguments.

    Raises
    ------
    argparse.ArgumentError
        If the provided command is not in the list of available commands.

    Notes
    -----
    This function uses the argparse module to parse command-line arguments.
    It expects a dictionary of available commands as input, where the keys
    represent the command names.

    Examples
    --------
    >>> commands = {'command1': 'Description of command 1', 'command2': 'Description of command 2'}
    >>> args = parse_args(commands)
    >>> print(args.command)
    ['command1']
    """

    parser = argparse.ArgumentParser(description="Parse arguments")
    parser.add_argument(
        "command",
        nargs=1,
        help=f"Please select a command from the following arguments: {set(_COMMANDS_.keys())}",
    )
    return parser.parse_args()
