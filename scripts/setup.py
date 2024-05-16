# File: scripts/setup.py
# Lasted Updated: 05/15/24
# Updated By: JW

import os
import subprocess
import sys

import utils as ut
import yaml


def create_conda_environment():
    """
    Create a conda environment from an environment.yml file.

    This function creates a conda environment by executing the
    `conda env create` command with the specified environment.yml file.
    It also creates a .pth file in the site-packages directory of the
    environment, which adds the root directory of the git repository to the
    Python module search path.

    Notes
    -------
    This function assumes that the `environment.yml` file is located
    in the root directory of the git repository. The Python version used in
    the .pth file path (`python3.11`) is just an example. You may
    need to modify it based on your specific Python version.

    Raises
    -------
    subprocess.CalledProcessError
        If an error occurs while creating the conda environment.

    Examples
    ---------
    To create a conda environment from the environment.yml file:
    >>> create_conda_environment()
    Conda environment created successfully.


    """
    environment_file = f"{ut.get_root()}/environment.yml"

    try:
        print(f"Creating Conda Env from: `{environment_file}`")
        subprocess.run(["conda", "env", "create", "-f", environment_file], check=True)
        git_root = ut.get_root()

        # Create a .pth file with the git root directory, so you don't need a sys.path.append when running in the env
        pth_file_path = os.path.join(
            os.getenv("CONDA_PREFIX"),
            "lib",
            "python3.11",
            "site-packages",
            "git_root.pth",
        )
        site_packages_dir = os.path.dirname(pth_file_path)
        if not os.path.exists(site_packages_dir):
            os.makedirs(site_packages_dir)
        with open(pth_file_path, "w") as f:
            f.write(git_root)

        print("Conda environment created successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error creating conda environment: {e}")


def _get_environment_name_from_file(environment_file):
    """
    Get the environment name from a .yml file.

    Parameters
    ----------
    environment_file : str
        The path to the .yml file containing the Conda environment.

    Returns
    -------
    str or None
        The name of the Conda environment if found in the file, None otherwise.
    """
    with open(environment_file, "r") as file:
        environment_data = yaml.safe_load(file)
        return environment_data.get("name")


def remove_conda_environment():
    """
    Remove a Conda environment.

    This function removes a Conda environment. Before removing the environment,
    make sure to deactivate it or not be using it.

    Notes
    -------
    The environment name must be specified in the environment.yml file.
    If the environment name is not found in the file, an error message will be
    printed.

    Raises
    -------
    subprocess.CalledProcessError
        If an error occurs while removing the Conda environment.
    """
    try:
        environment_file = f"{ut.get_root()}/environment.yml"

        env_name = _get_environment_name_from_file(environment_file)
        if env_name is None:
            print("Error: Environment name not found in the environment file.")
            return

        print(f"Removing Conda Env: {env_name}")
        subprocess.run(["conda", "env", "remove", "-n", env_name, "--yes"], check=True)
        print(f"Conda environment '{env_name}' removed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error removing Conda environment: {e}")


#  ╭─────────────────────────────────────╮
#  │ args, to add when running the main  |
#  ╰─────────────────────────────────────╯

_COMMANDS_ = {
    "create-env": create_conda_environment,
    "remove-env": remove_conda_environment,
}


def main(args):
    """
    Driver function to deal with environment creation

    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments parsed by argparse.

    Notes
    -----
    This function is the entry point for dealing with environment creation.
    It can be run with or without arguments, but currently there is no functionality without arguments.

    Examples
    --------
    Run the script to create a conda environment:

    >>> python setup.py create-env

    """
    if not args.command:
        pass
    else:
        command = args.command[0]
        if command in _COMMANDS_.keys():
            command_handler = _COMMANDS_[command]
            command_handler()
        else:
            print("Unknown Command")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            """ Setup.py requires a command line argument. 
             
            Options: 
            --------
            create-env: creates a conda virtual environment from environment.yml 

            remove-env: removes the conda virtual environment
             """
        )
    else:
        # handle main when run with args
        args = ut.parse_args(_COMMANDS_)
        main(args)
