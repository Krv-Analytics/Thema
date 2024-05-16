# File: scripts/setup.py
# Lasted Updated: 04-19-24
# Updated By: SW

import os
import sys
import yaml
import subprocess

import utils as ut


def create_conda_environment():
    """
    Function to create a conda environment from a environment.yml file
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
    Function to get the Conda environment name from a .yml file.
    """
    with open(environment_file, "r") as file:
        environment_data = yaml.safe_load(file)
        return environment_data.get("name")


def remove_conda_environment():
    """
    Function to remove a Conda environment.

    NOTE: you must deactivate / not be using the conda env to remove it
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

    NOTE: will run with or without args, currently no functionaliuty w/out args

    Example:
    ---------
        Run script to create conda envionment:
        ```
        $ python setup.py create-env
        ```

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
