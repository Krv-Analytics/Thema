import os
import sys

from dotenv import load_dotenv

################################################################################################
#  Handling Local Imports
################################################################################################


def env():
    """
    This function loads the .env file and adds necessary folders to the system path.

    Returns:
    -----------
    str
        The root directory of the project.
    """

    load_dotenv()
    root = os.getenv("root")
    src = os.getenv("src")
    sys.path.append(src + "jmapping/selecting/")
    sys.path.append(root + "logging/")
    sys.path.append(src + "modeling/synopsis/")
    return root, src


def get_imputed_files(directory_path, key):
    """
    Retrieve a list of file paths within the specified directory that
    contain a specified key in their filenames.

    Parameters:
    -----------
    directory_path <str>:
        - The path to the directory to search for files.
    key <str>:
        - The key to search for in the filenames of the files.

    Returns:
    -----------
    List[str]: A list of file paths that match the criteria.

    Raises:
    -----------
    AssertionError: If the specified directory_path does not exist.
    """
    root, _ = env()
    # Check if the directory path exists
    assert os.path.exists(
        directory_path
    ), f"Directory '{directory_path}' does not exist."
    files = []

    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        file = os.path.join(directory_path, filename)

        # Check if the file is a regular file and contains the specified key
        if os.path.isfile(file) and key in filename:
            files.append(os.path.join(root, file))

    return files
