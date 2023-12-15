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


def grab_imputed_data(directory_path, key):
    '''
    # TODO
    '''
    root, _ = env()
    # Check if the directory path exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return []

    # List to store file paths containing the specified key
    filtered_file_paths = []

    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # Check if the file is a regular file and contains the specified key
        if os.path.isfile(file_path) and key in filename:
            filtered_file_paths.append(os.path.join(root,file_path))

    return filtered_file_paths