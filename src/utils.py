import os
import sys
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

from dotenv import load_dotenv
from python_log_indenter import IndentedLoggerAdapter
from termcolor import colored
from tqdm import tqdm

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
    sys.path.append(src + "logging/")
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


def excecute_subprocess(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def function_scheduler(functions, max_workers=4, out_message="",resilient=False):
    """
    A function to help with scheduling functions to be executed in parallel.

    Parameters:
    -----------
    functions: list
        A list of tuples where each tuple contains a function and its arguments.
    max_workers: int
        The number of functions to be scheduled at one time (degree of parallelism).
    resilient: bool
        When set to False, will exit upon the first error thrown by a function. Set
        to True when expecting errors.
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(func, *args) for func, *args in functions]
        
        # Setting progress bar to track the number of completed functions
        progress_bar = tqdm(total=len(functions), desc="Progress", unit="function")
        
        outcomes = []
        for future in as_completed(futures):
            # Log the outcome of the function
            outcome = future.result()
            if isinstance(outcome, Exception):
                print(colored(str(outcome), "red"))
                if not resilient:
                    progress_bar.close()
                    raise outcome
            else:
                outcomes.append(outcome)
            # Update the progress bar for each completed function
            progress_bar.update(1)
    
    progress_bar.close()

    print("\n\n-------------------------------------------------------------------------------- \n\n")
    print(colored(f"{out_message} executed successfully.", "green"))
    print("\n\n-------------------------------------------------------------------------------- \n\n")



def log_error(message, resilient=False):
    if resilient:
        print(colored("Warning:" + message, "yellow"))

    else:
        print(colored("ERROR: " + message, "red"))
        exit()