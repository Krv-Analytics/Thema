import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from dotenv import load_dotenv
from python_log_indenter import IndentedLoggerAdapter
from termcolor import colored
from tqdm import tqdm


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
