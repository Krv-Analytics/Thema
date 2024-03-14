# File: src/utils.py 
# Last Update: 03-13-24
# Updated by: SW 

from tqdm import tqdm
from termcolor import colored
from concurrent.futures import ProcessPoolExecutor, as_completed



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