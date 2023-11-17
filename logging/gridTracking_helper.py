import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from dotenv import load_dotenv
from python_log_indenter import IndentedLoggerAdapter
from termcolor import colored
from tqdm import tqdm


def excecute_subprocess(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def subprocess_scheduler(
    subprocesses: list,
    num_processes: int,
    success_message: str = "SUCCESS",
    max_workers: int = 4,
    resilient=False,
):
    """
    A Function to help with scheduling processes during a grid search. This should facilitate error
    error messages, debugging and logging of errors.

    Parameters:
    -----------
    subproccess: list
        A list of subprocesses to be submitted for excecution
    num_processes: int
        The number of subprocesses being submitted
    success_message: str
        The message to be printed upon successful completion of a grid search
    max_workers: int
        The number of subprocesses to be scheduled at one time (dimension of parallelism)
    resilient: bool
        When set false, will exit grid search upon first error thrown by a subprocess. Set
        true when the grid search expects to see errors
    """
    # Running processes in Parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(excecute_subprocess, cmd) for cmd in subprocesses]
        # Setting Progress bar to track number of completed subprocesses
        progress_bar = tqdm(total=num_processes, desc="Progress", unit="subprocess")
        outcomes = []
        for future in as_completed(futures):
            # Log the outcome of the subprocess
            outcome = future.result()
            if not resilient:
                if not outcome.returncode == 0:
                    print(colored(outcome.stderr, "red"))
                    progress_bar.close()
                    sys.exit()
            else:
                outcomes.append(outcome.stderr)
            # Update the progress bar for each completed subprocess
            progress_bar.update(1)
    progress_bar.close()

    print(
        "\n\n-------------------------------------------------------------------------------- \n\n"
    )
    print(colored(success_message, "green"))
    print(
        "\n\n-------------------------------------------------------------------------------- \n\n"
    )


def log_error(message, resilient=False):
    if resilient:
        print(colored("Warning:" + message, "yellow"))

    else:
        print(colored("ERROR: " + message, "red"))
        exit()
