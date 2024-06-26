# File: thema/utils.py
# Last Update: 05/15/24
# Updated by: JW

import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm


def function_scheduler(
    functions, max_workers=4, out_message="", resilient=False, verbose=False
):
    """
    Schedule and execute functions in parallel.

    Parameters
    ----------
    functions : list
        A list of tuples where each tuple contains a function and its arguments.
    max_workers : int, optional
        The maximum number of functions to be scheduled at one time (degree of parallelism).
        Default is 4.
    out_message : str, optional
        A message to be printed after the execution of all functions. Default is an empty string.
    resilient : bool, optional
        When set to False, the execution will stop upon the first error thrown by a function.
        Set to True when expecting errors. Default is False.
    verbose : bool, optional
        When set to True, additional information will be printed after the execution of all functions.
        Default is False.

    Returns
    -------
    list
        A list of outcomes from the executed functions.

    Examples
    --------
    >>> def square(x):
    ...     return x ** 2
    >>> def cube(x):
    ...     return x ** 3
    >>> functions = [(square, 2), (cube, 3)]
    >>> outcomes = function_scheduler(functions, max_workers=2, out_message="Execution complete", resilient=True, verbose=True)
    >>> print(outcomes)
    [4, 27]
    """
    with warnings.catch_warnings(record=True) as outputs:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(func, *args) for func, *args in functions]

            # Setting progress bar to track the number of completed functions
            progress_bar = tqdm(
                total=len(functions),
                desc="Progress",
                unit="function",
                dynamic_ncols=True,
            )

            outcomes = []
            for future in as_completed(futures):
                # Log the outcome of the function
                outcome = future.result()
                if isinstance(outcome, Exception):
                    print(str(outcome), "red")
                    if not resilient:
                        progress_bar.close()
                        raise outcome
                else:
                    outcomes.append(outcome)
                # Update the progress bar for each completed function
                progress_bar.update(1)

        progress_bar.close()

    if verbose:
        print(
            "\n\n-------------------------------------------------------------------------------- \n\n"
        )
        print(f"{out_message} executed successfully.")
        print(
            "\n\n-------------------------------------------------------------------------------- \n\n"
        )
        print("\n\n Warnings: ")
        for warning in outputs:
            print(warning.message)
    return outcomes


def unpack_dataPath_types(data):
    """
    Unpack files and return the saved data as a dataframe.

    Parameters
    ----------
    data : str
        The path to the data file.

    Returns
    -------
    pandas.DataFrame
        The unpacked data as a pandas DataFrame.

    Raises
    ------
    ValueError
        If the file format is not supported or if there is an issue opening the data file.

    Examples
    --------
    >>> unpack_dataPath_types('/path/to/data.csv')
    # Returns the data from the CSV file as a pandas DataFrame

    >>> unpack_dataPath_types('/path/to/data.xlsx')
    # Returns the data from the Excel file as a pandas DataFrame

    >>> unpack_dataPath_types('/path/to/data.pkl')
    # Returns the data from the pickle file as a pandas DataFrame

    >>> unpack_dataPath_types('/path/to/data.txt')
    # Raises a ValueError since the file format is not supported

    >>> unpack_dataPath_types('/path/to/nonexistent_file.csv')
    # Raises a ValueError since the data file does not exist
    """
    try:
        assert os.path.isfile(data), "\n Invalid path to Data"
        if data.endswith(".csv"):
            return pd.read_csv(data)
        elif data.endswith(".xlsx"):
            return pd.read_excel(data)
        elif data.endswith(".pkl"):
            assert os.path.isfile(data), "\n Invalid path to Data"
            return pd.read_pickle(data)
        else:
            raise ValueError(
                "Unsupported file format. Supported data frame formats: .csv, \
                .xlsx, .pkl"
            )
    except:
        raise ValueError(
            "There was an issue opening your data file. \
            Please make sure it exists and is a supported file format."
        )


def create_file_name(className, classParameters, id=None):
    """
    Generate a filename for a class based on its name, parameters,
    and an optional ID.

    Parameters:
    -----------
    className : str
        The name of the class.
    classParameters : dict
        A dictionary containing the parameters of the class.
    id : int, optional
        An optional ID to append to the filename.

    Returns:
    -----------
    str
        The filename for the class.

    Examples:
    -----------
    >>> create_file_name("MyClass", {"param1": 10, "param2": "abc"})
    'MyClass_param110_param2abc.pkl'

    >>> create_file_name("AnotherClass", {"param1": 5, "param2": "xyz"}, id=123)
    'AnotherClass_param15_param2xyz_id123.pkl'
    """
    illegal_chars = ["[", "]", ",", "{", "}", ",", "'", "/", " ", ":"]
    filename = f"{className}_"
    for parameter, value in classParameters.items():
        filename += f"{parameter}{value}_"
    if id is not None:
        filename += f"id{id}"
    filename += ".pkl"
    filename = "".join(c for c in filename if c not in illegal_chars)
    return filename
