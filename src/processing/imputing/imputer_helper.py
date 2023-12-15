"Helper functions for handling missing values"
import os
from types import FunctionType as function

import numpy as np
import pandas as pd

#  ╭──────────────────────────────────────────────────────────╮
#  │ Main Function                                            |
#  ╰──────────────────────────────────────────────────────────╯


def impute_data(df: pd.DataFrame, fillna_method: function):
    """
    Function to handle NaN values in a DataFrame according to specified `fill_method`.
    Currently supported methods include:
    *
    *

    Inputs
    ----------
    df: <pd.DataFrame>
        - dataframe that we are  (dealing with missing data)

    fill_method: <function>
        - a column-wise function to impute missing values.

    """
    X = df.copy()
    return X.apply(fillna_method, axis=0)


#  ╭──────────────────────────────────────────────────────────╮
#  │ Fill Methods (column-wise imputation functions)          |
#  ╰──────────────────────────────────────────────────────────╯


def random_sampling(column):
    """
    Function to fill in missing data based on sampling from the normal distribution of the column
    """
    mean = column.mean()
    std = column.std()
    na_mask = column.isna()
    na_count = na_mask.sum()

    if na_count > 0:
        # Generate random samples from a normal distribution
        random_samples = np.random.normal(loc=mean, scale=std, size=na_count)
        # Fill NAs with the generated random samples
        column.loc[na_mask] = random_samples

    return column


def average(column):
    """
    Function to fill in missing data in a column based on its average
    """
    average = column.mean()
    column.fillna(average, inplace=True)
    return column


def mode(column):
    """
    Function to fill in missing data in a column based on its mode (most frequent value)
    """
    mode_value = column.mode().iloc[0]  # Get the first mode if multiple modes exist
    column.fillna(mode_value, inplace=True)
    return column


def median(column):
    """
    Function to fill in missing data in a column based on its median value
    """
    average = column.median()
    column.fillna(average, inplace=True)
    return column


def drop(column):
    pass


# Methods that require a distribution of imputations
sampling_methods = ["random_sampling"]


#  ╭──────────────────────────────────────────────────────────╮
#  │ Helper Functions                                         |
#  ╰──────────────────────────────────────────────────────────╯


def imputed_filename(run_name, fill_method, number=None):
    """
    Generate a filename for the cleaned and preprocessed data.

    Parameters
    ----------
    fill_method : string, optional
        fill_method pulled from the params.yaml file

    number : int, optional
        The encoding type used for categorical variables. Default is "integer".

    Returns
    -------
    str
        A filename for the cleaned and preprocessed data.

    """
    if number is not None:
        number_out = "_" + str(number)
    else:
        number_out = ""

    return f"{run_name}_imputed_data_{fill_method}{number_out}.pkl"


def add_imputed_flags(df):
    """
    Add a flag for each value per column that is NA
    """
    imputationdf = df.copy()

    for column in imputationdf.columns[imputationdf.isna().any()]:
        imputed_column_name = f"imputed_{column}"
        imputationdf[imputed_column_name] = imputationdf[column].isna().astype(int)

    imputed_columns = imputationdf.filter(like="imputed_")

    return pd.concat([df, imputed_columns], axis=1)


def clear_current_imputations(dir, key):
    """
    Clear all files in the current directory that contain `key`.
    """
    files = os.listdir(dir)

    # Loop through each file in the directory
    for file_name in files:
        # Check if the file contains the specified key
        if key in file_name:
            file_path = os.path.join(dir, file_name)

            try:
                # Attempt to remove the file
                os.remove(file_path)
            except Exception as e:
                # Handle the exception if the file cannot be deleted
                print(f"Error deleting {file_path}: {e}")
