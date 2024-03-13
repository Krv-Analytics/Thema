import os
from types import FunctionType as function
import numpy as np
import pandas as pd


def integer_encoder(column_values: np.array):
    """
    Encode the given array of categorical values into integers.

    Parameters
    ----------
    column_values : numpy.ndarray
        An array of categorical values to be encoded.

    Returns
    -------
    numpy.ndarray
        An array of integers representing the encoded categorical values.

    """
    _, integer_encoding = np.unique(column_values, return_inverse=True)
    return integer_encoding


def clean_data_filename(data_name, scaler=None, encoding=None, id=None):
    """
    Generate a filename for the cleaned and preprocessed data.

    Parameters
    ----------
    data_name: str
        The name of the raw dataframe. 

    scaler : object, optional
        A scaler object used for scaling the data. If None, the filename will not contain
        a scaler label. Default is None.

    encoding : str, optional
        The encoding type used for categorical variables. Default is "integer".

    filter : bool, optional
        Whether or not columns were filtered during cleaning. If True, the filename
        will contain a filter label. Default is True.

    Returns
    -------
    str
        A filename for the clean data.

    """

    return f"{data_name}_{scaler}_{encoding}_imputed_{id}.pkl"





def random_sampling(column):
    """
    Function to fill in missing data based on sampling from the normal distribution of the column
    """
    column = column.copy()
    numeric_column = column.dropna()
    mean = numeric_column.mean()
    std = numeric_column.std()
    na_mask = column.isna()
    na_count = na_mask.sum()

    if na_count > 0:
        # Generate random samples from a normal distribution
        random_samples = np.random.normal(loc=mean, scale=std, size=na_count)
        # Fill NAs with the generated random samples
        column.loc[na_mask] = random_samples

    return column


def mean(column):
    """
    Function to fill in missing data in a column based on its average
    """
    column = column.copy()
    average = column.mean()
    column.fillna(average, inplace=True)
    return column


def mode(column):
    """
    Function to fill in missing data in a column based on its mode (most frequent value)
    """
    column = column.copy()
    mode_value = column.mode().iloc[0]  # Get the first mode if multiple modes exist
    column.fillna(mode_value, inplace=True)
    return column


def median(column):
    """
    Function to fill in missing data in a column based on its median value
    """
    column = column.copy()
    average = column.median()
    column.fillna(average, inplace=True)
    return column


def drop(column):
    """
    Leave column as is and remove element 
    """
    
    return column



#  ╭──────────────────────────────────────────────────────────╮
#  │ Helper Functions                                         |
#  ╰──────────────────────────────────────────────────────────╯


def add_imputed_flags(df, impute_columns):
    """
    Add a flag for each value per column that is NA
    """
    imputationdf = df.copy()

    for column in impute_columns:
        imputed_column_name = f"impute_{column}"
        imputationdf[imputed_column_name] = imputationdf[column].isna().astype(int)

    imputed_columns = imputationdf.filter(like="impute_")

    return pd.concat([df, imputed_columns], axis=1)


def clear_previous_imputations(dir, key):
    """
    Clear all files in directory that contain `key`.
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