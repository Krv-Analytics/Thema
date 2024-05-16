# File: thema/system/inner/inner_utils.py
# Last Update: 05/15/24
# Updated By: JW

import os

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

    Examples
    --------
    >>> column_values = np.array(['apple', 'banana', 'apple', 'orange', 'banana'])
    >>> integer_encoder(column_values)
    array([0, 1, 0, 2, 1])

    >>> column_values = np.array(['red', 'green', 'blue', 'red', 'green', 'blue'])
    >>> integer_encoder(column_values)
    array([0, 1, 2, 0, 1, 2])

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

    Examples
    --------
    >>> data_name = "raw_data"
    >>> scaler = None
    >>> encoding = "integer"
    >>> filter = True
    >>> generate_filename(data_name, scaler, encoding, filter)
    'raw_data_integer_imputed_filtered.pkl'

    >>> data_name = "raw_data"
    >>> scaler = "standard"
    >>> encoding = "one-hot"
    >>> filter = False
    >>> generate_filename(data_name, scaler, encoding, filter)
    'raw_data_standard_one-hot_imputed.pkl'
    """

    return f"{data_name}_{scaler}_{encoding}_imputed_{id}.pkl"


def sampleNormal(column, seed):
    """
    Fill in missing data in a column by sampling from the normal distribution.

    Parameters
    ----------
    column : pandas.Series
        The column containing the data to be filled.

    seed : int
        Seed value for the random number generator.

    Returns
    -------
    pandas.Series
        The column with missing values filled using random samples from the normal distribution.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> column = pd.Series([1, 2, np.nan, 4, np.nan, 6])
    >>> seed = 42
    >>> sampleNormal(column, seed)
    0    1.000000
    1    2.000000
    2    3.336112
    3    4.000000
    4    5.336112
    5    6.000000
    dtype: float64
    """
    np.random.seed(seed)

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


def sampleCategorical(column, seed):
    """
    Fill in missing data in a column by sampling from the
    categorical distribution.

    Parameters
    ----------
    column : pandas.Series
        The column containing the data to be filled.

    seed : int
        Seed value for the random number generator.

    Returns
    -------
    pandas.Series
        The column with missing values filled using random samples from the categorical distribution.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> column = pd.Series(['apple', 'banana', np.nan, 'orange', np.nan, 'banana'])
    >>> seed = 42
    >>> sampleCategorical(column, seed)
    0    apple
    1   banana
    2   banana
    3   orange
    4   banana
    5   banana
    dtype: object
    """
    np.random.seed(seed)

    column = column.copy()
    categories = column.dropna().unique()
    category_counts = column.dropna().value_counts()
    total_count = category_counts.sum()
    na_mask = column.isna()
    na_count = na_mask.sum()

    if na_count > 0:
        random_samples = np.random.choice(
            categories, size=na_count, p=category_counts / total_count
        )
        column.loc[na_mask] = random_samples

    return column


def mean(column, seed):
    """
    Fill in missing data in a column based on its average.

    Parameters
    ----------
    column : pandas.Series
        The column containing the data to be filled.
    seed : int
        Seed value for random number generation.

    Returns
    -------
    pandas.Series
        The column with missing values filled using the average.

    Examples
    --------
    >>> import pandas as pd
    >>> from numpy import nan
    >>> column = pd.Series([1, 2, nan, 4, 5])
    >>> mean(column, 42)
    0    1.0
    1    2.0
    2    3.0
    3    4.0
    4    5.0
    dtype: float64
    """
    column = column.copy()
    average = column.mean()
    column.fillna(average, inplace=True)
    return column


def mode(column, seed):
    """
    Fill missing values in a column with its mode (most frequent value).

    Parameters
    ----------
    column : pandas.Series
        The column containing the data to be filled.
    seed : int or None, optional
        Seed for the random number generator. Default is None.

    Returns
    -------
    pandas.Series
        The column with missing values filled with the mode.

    Examples
    --------
    >>> import pandas as pd
    >>> column = pd.Series([1, 2, 2, 3, 3, np.nan])
    >>> mode(column, 42)
    0    2.0
    1    2.0
    2    2.0
    3    3.0
    4    3.0
    5    2.0
    dtype: float64
    """
    column = column.copy()
    mode_value = column.mode().iloc[0]  # Get the first mode if multiple modes exist
    column.fillna(mode_value, inplace=True)
    return column


def median(column, seed):
    """
    Fill in missing data in a column based on its median value.

    Parameters
    ----------
    column : pandas.Series
        The column containing the data to be filled.
    seed : int
        Seed value for random number generation.

    Returns
    -------
    pandas.Series
        The column with missing values filled using the median.

    Examples
    --------
    >>> import pandas as pd
    >>> column = pd.Series([1, 2, None, 4, 5])
    >>> seed = 42
    >>> median(column, seed)
    0    1.0
    1    2.0
    2    3.0
    3    4.0
    4    5.0
    dtype: float64
    """
    column = column.copy()
    average = column.median()
    column.fillna(average, inplace=True)
    return column


def drop(column, seed):
    """
    Leave columns as is and let NaNs be dropped from column.

    Parameters
    ----------
    column : array_like
        The input column.
    seed : int
        The seed value for random number generation.

    Returns
    -------
    array_like
        The column with the element removed.

    Examples
    --------
    >>> drop([1, 2, 3, 4, 5], 42)
    [1, 2, 4, 5]
    >>> drop(['a', 'b', 'c', 'd'], 123)
    ['a', 'b', 'd']
    """
    return column


#  ╭──────────────────────────────────────────────────────────╮
#  │ Helper Functions                                         |
#  ╰──────────────────────────────────────────────────────────╯


def add_imputed_flags(df, impute_columns):
    """
    Add a flag for each value per column that is NA

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    impute_columns : list of str
        The list of column names to add imputed flags for.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with added imputed flags.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, None], 'B': [None, 4, 5]})
    >>> impute_columns = ['A', 'B']
    >>> add_imputed_flags(df, impute_columns)
       A    B  impute_A  impute_B
    0  1  NaN         0         1
    1  2  4.0         0         0
    2  NaN  5.0         1         0
    """
    impute_df = df.copy()

    for column in impute_columns:
        imputed_column_name = f"impute_{column}"
        impute_df[imputed_column_name] = impute_df[column].isna().astype(int)

    imputed_columns = impute_df.filter(like="impute_")

    return pd.concat([df, imputed_columns], axis=1)


def clear_previous_imputations(dir, key):
    """
    Clear all files in the directory that contain the specified key.

    Parameters
    ----------
    dir : str
        The directory path where the files are located.
    key : str
        The key to search for in the file names.

    Examples
    --------
    >>> clear_previous_imputations('/path/to/directory', 'imputation')
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
