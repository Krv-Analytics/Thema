
import numpy as np


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


def clean_data_filename(
    data_name, scaler=None, encoding=None, filter: bool = True
):
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

    return f"{data_name}_clean_data_{scaler}_{encoding}-encoding_{filter}.pkl"
