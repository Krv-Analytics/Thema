import numpy as np
import pandas as pd


def data_cleaner(data: pd.DataFrame, scaler=None, column_filter=[], encoding="integer"):
    """
    Clean and preprocess the input DataFrame according to the given parameters.
    We currently support:
        1) filtering columns
        2) scaling data
        3) encoding categorical variables
        4) removing NaN values

    Thi function returns a new, cleaned DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
    The input DataFrame to be cleaned and preprocessed.

    scaler : object, optional
    A scaler object to be used for scaling the data. If None, no scaling is applied.
    Default is None.

    column_filter : list, optional
    A list of column names to be dropped from the DataFrame. Default is an empty list.

    encoding : str, optional
    The encoding type to be used for categorical variables. Only "integer" and "one_hot"
    encodings are supported. Default is "integer".

    Returns
    -------
    pandas.DataFrame
    A cleaned and preprocessed DataFrame according to the given parameters.

    Raises
    ------
    AssertionError
    If encoding is not one of "integer" or "one_hot".
    """
    # Dropping columns
    cleaned_data = data.drop(columns=column_filter)

    # Encode
    assert encoding in [
        "integer",
        "one_hot",
    ], "Currently we only support `integer` and `one_hot` encodings"
    if encoding == "one_hot":
        # Use Pandas One Hot encoding
        cleaned_data = pd.get_dummies(cleaned_data, prefix="One_hot", prefix_sep="_")

    if encoding == "integer":
        encoder = integer_encoder

        categorical_variables = cleaned_data.select_dtypes(
            exclude=["number"]
        ).columns.tolist()
        # Rewrite Columns
        for column in categorical_variables:
            vals = cleaned_data[column].values
            cleaned_data[column] = encoder(vals)

    # Scale
    cleaned_data = pd.DataFrame(
        scaler.fit_transform(cleaned_data), columns=list(cleaned_data.columns)
    ).dropna()

    return cleaned_data


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


def clean_data_filename(scaler=None, encoding="integer", filter: bool = True):
    """
    Generate a filename for the cleaned and preprocessed data.

    Parameters
    ----------
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
        A filename for the cleaned and preprocessed data.

    """
    if scaler is None:
        scaler = ""
    else:
        scaler = "standard_scaled"

    if filter:
        filter = "filtered"
    else:
        filter = ""

    return f"clean_data_{scaler}_{encoding}-encdoding_{filter}.pkl"