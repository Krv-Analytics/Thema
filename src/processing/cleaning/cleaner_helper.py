import category_encoders as ce
import numpy as np
import pandas as pd
from termcolor import colored


def data_cleaner(data: pd.DataFrame, scaler=None, column_filter=[], encoding="integer"):
    """
    Clean and preprocess the input DataFrame according to the given parameters.
    We currently support:
        1) filtering columns
        2) scaling data
        3) encoding categorical variables

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
    If encoding is not one of "integer", "one_hot", or "hash".
    """
    # Dropping columns
    try:
        cleaned_data = data.drop(columns=column_filter)
    except:
        print(
            colored(
                " \n WARNING: Invalid Dropped Columns in Parameter file: Defaulting to no dropped columns.",
                "yellow",
            )
        )
        cleaned_data = data

    # Encode
    assert encoding in ["integer", "one_hot", "hash"], colored(
        "\n ERROR: Invalid Encoding. Currently we only support `integer`,`one_hot` and `hash` encodings",
        "red",
    )
    if encoding == "one_hot":
        # Use Pandas One Hot encoding

        # cleaned_data = pd.get_dummies(cleaned_data, prefix="One_hot", prefix_sep="_")

        non_numeric_columns = cleaned_data.select_dtypes(exclude=["number"]).columns
        for column in non_numeric_columns:
            cleaned_data = pd.get_dummies(
                cleaned_data, prefix=f"OH_{column}", columns=[column]
            )

    if encoding == "integer":
        encoder = integer_encoder

        categorical_variables = cleaned_data.select_dtypes(
            exclude=["number"]
        ).columns.tolist()
        # Rewrite Columns
        for column in categorical_variables:
            vals = cleaned_data[column].values
            cleaned_data[column] = encoder(vals)

    if encoding == "hash":
        categorical_variables = cleaned_data.select_dtypes(
            exclude=["number"]
        ).columns.tolist()
        hashing_encoder = ce.HashingEncoder(cols=categorical_variables, n_components=10)
        cleaned_data = hashing_encoder.fit_transform(cleaned_data)

    # Scale
    if scaler is not None:
        cleaned_data = pd.DataFrame(
            scaler.fit_transform(cleaned_data), columns=list(cleaned_data.columns)
        )
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


def clean_data_filename(
    run_name="My_Sim", scaler=None, encoding="integer", filter: bool = True
):
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
    if scaler == "None":
        scaler = ""
    else:
        scaler = "standard_scaled"

    if filter:
        filter = "filtered"
    else:
        filter = ""

    return f"{run_name}_clean_data_{scaler}_{encoding}-encoding_{filter}.pkl"
