import numpy as np
import pandas as pd


def data_cleaner(data: pd.DataFrame, scaler=None, column_filter=[], encoding="integer"):

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
    _, integer_encoding = np.unique(column_values, return_inverse=True)
    return integer_encoding


def clean_data_filename(scaler=None, encoding="integer", filter: bool = True):

    if scaler is None:
        scaler = ""
    else:
        scaler = "standard_scaled"

    if filter:
        filter = "filtered"
    else:
        filter = ""

    return f"clean_data_{scaler}_{encoding}-encdoding_{filter}.pkl"
