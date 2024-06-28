# File: multiverse/system/inner/moon.py
# Last Update: 05/15/24
# Updated By: JW

import pickle
import warnings

import category_encoders as ce
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ....core import Core
from . import inner_utils


class Moon(Core):
    """
    The Moon: Modify, Omit, Oscillate and Normalize.
    ------------------------------------------------

    The Moon data class resides cosmically near to the original raw dataset.
    This class handles a multitude of individual preprocessing steps helpful
    for smooth computation and analysis farther downstream the analysis pipeline.

    The intended use of this class is simplify the cleaning process and
    automate the production of an imputeData dataframe - a format of the
    data fit for more expansive exploration.

    The Moon class supports standard sklearn.preprocessing measures for
    scaling and encoding, with the primary additive feature being supported
    imputation methods for filling N/A values.

    Attributes
    ----------
    data : pd.DataFrame
        A pandas dataframe of raw data.
    imputeData : pd.DataFrame
        A pandas dataframe of complete, encoded, and scaled data.
    encoding : list
        A list of encoding methods used for categorical variables.
    scaler : str
        The scaling method used.
    dropColumns : list
        A list of columns dropped from the raw data.
    imputeColumns : list
        A list of columns with missing values.
    imputeMethods : list
        A list of imputation methods used to fill missing values.
    seeds : int
        The random seed used.
    outDir : str
        The path to the output data directory.

    Methods
    -------
    fit()
        Performs the cleaning procedure according to the constructor arguments.
    save(file_path)
        Saves the current object using pickle serialization.

    Examples
    --------
    >>> data = pd.DataFrame({"A": ["Sally", "Freddy", "Johnny"],
    ...                      "B": ["cat", "dog", None],
    ...                      "C": [14, 22, 43]})
    >>> data.to_pickle("myRawData")
    >>> data_path = "myRawData.pkl"
    >>> moon = Moon(data=data_path,
    ...             dropColumns=["A"],
    ...             encoding=["one_hot"],
    ...             scaler="standard",
    ...             imputeColumns=["B"],
    ...             imputeMethod=["mode"])
    >>> moon.fit()
    >>> moon.imputeData.to_pickle("myCleanData")
    """

    def __init__(
        self,
        data,
        dropColumns=[],
        encoding="one_hot",
        scaler="standard",
        imputeColumns=[],
        imputeMethods=[],
        id=None,
        seed=None,
    ):
        """
        Constructor for Moon class.

        Initializes a Moon object and sets cleaning parameters.

        Parameters
        ----------
        data : str or pd.DataFrame
            The path to the raw data file or a pandas dataframe of raw data.
        dropColumns : list, optional
            A list of column names that will be dropped from the clean data.
        encoding : list or str, optional
            The encoding method(s) used for categorical variables.
        scaler : str, optional
            The scaling method used.
        imputeColumns : list, optional
            A list of column names containing missing values.
        imputeMethods : list, optional
            A list of imputation methods used to fill missing values.
        id : None, optional
            The ID of the Moon object.
        seed : None, optional
            The random seed used.

        """
        super().__init__(data_path=data, clean_path=None, projection_path=None)

        self.dropColumns = dropColumns
        self.encoding = encoding
        self.scaler = scaler
        self.imputeColumns = imputeColumns
        self.imputeMethods = imputeMethods
        self.id = id
        self.seed = seed
        self.imputeData = None

    def fit(self):
        """
        Performs the cleaning procedure according to the constructor arguments.
        Initializes the imputeData member as a DataFrame, which is a scaled,
        numeric, and complete representation of the original raw data set.

        Examples
        ----------
        >>> moon = Moon()
        >>> moon.fit()
        """

        self.imputeData = inner_utils.add_imputed_flags(self.data, self.imputeColumns)
        for index, column in enumerate(self.imputeColumns):
            impute_function = getattr(inner_utils, self.imputeMethods[index])
            self.imputeData[column] = impute_function(self.data[column], self.seed)

        self.dropColumns = [col for col in self.dropColumns if col in self.data.columns]
        # Drop Columns
        if not self.dropColumns == []:
            self.imputeData = self.data.drop(columns=self.dropColumns)

        # Drop Rows with Nans
        self.imputeData.dropna(axis=0, inplace=True)

        if type(self.encoding) == str:
            self.encoding = [
                self.encoding
                for _ in range(
                    len(self.imputeData.select_dtypes(include=["object"]).columns)
                )
            ]

        # Encoding
        assert len(self.encoding) == len(
            self.imputeData.select_dtypes(include=["object"]).columns
        ), f"length of encoding: {len(self.encoding)}, length of cat variables: {len(self.imputeData.select_dtypes(include=['object']).columns)}"

        for i, column in enumerate(
            self.imputeData.select_dtypes(include=["object"]).columns
        ):
            encoding = self.encoding[i]

            if encoding == "one_hot":
                if self.imputeData[column].dtype == object:
                    self.imputeData = pd.get_dummies(
                        self.imputeData, prefix=f"OH_{column}", columns=[column]
                    )

            elif encoding == "integer":
                if self.imputeData[column].dtype == object:
                    vals = self.imputeData[column].values
                    self.imputeData[column] = inner_utils.integer_encoder(vals)

            elif encoding == "hash":
                if self.imputeData[column].dtype == object:
                    hashing_encoder = ce.HashingEncoder(cols=[column], n_components=10)
                    self.imputeData = hashing_encoder.fit_transform(self.imputeData)

            else:
                pass

        # Scaling
        assert self.scaler in ["standard"], "Invalid Scaler"
        if self.scaler == "standard":
            scaler = StandardScaler()
            self.imputeData = pd.DataFrame(
                scaler.fit_transform(self.imputeData),
                columns=list(self.imputeData.columns),
            )

    def save(self, file_path):
        """
        Saves the current object using pickle serialization.

        Parameters
        ----------
        file_path : str
            The file path for the object to be written to.

        Examples
        --------
        >>> moon = Moon()
        >>> moon.fit()
        >>> moon.save("myMoonObject.pkl")
        """
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
