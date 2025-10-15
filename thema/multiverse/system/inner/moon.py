# File: multiverse/system/inner/moon.py
# Last Update: 10/15/25
# Updated By: SG

import pickle
import logging

import category_encoders as ce
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ....core import Core
from . import inner_utils

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Moon(Core):
    """
    The Moon: Modify, Omit, Oscillate and Normalize.
    ----------

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

        # Log initial state
        logger.debug(f"Moon initialized with data shape: {self.data.shape}")
        logger.debug(f"Drop columns: {self.dropColumns}")
        logger.debug(f"Impute columns: {self.imputeColumns}")
        logger.debug(f"Impute methods: {self.imputeMethods}")
        logger.debug(
            f"Encoding: {self.encoding}, Scaler: {self.scaler}, Seed: {self.seed}"
        )

    def fit(self):
        # Add imputed flags
        self.imputeData = inner_utils.add_imputed_flags(self.data, self.imputeColumns)
        logger.debug("Added imputed flags to columns")
        logger.debug(f"Data shape after adding flags: {self.imputeData.shape}")

        # Apply imputation
        for index, column in enumerate(self.imputeColumns):
            impute_function = getattr(inner_utils, self.imputeMethods[index])
            self.imputeData[column] = impute_function(self.data[column], self.seed)
            logger.debug(
                f"Column '{column}' imputed using '{self.imputeMethods[index]}'. "
                f"NaNs remaining: {self.imputeData[column].isna().sum()}"
            )

        # Drop specified columns
        self.dropColumns = [col for col in self.dropColumns if col in self.data.columns]
        if self.dropColumns:
            before_drop = self.imputeData.shape
            self.imputeData = self.imputeData.drop(columns=self.dropColumns)
            logger.debug(
                f"Dropped columns: {self.dropColumns}. Shape before: {before_drop}, after: {self.imputeData.shape}"
            )

        # Drop rows with NaNs
        nan_cols = self.imputeData.columns[self.imputeData.isna().any()]
        logger.debug(f"Columns with NaN values before dropping rows: {list(nan_cols)}")
        self.imputeData.dropna(axis=0, inplace=True)
        logger.debug(f"Shape after dropping rows with NaNs: {self.imputeData.shape}")

        # Ensure encoding is a list
        if isinstance(self.encoding, str):
            self.encoding = [
                self.encoding
                for _ in range(
                    len(self.imputeData.select_dtypes(include=["object"]).columns)
                )
            ]

        # Encoding
        cat_cols = self.imputeData.select_dtypes(include=["object"]).columns
        assert len(self.encoding) == len(cat_cols), (
            f"length of encoding: {len(self.encoding)}, "
            f"length of categorical variables: {len(cat_cols)}"
        )
        for i, column in enumerate(cat_cols):
            encoding_method = self.encoding[i]
            if encoding_method == "one_hot" and self.imputeData[column].dtype == object:
                self.imputeData = pd.get_dummies(
                    self.imputeData, prefix=f"OH_{column}", columns=[column]
                )
                logger.debug(f"Column '{column}' one-hot encoded")

            elif (
                encoding_method == "integer" and self.imputeData[column].dtype == object
            ):
                vals = self.imputeData[column].values
                self.imputeData[column] = inner_utils.integer_encoder(vals)
                logger.debug(f"Column '{column}' integer encoded")

            elif encoding_method == "hash" and self.imputeData[column].dtype == object:
                hashing_encoder = ce.HashingEncoder(cols=[column], n_components=10)
                self.imputeData = hashing_encoder.fit_transform(self.imputeData)
                logger.debug(f"Column '{column}' hash encoded")

        # Scaling
        assert self.scaler in ["standard"], "Invalid Scaler"
        if self.scaler == "standard":
            scaler = StandardScaler()
            self.imputeData = pd.DataFrame(
                scaler.fit_transform(self.imputeData),
                columns=list(self.imputeData.columns),
            )
            logger.debug(
                f"Data scaled using StandardScaler. Final shape: {self.imputeData.shape}"
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
        logger.debug(f"Moon object saved to {file_path}")
