# File: /multiverse/system/inner/planet.py
# Last Update: 05/15/24
# Updated By: JW

import os
import pickle
import random
import logging
import time

import numpy as np
import pandas as pd
from omegaconf import ListConfig, OmegaConf

from ....core import Core
from ....utils import (
    function_scheduler,
    get_current_logging_config,
    configure_child_process_logging,
)
from .inner_utils import clean_data_filename
from .moon import Moon


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Planet(Core):
    """
    Perturb, Label And Navigate Existsing Tabulars
    ---

    Plan It. Planet!

    The Planet class lives in the --inner system-- and handles the transition
    from raw tabular data to scaled, encoded, and complete data. Specifically,
    this class is designed to handle datasets with missing values by filling
    missing values with randomly-sampled data, exploring the distribution of
    possible missing values.

    Parameters
    ----------
    data : pd.Dataframe, optional
        A pandas dataframe of raw data. Default is None.
    outDir : str, optional
        The directory path where the processed data will be saved. Default is None.
    scaler : str, optional
        The method used for scaling the data. Default is "standard".
    encoding : str or list, optional
        The method used for encoding categorical variables. Default is "one_hot" for all categorical variables.

        **Accepted Values**
        - "one_hot"
        - "integer"
        - "hash"

    dropColumns : list, optional
        A list of columns to be dropped from the data. Default is None.
    imputeMethods : list, str, optional
        A dictionary mapping column names to the imputation method to be used for each column. Default is None.

        NOTE: this parameter can take multiple types

        **Behavior**
        - imputeMethods: list
            - Will iterate overall imputation methods contained in list and create datasets that have been imputed based on the selected methods.
        - imputeMethods: "sampleNormal" -> str
            - Will use the sampleNormal method (or other).
        - imputeMethods: None
            - Will default to dropping columns with missing values, not imputing (as the imputeMethod is None).

        **Accepted Values**
        - "sampleNormal"
        - "drop"
        - "mean"
        - "median"
        - "mode"

    imputeColumns : list, optional str "all"
        A list of columns to be imputed. Default is None.

        NOTE: this parameter can take multiple types

        **Behavior**
        - imputeColumns: list
            - Will only impute the selection of data columns passed in the list.
        - imputeColumns: "all" -> str
            - Will impute all columns with missing values per the specified imputeMethods.
            - NOTE: no other string values accepted.
        - imputeColumns: None
            - Will drop all columns with missing values (ignores parameter(s) specified in imputeMethods when this is the case).

    numSamples : int, optional
        The number of samples to generate. Default is 1.
    seeds : list, optional
        A list of random seeds to use for reproducibility. Default is [42].
    verbose : bool, optional
        Whether to print progress messages. Default is False.
    YAML_PATH : str, optional
        The path to a YAML file containing configuration settings. Default is None.

    Attributes
    ----------
    data : pd.Dataframe
        A pandas dataframe of raw data.
    encoding : str or list
        The method used for encoding categorical variables.
    scaler : str
        The method used for scaling the data.
    dropColumns : list
        A list of columns dropped from the raw data.
    imputeColumns : list
        A list of impute columns.
    imputeMethods : list
        The methodology used to impute columns.
    numSamples : int
        The number of clean data frames produced when imputing.
    seeds : list
        A list of random seeds.
    outDir : str
        The path to the out data directory.
    YAML_PATH : str
        The path to the YAML parameter file.

    Methods
    -------
    get_data_path() -> str
        Returns the path to the raw data file.
    get_missingData_summary() -> dict
        Returns a dictionary summarizing missing data.
    get_recommended_sampling_method() -> list
        Returns the recommended sample method for a dataset.
    get_na_as_list() -> list
        Returns a list columns containing NaN values.
    getParams() -> dict
        Get a dictionary of parameters used in planet construction.
    writeParams_toYaml() -> None
        Saves your parameters to a YAML file.
    fit()
        Fits numSamples number of Moon objects and writes to outDir.
    save()
        Saves Planet to `.pkl` serialized object file.

    Example
    -------
    >>> data = pd.DataFrame({"A": ["Sally", "Freddy", "Johnny"],
                          "B": ["cat", "dog", None],
                          "C": [14, 22, None]})

    >>> data.to_pickle("myRawData")

    >>> data_path = "myRawData.pkl"
    >>> planet = Planet(
        data = data_path,
        outDir = "/<PATH TO OUT DIRECTORY>",
        scaler= "standard",
        encoding = "one_hot",
        dropColumns = None,
        imputeMethods = "sampleNormal",
        imputeColumns = "all",
        )

    >>> planet.fit()

    >>> planet.imputeData.to_pickle("myCleanData")
    """

    def __init__(
        self,
        data=None,
        outDir=None,
        scaler: str = "standard",
        encoding: str = "one_hot",
        dropColumns=None,
        imputeMethods=None,
        imputeColumns=None,
        numSamples: int = 1,
        seeds: list = [42],
        verbose: bool = False,
        YAML_PATH=None,
    ):
        """
        Construct a Planet instance

        Parameters
        ----------
        NOTE: all parameters can be provided via the YAML_PATH attr.

        data : str, optional
            Path to input data to be processed
        outDir : str, optional
            The directory path where the processed data will be saved. Default is None.
        scaler : str, optional
            The method used for scaling the data. Default is "standard".
        encoding : str or list
            The method used for encoding categorical variables. Default is "one_hot" for all categorical variables

            **Accepted Values**
            ```python
            "one_hot"
            "integer"
            "hash"
            ```

        dropColumns : list, optional
            A list of columns to be dropped from the data. Default is None.
        imputeMethods : list, str, optional
            A dictionary mapping column names to the imputation method to be
            used for each column. Default is None.

            NOTE: this parameter can take multiple types

            **Behavior**
            imputeMethods: list
            - will iterate overall imputation methods contained in list and
            create datasets that have been imputed based on the selected methods
            imputeMethods: "sampleNormal" -> str
            - will use the sampleNormal method (or other)
            imputeMethods: None
            - will default to dropping columns with missing values, not
            imputing (as the imputeMethod is None)

            **Accepted Values**
            ```python
            "sampleNormal"
            "drop"
            "mean"
            "median"
            "mode"
            ```

        imputeColumns : list, optional str "all"
            A list of columns to be imputed. Default is None.

            NOTE: this parameter can take multiple types

            **Behavior**
            imputeColumns: list
            - will only impute the selection of data columns passed in the list
            imputeColumns: "all" -> str
            - will impute all columns with missing values per the specified imputeMethods
            - NOTE: no other string values accepted
            imputeColumns: None
            - will drop all columns with missing values (ignores parameter(s) specified in imputeMethods when this is the case)

        numSamples : int, optional
            The number of samples to generate. Default is 1.
        seeds : list, optional
            A list of random seeds to use for reproducibility. Default is [42].
        verbose : bool
            Whether to print progress messages. Default is False.
        YAML_PATH : str, optional
            The path to a YAML file containing configuration settings. Default is None.
        """

        if YAML_PATH is None and data is None:
            raise ValueError(
                "Please provide config parameters or a path to a \
                yaml configuration file."
            )

        self.verbose = verbose
        self.YAML_PATH = None
        if YAML_PATH is not None:
            assert os.path.isfile(YAML_PATH), "params.yaml file not found!"

            self.YAML_PATH = YAML_PATH
            with open(YAML_PATH, "r") as f:
                params = OmegaConf.load(f)
            data = params.data
            scaler = params.Planet.scaler
            encoding = params.Planet.encoding
            dropColumns = params.Planet.dropColumns
            imputeColumns = params.Planet.imputeColumns
            imputeMethods = params.Planet.imputeMethods
            numSamples = params.Planet.numSamples
            seeds = params.Planet.seeds
            outDir = os.path.join(params.outDir, params.runName + "/clean")

        super().__init__(data_path=data, clean_path=None, projection_path=None)
        self.outDir = outDir

        if self.outDir is not None and not os.path.isdir(self.outDir):
            try:
                os.makedirs(str(outDir))
            except Exception as e:
                print(e)

        # HARD CODED SUPPORTED TYPED
        supported_imputeMethods = [
            "sampleNormal",
            "sampleCategorical",
            "drop",
            "mean",
            "median",
            "mode",
        ]

        self.scaler = scaler
        self.encoding = encoding

        self.numSamples = numSamples

        if seeds == "auto":
            seeds = [random.randint(0, 100) for _ in range(numSamples)]

        self.seeds = seeds
        assert numSamples > 0
        assert len(seeds) == numSamples

        assert self.scaler in ["standard"]

        if dropColumns is None or (
            type(dropColumns) == str and dropColumns.lower() == "none"
        ):
            self.dropColumns = []
        else:
            assert (
                type(dropColumns) == list or type(dropColumns) == ListConfig
            ), "dropColumns must be a list"
            self.dropColumns = dropColumns

        if imputeColumns is None or imputeColumns == "None":
            self.imputeColumns = []

        elif imputeColumns == "auto":

            self.imputeColumns = self.data.columns[
                self.data.isna().any()
            ].tolist()

        elif type(imputeColumns) == ListConfig or type(imputeColumns) == list:
            self.imputeColumns = imputeColumns
            for c in imputeColumns:
                if c not in self.data.columns:
                    print("Invalid impute column. Defaulting to 'None'")
                    self.imputeColumns = []
        else:
            self.imputeColumns = []

        if imputeMethods is None or imputeMethods == "None":
            self.imputeMethods = [
                "drop" for _ in range(len(self.imputeColumns))
            ]
        elif imputeMethods == "auto":
            self.imputeMethods = self.get_recomended_sampling_method()
        elif type(imputeMethods) == str:
            if not imputeMethods in supported_imputeMethods:
                print("Invalid impute methods. Defaulting to 'drop'")
                imputeMethods = "drop"
            self.imputeMethods = [
                imputeMethods for _ in range(len(self.imputeColumns))
            ]
        else:
            assert len(imputeMethods) == len(
                self.imputeColumns
            ), f"Lengh of imputeMethods: {len(imputeMethods)} must match length of imputeColumns: {len(self.imputeColumns)}"
            for index, method in enumerate(imputeMethods):
                if not method in supported_imputeMethods:
                    print("Invalid impute methods. Defaulting to 'drop'")
                    imputeMethods[index] = "drop"
            self.imputeMethods = imputeMethods

        # Check if any imputeMethods are randomized
        has_randomized_imputation = any(
            method in ["sampleNormal", "sampleCategorical"]
            for method in self.imputeMethods
        )
        if not has_randomized_imputation and self.numSamples > 1:
            print(
                "Warning: No randomized imputation methods specified. "
                "Overwriting to single sample."
            )
            self.numSamples = 1
            self.seeds = [42]

        # Log Planet configuration
        logger.debug(f"Planet initialized with data shape: {self.data.shape}")
        logger.debug(f"Number of samples to generate: {self.numSamples}")
        logger.debug(f"Impute columns: {self.imputeColumns}")
        logger.debug(f"Impute methods: {self.imputeMethods}")
        logger.debug(f"Drop columns: {self.dropColumns}")
        logger.debug(f"Encoding: {self.encoding}, Scaler: {self.scaler}")
        logger.debug(f"Output directory: {self.outDir}")

    def get_missingData_summary(self) -> dict:
        """
        Get a summary of missing data in the columns of the 'data' dataframe.

        Returns
        -------
        summary : dict
            A dictionary containing a breakdown of columns from 'data' that are:
            - 'numericMissing': Numeric columns with missing values
            - 'numericComplete': Numeric columns without missing values
            - 'categoricalMissing': Categorical columns with missing values
            - 'categoricalComplete': Categorical columns without missing values

        Examples
        --------
        >>> data = pd.DataFrame({"A": [1, 2, None],
                                "B": [3, None, 5],
                                "C": ["a", "b", None]})

        >>> planet = Planet(data=data)
        >>> summary = planet.get_missingData_summary()
        >>> print(summary)
        {'numericMissing': ['A', 'B'], 'numericComplete': [], 'categoricalMissing': ['C'], 'categoricalComplete': ['A', 'B']}
        """

        numeric_missing = []
        numeric_not_missing = []
        categorical_missing = []
        categorical_complete = []

        for column in self.data.columns:
            if self.data[column].dtype.kind in "biufc":
                if self.data[column].isna().any():
                    numeric_missing.append(column)
                else:
                    numeric_not_missing.append(column)
            else:
                if self.data[column].isna().any():
                    categorical_missing.append(column)
                else:
                    categorical_complete.append(column)

        summary = {
            "numericMissing": numeric_missing,
            "numericComplete": numeric_not_missing,
            "categoricalMissing": categorical_missing,
            "categoricalComplete": categorical_complete,
        }

        return summary

    def get_na_as_list(self) -> list:
        """
        Get a list of columns that contain NaN values.

        Returns
        -------
        list of str
            A list of column names that contain NaN values.

        Examples
        --------
        >>> data = pd.DataFrame({"A": [1, 2, None],
                    "B": [3, None, 5],
                    "C": ["a", "b", None]})

        >>> planet = Planet(data=data)
        >>> na_columns = planet.get_na_as_list()
        >>> print(na_columns)
        ['A', 'B', 'C']
        """
        return self.data.columns[self.data.isna().any()].tolist()

    def get_recomended_sampling_method(self) -> list:
        """
        Get a recommended sampling method for columns with missing values.

        Returns
        -------
        list
            A list of recommended sampling methods for columns with
            missing values.
            For numeric columns, "sampleNormal" is recommended.
            For non-numeric columns, "sampleCategorical"
            (most frequent value) is recommended.

        Examples
        --------
        >>> data = pd.DataFrame({"A": [1, 2, None],
                    "B": [3, None, 5],
                    "C": ["a", "b", None]})

        >>> planet = Planet(data=data)
        >>> methods = planet.get_recommended_sampling_method()
        >>> print(methods)
        ['sampleNormal', 'sampleCategorical', 'sampleCategorical']
        """
        methods = []
        for column in self.data.columns[self.data.isna().any()].tolist():
            if pd.api.types.is_numeric_dtype(self.data[column]):
                methods.append("sampleNormal")
            else:
                methods.append("sampleCategorical")

        return methods

    def fit(self):
        """
        The meat and potatoes -- configure and run your planet object based on the specified params.

        Uses the `ProcessPoolExecutor` library to spawn multiple processes and generate results in a time-efficient manner.

        Returns
        -------
        None
            Saves numSamples of files (cleaned, imputed, scaled etc. data) to the specified outDir.

        Examples
        --------
        >>> data = pd.DataFrame({"A": ["Sally", "Freddy", "Johnny"],
                  "B": ["cat", "dog", None],
                  "C": [14, 22, None]})

        >>> data.to_pickle("myRawData")

        >>> data_path = "myRawData.pkl"
        >>> planet = Planet(
            data = data_path,
            outDir = "<PATH TO OUT DIRECTORY>",
            scaler= "standard",
            encoding = "one_hot",
            dropColumns = None,
            imputeMethods = "sampleNormal",
            imputeColumns = "all",
            )

        >>> planet.fit()

        >>> planet.imputeData.to_pickle("myCleanData")
        """
        logger.info(
            f"Starting Planet.fit() – creating {self.numSamples} Moon object(s)"
        )
        logger.debug(f"Using seeds: {self.seeds}")
        logger.debug(f"Output directory: {self.outDir}")

        # Get current logging config to pass to child processes
        logging_config = get_current_logging_config()

        assert len(self.seeds) == self.numSamples
        subprocesses = []
        for i in range(self.numSamples):
            logger.debug(
                f"Preparing Moon {i+1}/{self.numSamples} with seed {self.seeds[i]}"
            )
            cmd = (self._instantiate_moon, i, logging_config)
            subprocesses.append(cmd)

        # Pre-count outputs for delta reporting
        pre_count = 0
        try:
            if self.outDir and os.path.isdir(self.outDir):
                pre_count = len(
                    [f for f in os.listdir(self.outDir) if f.endswith(".pkl")]
                )
        except Exception:
            pass

        workers = min(4, self.numSamples)
        logger.info(
            f"Launching {len(subprocesses)} Moon process(es) with max {workers} worker(s)…"
        )
        t0 = time.perf_counter()
        results = function_scheduler(
            subprocesses,
            max_workers=workers,
            out_message="SUCCESS: Imputation(s)",
            resilient=True,
            verbose=self.verbose,
        )
        t1 = time.perf_counter()

        # Post-count outputs for delta reporting
        created = None
        try:
            if self.outDir and os.path.isdir(self.outDir):
                post_count = len(
                    [f for f in os.listdir(self.outDir) if f.endswith(".pkl")]
                )
                created = max(0, post_count - pre_count)
        except Exception:
            created = None

        total = len(subprocesses)
        duration = t1 - t0
        if isinstance(results, list) and len(results) == total:
            logger.info(
                f"Planet.fit() complete in {duration:.2f}s – processed {total} Moon object(s){'' if created is None else f', created ~{created} file(s)'}"
            )
        else:
            logger.info(
                f"Planet.fit() complete in {duration:.2f}s – processed {total} Moon object(s)."
            )

    def _instantiate_moon(self, id, logging_config):
        """
        Helper function for the fit() method. See `fit()` for more details.

        This method creates and processes a single Moon instance with proper
        logging configuration for multiprocessing environments.

        Parameters
        ----------
        id : int
            Identifier for the Moon instance.
        logging_config : dict or None
            Logging configuration from parent process.

        Returns
        -------
        None

        Examples
        --------
        >>> planet = Planet()
        >>> planet._instantiate_moon(1, logging_config)
        """
        # Configure logging in this child process
        configure_child_process_logging(logging_config)

        if self.seeds is None:
            self.seeds = dict()
            self.seeds[id] = np.random.randint(0, 1000)

        my_moon = Moon(
            data=self.get_data_path(),
            dropColumns=self.dropColumns,
            encoding=self.encoding,
            scaler=self.scaler,
            imputeColumns=self.imputeColumns,
            imputeMethods=self.imputeMethods,
            seed=self.seeds[id],
            id=id,
        )
        my_moon.fit()

        filename_without_extension, extension = os.path.splitext(
            self.get_data_path()
        )
        data_name = filename_without_extension.split("/")[-1]
        file_name = clean_data_filename(
            data_name=data_name,
            id=id,
            scaler=self.scaler,
            encoding=self.encoding,
        )
        output_dir = str(self.outDir)
        output_filepath = os.path.join(output_dir, file_name)

        my_moon.save(file_path=output_filepath)

    def getParams(self) -> dict:
        """
        Get the parameters used to initialize the space of
        Moons around this Planet.

        Returns
        -------
        dict
            A dictionary containing the parameters used to
            initialize this specific Planet instance.

        Examples
        --------
        >>> planet = Planet()
        >>> params = planet.getParams()
        >>> print(params)
        {'data': None, 'scaler': 'standard', 'encoding': 'one_hot',
        'dropColumns': None, 'imputeColumns': None, 'imputeMethods': None,
        'numSamples': 1, 'seeds': [42], 'outDir': None}
        """

        return {
            "data": self.get_data_path(),
            "scaler": self.scaler,
            "encoding": self.encoding,
            "dropColumns": self.dropColumns,
            "imputeColumns": self.imputeColumns,
            "imputeMethods": self.imputeMethods,
            "numSamples": self.numSamples,
            "seeds": self.seeds,
            "outDir": self.outDir,
        }

    def writeParams_toYaml(self, YAML_PATH=None):
        """
        Write or create a YAML file with the Planet parameters.

        Parameters
        ----------
        YAML_PATH : str, optional
            Path to an existing or new YAML file.

        Returns
        -------
        None
        """
        if YAML_PATH is None and self.YAML_PATH is not None:
            YAML_PATH = self.YAML_PATH
        if YAML_PATH is None:
            raise ValueError("Please provide a valid filepath to YAML")

        # If the file exists, load it; otherwise start a new config
        if os.path.isfile(YAML_PATH):
            params = OmegaConf.load(YAML_PATH)
        else:
            params = OmegaConf.create()

        # Update with this object's parameters
        params.Planet = self.getParams()
        params.Planet.pop("outDir", None)
        params.Planet.pop("data", None)

        # Ensure directory exists
        os.makedirs(os.path.dirname(YAML_PATH), exist_ok=True)

        # Save the YAML file
        OmegaConf.save(params, YAML_PATH)
        print(
            f"YAML file successfully {'updated' if os.path.isfile(YAML_PATH) else 'created'} at {YAML_PATH}"
        )

    def save(self, file_path):
        """
        Save the current object instance to a file using pickle serialization.

        Parameters
        ----------
        file_path : str
            The path to the file where the object will be saved.

        Examples
        --------
        >>> planet = Planet()
        >>> planet.save("myPlanet.pkl")
        """
        try:
            with open(file_path, "wb") as f:
                pickle.dump(self, f)
        except Exception as e:
            print(e)
