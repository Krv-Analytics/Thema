# File: multiverse/system/outer/oort.py
# Last Update: 05/15/24
# Updated by: JW

import glob
import importlib
import itertools
import logging
import os
import pickle
import time

from omegaconf import OmegaConf

from .... import config
from ....core import Core
from ....utils import (
    create_file_name,
    function_scheduler,
    get_current_logging_config,
    configure_child_process_logging,
)

# Configure module logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Oort(Core):
    """
    The space of COMET objects.
    ----------------------------

    The Oort cloud, sometimes called the Öpik–Oort cloud,
    is theorized to be a vast cloud of icy planetesimals surrounding
    the Sun at distances ranging from 2,000 to 200,000 AU.

    Our Oort class generates a space of projected representations of
    an original, high dimensional dataset. Though sometimes it can be difficult
    to see through the cloud of projections, our tools allow you to easily
    navigate this terrain and properly explore your data.

    Parameters
    ----------
    data : pd.DataFrame
        A pandas DataFrame of raw data.
    params : dict, optional
        A parameter dictionary. Default is None.
    cleanDir : str, optional
        Path to the clean data directory. Default is None.
    outDir : str, optional
        Path to the out data directory. Default is None.
    YAML_PATH : str, optional
        Path to the YAML parameter file. Default is None.

    Attributes
    ----------
    data : pd.DataFrame
        A pandas DataFrame of raw data.
    params : dict
        A parameter dictionary.
    cleanDir : str
        Path to the clean data directory.
    outDir : str
        Path to the out data directory.
    YAML_PATH : str
        Path to the YAML parameter file.

    Methods
    -------
    get_data_path() -> str
        Returns the path to the raw data file.
    fit() -> None
        Fits projection space.
    save(file_path: str) -> None
        Saves object as a pickle file.
    getParams() -> dict
        Returns a dictionary of parameters.
    writeParams_toYaml(YAML_PATH: str) -> None
        Writes out the specified parameters to a YAML file.

    Examples
    --------
    >>> cleanDir = "<PATH TO MOON OBJECT FILES>"
    >>> data = "<PATH TO RAW DATA FILE>"
    >>> outDir = "<PATH TO OUT DIRECTORY OF PROJECTIONS>"
    >>> params = {
    ...     "tsne" : {
    ...         "perplexity" : [2, 5, 10],
    ...         "dimensions" : [2],
    ...         "seed" : [42]
    ...     }
    ... }
    >>> oort = Oort(
    ...     params=params,
    ...     data=data,
    ...     cleanDir=cleanDir,
    ...     outDir=outDir,
    ...     YAML_PATH=None
    ... )
    >>> oort.fit()

    Note
    ----
    `oort.fit()` will produce 6 * len(os.listdir(cleanDir)) files in
    `outDir` in this example.
    """

    def __init__(
        self,
        params=None,
        data=None,
        cleanDir=None,
        outDir=None,
        YAML_PATH=None,
        verbose=False,
    ):
        """
        Constructs an OORT instance.

        Parameters
        ----------
        data : str, optional
            Path to input data.
        cleanDir : str, optional
            Path to directory containing saved Moon Objects.
        outDir : str, optional
            The directory path where the projection data will be saved.
        params : dict, optional
            A parameter dictionary specifying projectors and corresponding parameter lists.
            **Behavior**
            {"projector0_name" : {"proj0_parameter0":[list of proj0_parameter0 values],
                      "proj0_parameter1": [list of proj0_parameter1 values]},
             "projector1_name": {"proj1_parameter0": [list of proj1_parameter0 values]} }
        YAML_PATH : str, optional
            The path to a YAML file containing configuration settings. Default is None.
        verbose : bool, optional
            Set to True to see warnings and print messages.

        Examples
        --------
        >>> cleanDir = "<PATH TO MOON OBJECT FILES>"
        >>> data = "<PATH TO RAW DATA FILE>"
        >>> outDir = "<PATH TO OUT DIRECTORY OF PROJECTIONS>"
        >>> params = {
        ...     "tsne" : {
        ...         "perplexity" : [2, 5, 10],
        ...         "dimensions" : [2],
        ...         "seed" : [42]
        ...     }
        ... }
        >>> oort = Oort(
        ...     params=params,
        ...     data=data,
        ...     cleanDir=cleanDir,
        ...     outDir=outDir,
        ...     YAML_PATH=None,
        ...     verbose=True
        ... )
        >>> oort.fit()
        """
        if YAML_PATH is not None:
            self.YAML_PATH = YAML_PATH
            assert os.path.isfile(YAML_PATH), "yaml parameter file could not be found."
            try:
                with open(YAML_PATH, "r") as f:
                    yamlParams = OmegaConf.load(f)
            except Exception as e:
                print(e)

            data = yamlParams.data
            cleanDir = os.path.join(yamlParams.outDir, yamlParams.runName + "/clean/")
            outDir = os.path.join(
                yamlParams.outDir, yamlParams.runName + "/projections/"
            )

            if type(yamlParams.Oort.projectiles) == str:
                projectors = [yamlParams.Oort.projectiles]
            else:
                projectors = yamlParams.Oort.projectiles

            self.params = {}
            for projector in projectors:
                self.params[projector] = yamlParams.Oort[projector]

        elif params is not None:
            self.YAML_PATH = None
            self.params = params

        else:
            raise ValueError(
                "please provide a parameter dictionary or a valid Yaml Parameter File"
            )

        super().__init__(data_path=data, clean_path=None, projection_path=None)
        self.cleanDir = cleanDir
        self.outDir = outDir
        self.verbose = verbose

        assert self.cleanDir is not None, "Missing 'cleanDir' parameter'"
        assert self.outDir is not None, "Missing 'outDir paramter' parameter"
        assert os.path.isdir(self.cleanDir), "Invalid clean data directory."
        assert (
            len(os.listdir(self.cleanDir)) > 0
        ), "No clean data found. Please make sure you generated clean data."

        for proj in self.params.keys():
            assert (
                proj in config.projector_tag_to_config.keys()
            ), f"{proj} is currently not a supported projectile."

        if not os.path.isdir(self.outDir):
            try:
                os.makedirs(self.outDir)
            except Exception as e:
                print(e)

        # Log Oort initialization
        clean_files = len(os.listdir(self.cleanDir))
        logger.info(
            f"Oort initialized with {len(self.params)} projector type(s) and {clean_files} clean data file(s)"
        )
        logger.debug(f"Clean directory: {self.cleanDir}")
        logger.debug(f"Output directory: {self.outDir}")
        for proj_name, proj_params in self.params.items():
            logger.debug(f"Projector '{proj_name}' parameters: {proj_params}")

    def fit(self):
        """
        Configure and run your projections.

        Uses the `ProcessPoolExecutor` library to spawn multiple projectile
        instances and fit them.

        Returns
        -------
        None
            Saves projections to the specified outDir

        Examples
        --------
        >>> oort = Oort()
        >>> oort.fit()
        """
        logger.info(
            f"Starting Oort.fit() – processing {len(self.params)} projector type(s)"
        )

        # Get current logging config to pass to child processes
        logging_config = get_current_logging_config()

        subprocesses = []
        for projectorName, projectorParamsDict in self.params.items():
            logger.debug(f"Setting up projector: {projectorName}")
            projConfig = config.projector_tag_to_config[projectorName]
            cfg = getattr(config, projConfig)
            module = importlib.import_module(cfg.module, package="thema")
            projector = module.initialize()

            file_pattern = os.path.join(self.cleanDir, "*.pkl")
            valid_files = glob.glob(file_pattern)
            logger.debug(
                f"Found {len(valid_files)} clean files for projector '{projectorName}'"
            )

            for i, cleanFile in enumerate(valid_files):
                parameter_combinations = itertools.product(
                    itertools.product(
                        *[
                            projectorParamsDict[attr]
                            for attr in sorted(cfg.__annotations__)
                            if attr not in ["name", "module"]
                        ]
                    )
                )
                cleanFile = os.path.join(self.cleanDir, cleanFile)
                param_count = 0
                for j, combination in enumerate(parameter_combinations):
                    projectorParameters = {
                        key: value
                        for key, value in zip(
                            sorted(projectorParamsDict.keys()), combination[0]
                        )
                    }
                    cmd = (
                        self._instantiate_projection,
                        self.get_data_path(),
                        cleanFile,
                        projector,
                        projectorParameters,
                        projectorName,
                        f"{j}_{i}",
                        logging_config,
                    )
                    subprocesses.append(cmd)
                    param_count += 1
                logger.debug(
                    f"Generated {param_count} parameter combinations for clean file {i+1}"
                )

        # Pre-count outputs for delta reporting
        pre_count = 0
        try:
            if self.outDir and os.path.isdir(self.outDir):
                pre_count = len(
                    [f for f in os.listdir(self.outDir) if f.endswith(".pkl")]
                )
        except Exception:
            pass

        # TODO: optimize max-Works based on OS availability
        total_projections = len(subprocesses)
        workers = 4
        logger.info(
            f"Launching {total_projections} projection process(es) with max {workers} worker(s)…"
        )

        t0 = time.perf_counter()
        results = function_scheduler(
            subprocesses,
            4,
            "SUCCESS: Projection(s)",
            resilient=True,
            verbose=self.verbose,
        )
        t1 = time.perf_counter()

        # Log completion stats
        if results:
            failed_count = sum(1 for r in results if r is False)
            success_count = total_projections - failed_count
            success_rate = (
                (success_count / total_projections * 100)
                if total_projections > 0
                else 0
            )
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

            logger.info(
                f"Oort.fit() complete in {t1 - t0:.2f}s – {success_count}/{total_projections} ({success_rate:.1f}%) successful{'' if created is None else f', created ~{created} file(s)'}"
            )
            if failed_count > 0:
                logger.warning(f"{failed_count} projections failed")
        else:
            logger.info(f"Oort.fit() complete in {t1 - t0:.2f}s")

    def _instantiate_projection(
        self,
        data,
        cleanFile,
        projector,
        projectorParameters,
        projectorName,
        id,
        logging_config,
    ):
        """
        Helper function for the fit() method. Creates a projectile instance
        and fits it.

        Parameters
        ----------
        data : str
            Path to the input data.
        cleanFile : str
            Path to the complete, scaled, encoded data file.
        projector : class
            Projector class to be configured and fitted.
        projectorParameters : dict
            Parameter configuration for the projector.
        projectorName : str
            Name of the projector class.
        id : int
            Identifier.
        logging_config : dict or None
            Logging configuration from parent process.

        Returns
        -------
        bool
            True if successful, False otherwise.

        See Also
        --------
        Comet : Class that provides more information on an individual fit.
        projectiles directory : Directory containing information on projectiles.

        Examples
        --------
        >>> data = "/path/to/data.csv"
        >>> cleanFile = "/path/to/clean_data.csv"
        >>> projector = MyProjector
        >>> projectorParameters = {"param1": 10, "param2": "abc"}
        >>> projectorName = "MyProjector"
        >>> id = 1
        >>> _instantiate_projection(data, cleanFile, projector, projectorParameters, projectorName, id, logging_config)
        """
        # Configure logging in this child process
        configure_child_process_logging(logging_config)

        try:
            my_projector = projector(
                data_path=data, clean_path=cleanFile, **projectorParameters
            )
            my_projector.fit()
            output_file = create_file_name(
                className=projectorName,
                classParameters=projectorParameters,
                id=id,
            )
            output_file = os.path.join(self.outDir, output_file)
            my_projector.save(output_file)
            return True
        except Exception as e:
            logger.error(f"Projection {projectorName} #{id} failed: {str(e)}")
            return False

    def getParams(self):
        """
        Get the parameters used to initialize the space of Comets in this Oort.

        Returns
        -------
        dict
            A dictionary containing the parameters used to initialize an Oort instance.

        Examples
        --------
        >>> oort = Oort()
        >>> params = oort.getParams()
        >>> print(params)
        {
            "params": {...},  # dictionary containing the parameters used to initialize the Oort instance
            "data": "/path/to/data",  # path to the data
            "cleanDir": True,  # whether to clean the directory
            "outDir": "/path/to/output"  # path to the output directory
        }
        """
        return {
            "params": self.params,
            "data": self.get_data_path(),
            "cleanDir": self.cleanDir,
            "outDir": self.outDir,
        }

    def writeParams_toYaml(self, YAML_PATH=None):
        """
        Write out the specified parameters to a YAML type file.

        Parameters
        ----------
        YAML_PATH : str (filepath), optional
            The path to an existing .yaml type file. If not provided, the value
            of `self.YAML_PATH` will be used. If `self.YAML_PATH` is also `None`,
            a `ValueError` will be raised.

        Returns
        ------
        None
            Saves a yaml file to the specified YAML_PATH.

        Raises
        ------
        ValueError
            If `YAML_PATH` is `None` and `self.YAML_PATH` is also `None`.
        TypeError
            If the file path specified by `YAML_PATH` does not point to a YAML file.

        Examples
        --------
        Example usage of `writeParams_toYaml`:

        >>> oort = Oort()
        >>> oort.writeParams_toYaml('/path/to/params.yaml')
        YAML file successfully updated
        """

        if YAML_PATH is None and self.YAML_PATH is not None:
            YAML_PATH = self.YAML_PATH

        if YAML_PATH is None and self.YAML_PATH is None:
            raise ValueError("Please provide a valid filepath to YAML")

        if not os.path.isfile(YAML_PATH):
            raise TypeError("File path does not point to a YAML file")

        with open(YAML_PATH, "r") as f:
            params = OmegaConf.load(f)

        params.Oort = self.getParams()["params"]
        params.Oort.projectiles = list(self.getParams()["params"].keys())

        with open(YAML_PATH, "w") as f:
            OmegaConf.save(params, f)

        print("YAML file successfully updated")

    def save(self, file_path):
        """
        Save the current object instance to a file using pickle serialization.

        Parameters
        ----------
        file_path : str
            The path to the file where the object will be saved.

        Raises
        ------
        IOError
            If there is an error while saving the object to the file.

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.save("data.pkl")  # Save the object to a file named "data.pkl"
        """

        try:
            with open(file_path, "wb") as f:
                pickle.dump(self, f)
        except IOError as e:
            print(e)
