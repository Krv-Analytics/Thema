# File: multiverse/universe.py
# Lasted Updated: 05/15/24
# Updated By: JW

import glob
import importlib
import itertools
import os
import pickle

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS

from .utils import starFilters, starSelectors

from ... import config
from ...utils import create_file_name, function_scheduler
from . import geodesics


class Galaxy:
    """
    A space of stars.

    The largest space of data representations, a galaxy can be searched to
    find particular stars and systems most suitable for a particular explorer.

    Galaxy generates a space of star objects from the distribution of
    inner and outer systems.


    Members
    ------
    data: str
        Path to the original raw data file.
    cleanDir: str
        Path to a populated directory containing Moons.
    projDir: str
        Path to a populated directory containing Comets
    outDir: str
        Path to an out directory to store star objects.
    selection: dict
        Dictionary containing selected representative stars. Set by collapse function.
    YAML_PATH: str
        Path to yaml configuration file.

    Functions
    ---------
    get_data_path() -> str
        returns path to the raw data file
    fit() -> None
        fits a space of Stars and saves to outDir
    collapse() -> list
        clusters and selects representatives of star models
    get_galaxy_coordinates() -> np.ndarray
        computes a 2D coordinate system of stars in the galaxy using Multidimensional Scaling (MDS)
    save() -> None
        Saves instance to pickle file.

    Example
    --------
    >>> cleanDir = <PATH TO MOON OBJECT FILES>
    >>> data = <PATH TO RAW DATA FILE>
    >>> projDir = <PATH TO COMET OBJECT FILES>
    >>> outDir = <PATH TO OUT DIRECTORY OF PROJECTIONS>

    >>> params = {
    ...   "jmap": {   "nCubes":[2,5,8],
    ...                "percOverlap": [0.2, 0.4],
    ...            "minIntersection":[-1],
    ...            "clusterer": [["HDBSCAN", {"minDist":0.1}]]
    ...            }
    ... }
    >>> galaxy = Galaxy(params=params,
    ...            data=data,
    ...            cleanDir = cleanDir,
    ...            projDir = projDir,
    ...            outDir = outDir)

    >>> galaxy.fit()
    >>> # First, compute distances and cluster the stars
    >>> selected_stars = galaxy.collapse()
    >>> print(f"Selected {len(selected_stars)} representative stars")
    >>>
    >>> # Generate and visualize the galaxy coordinates with custom plotting
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>>
    >>> # Manual plotting of the galaxy coordinates (NOTE: `Thema` does not have built-in visualization dependencies)
    >>> coordinates = galaxy.get_galaxy_coordinates()
    >>> plt.figure(figsize=(8, 6))
    >>> plt.scatter(coordinates[:, 0], coordinates[:, 1], alpha=0.7)
    >>> plt.title('2D Coordinate Map of Star Models')
    >>> plt.xlabel('X Coordinate')
    >>> plt.ylabel('Y Coordinate')
    >>> plt.show()
    ```
    """

    def __init__(
        self,
        params=None,
        data=None,
        cleanDir=None,
        projDir=None,
        outDir=None,
        metric="stellar_curvature_distance",
        selector="max_nodes",
        nReps=3,
        filter_fn=None,
        YAML_PATH=None,
        verbose=False,
    ):
        """
        Constructs a Galaxy Instance

        Parameters
        ----------
        NOTE: all parameters can be provided via the YAML_PATH attr.
        Please see docs/yaml_configuration.md.

        data : str, optional
            Path to input data
        cleanDir: str, optional
            Path to directory containg saved Moon Objects
        projDir: str, optional
            Path to directort containing saved Comet Objects
        outDir : str, optional
            The directory path where the stars will be saved.
        params: dict, optional
            A parameter dictionary specifying stars and corresponding parameter lists
            **Behavior**
            {"star0_name" : {   "star0_parameter0":[list of star0_parameter0 values],
                                "star0_parameter1": [list of star0_parameter1 values]},
             "star1_name": {"star1_parameter0": [list of star1_parameter0 values]} }
        filter_fn: str, callable, or None, optional
            Filter function to apply to stars before distance calculations.
            Can be a string name of a function in starFilters, a callable, or None for no filtering.
        YAML_PATH : str, optional
            The path to a YAML file containing configuration settings. Default is None.
        verbose: bool
            Set to true to see warnings + print messages
        """
        if YAML_PATH is not None:
            assert os.path.isfile(YAML_PATH), "yaml parameter file could not be found."
            try:
                with open(YAML_PATH, "r") as f:
                    yamlParams = OmegaConf.load(f)
            except Exception as e:
                print(e)

            data = yamlParams.data
            cleanDir = os.path.join(yamlParams.outDir, yamlParams.runName + "/clean/")
            projDir = os.path.join(
                yamlParams.outDir, yamlParams.runName + "/projections/"
            )
            outDir = os.path.join(yamlParams.outDir, yamlParams.runName + "/models/")

            metric = yamlParams.Galaxy.metric
            selector = yamlParams.Galaxy.selector
            nReps = yamlParams.Galaxy.nReps
            filter_fn = yamlParams.Galaxy.filter

            if type(yamlParams.Galaxy.stars) == str:
                stars = [yamlParams.Galaxy.stars]
            else:
                stars = yamlParams.Galaxy.stars

            self.params = {}
            for star in stars:
                self.params[star] = yamlParams.Galaxy[star]

        elif params is not None:
            self.params = params

        else:
            raise ValueError("please provide a parameter dictionary")

        self.data = data
        self.cleanDir = cleanDir
        self.projDir = projDir
        self.outDir = outDir

        self.metric = metric
        self.selector = selector
        self.nReps = nReps
        self.filterFn = filter_fn

        self.keys = None
        self.distances = None
        self.verbose = verbose
        self.selection = {}

        assert self.data is not None, "Missing path to raw data file"
        assert self.cleanDir is not None, "Missing 'cleanDir' parameter'"
        assert self.projDir is not None, "Missing 'projDir' parameter"
        assert self.outDir is not None, "Missing 'outDir' parameter"

        assert os.path.isdir(self.cleanDir), "Invalid clean data directory."
        assert (
            len(os.listdir(self.cleanDir)) > 0
        ), "No clean data found. Please make sure you generated clean data."

        assert os.path.isdir(self.projDir), "Invalid projection directory."
        assert (
            len(os.listdir(self.projDir)) > 0
        ), "No projections found. Please make sure you have generated them correctly."

        if not os.path.isdir(self.outDir):
            try:
                os.makedirs(self.outDir)
            except Exception as e:
                print(e)

    def fit(self):
        """
        Configure and generate space of Stars.
        Uses the `function_scheduler` to spawn multiple star
        instances and fit them in parallel.

        Returns
        ------
        None
            Saves star objects to outDir and prints a count of failed saves.
        """

        subprocesses = []

        for starName, starParamsDict in self.params.items():
            star_configName = config.tag_to_class[starName]
            cfg = getattr(config, star_configName)
            module = importlib.import_module(cfg.module)
            star = module.initialize()

            # Load matching files
            cleanfile_pattern = os.path.join(self.cleanDir, "*.pkl")
            valid_cleanFiles = glob.glob(cleanfile_pattern)

            projfile_pattern = os.path.join(self.projDir, "*.pkl")
            valid_projFiles = glob.glob(projfile_pattern)

            for j, projFile in enumerate(valid_projFiles):
                projFilePath = os.path.join(self.projDir, projFile)
                with open(projFilePath, "rb") as f:
                    cleanFile = pickle.load(f).get_clean_path()

                param_attr_names = [
                    attr
                    for attr in sorted(cfg.__annotations__)
                    if attr not in ["name", "module"]
                ]
                param_combinations = itertools.product(
                    *[starParamsDict[attr] for attr in param_attr_names]
                )

                for k, combination in enumerate(param_combinations):
                    starParameters = dict(zip(param_attr_names, combination))

                    subprocesses.append(
                        (
                            self._instantiate_star,
                            self.data,
                            cleanFile,
                            projFilePath,
                            star,
                            starParameters,
                            starName,
                            f"{k}_{j}",
                        )
                    )

        # Run with function scheduler
        results = function_scheduler(
            subprocesses,
            max_workers=4,
            resilient=True,
            verbose=self.verbose,
        )

        failed_saves = sum(1 for r in results if r is False)
        print(
            f"\n⭐️ {len(results)-failed_saves}({(len(results)- failed_saves)/len(results)*100}%) star objects successfully saved."
        )

    def _instantiate_star(
        self,
        data_path,
        cleanFile,
        projFile,
        star,
        starParameters,
        starName,
        id,
    ):
        """Helper function for the fit() method. Creates a Star instances and fits it.

        Parameters
        ----------
        data_path: str
            Path to input data
        cleanFile: str
            Path to a moon instance.
        projFile: str
            Path to comet instance.
        star: class
            A class oject defined in stars/
        starParameters: dict
            Parameter configuration for specified star.
        starName: str
            Name of star class
        id : int
            Identifier

        Returns
        -------
        None

        See Also
        --------
        `Star` class and stars directory for more info on an individual fit.
        """
        my_star = star(
            data_path=data_path,
            clean_path=cleanFile,
            projection_path=projFile,
            **starParameters,
        )
        my_star.fit()
        output_file = create_file_name(starName, starParameters, id)
        output_file = os.path.join(self.outDir, output_file)
        return my_star.save(output_file)

    def collapse(
        self,
        metric=None,
        nReps=None,
        selector=None,
        filter_fn=None,
        **kwargs,
    ):
        """
        Collapses the space of Stars into a small number of representative Stars

        Parameters
        ----------
        metric: str
            metric used when comparing graphs. Currently, supported types are `stellar_curvature_distance`
            and `stellar_kernel_distance`.
        nReps: int
            The number of representative stars
        selector: str
            The selection criteria to choose representatives from a cluster. Currently, only "random" supported.
        **kwargs:
            Arguments necessary for different metric functions.

        Returns
        -------
        dict
            A dictionary containing the path to the star and the size of the group it represents.
        """

        if metric is None:
            metric = self.metric
        if nReps is None:
            nReps = self.nReps
        if selector is None:
            selector = self.selector

        # Handle the filter function - use parameter, fallback to instance default, then to no filter
        if filter_fn is None:
            filter_fn = self.filterFn or starFilters.nofilterfunction
        
        # Convert string filter names to actual functions
        if isinstance(filter_fn, str):
            filter_fn = getattr(starFilters, filter_fn, starFilters.nofilterfunction)
        elif callable(filter_fn):
            pass  # Use the callable directly
        elif filter_fn is not None:
            raise ValueError(
                f"filter_fn must be None, callable, or string, got {type(filter_fn)}"
            )

        metric_fn = getattr(geodesics, metric, geodesics.stellar_curvature_distance)
        selector_fn = getattr(starSelectors, selector, starSelectors.max_nodes)

        self.keys, self.distances = metric_fn(
            files=self.outDir, filterfunction=filter_fn, **kwargs
        )
        model = AgglomerativeClustering(
            metric="precomputed",
            linkage="average",
            compute_distances=True,
            distance_threshold=None,
            n_clusters=nReps,
        )
        model.fit(self.distances)

        labels = model.labels_
        subgroups = {}

        for label in labels:
            mask = np.where(labels == label, True, False)
            subkeys = self.keys[mask]
            subgroups[label] = subkeys

        for key in subgroups.keys():
            subgroup = subgroups[key]
            selected_star = selector_fn(subgroup)
            self.selection[key] = {
                "star": selected_star,
                "cluster_size": len(subgroup),
            }
        return self.selection

    def get_galaxy_coordinates(self) -> np.ndarray:
        """
        Computes a 2D coordinate system for stars in the galaxy, allowing visualization
        of their relative positions. This function uses Multidimensional Scaling (MDS)
        to project the high-dimensional distance matrix into a 2D space, preserving the
        relative distances between stars as much as possible.

        Note: This method requires that distances have been computed first, usually by calling the
        collapse() method or directly computing distances with a metric function.

        Returns
        -------
        np.ndarray
            A 2D array of shape (n_stars, 2) containing the X,Y coordinates of each star in the galaxy.
            Each row represents the 2D coordinates of one star.

        Examples
        --------
        >>> # After fitting the galaxy and computing distances
        >>> import matplotlib.pyplot as plt
        >>> coordinates = galaxy.get_galaxy_coordinates()
        >>>
        >>> # Basic scatter plot
        >>> plt.figure(figsize=(10, 8))
        >>> plt.scatter(coordinates[:, 0], coordinates[:, 1], alpha=0.7)
        >>> plt.title('Star Map of the Galaxy')
        >>> plt.xlabel('X Coordinate')
        >>> plt.ylabel('Y Coordinate')
        >>> plt.show()
        >>>
        >>> # Advanced plot with cluster coloring
        >>> if galaxy.selection:  # If collapse() has been called
        >>>     plt.figure(figsize=(12, 10))
        >>>     # Plot all stars
        >>>     plt.scatter(coordinates[:, 0], coordinates[:, 1], c='lightgray', alpha=0.5)
        >>>     # Highlight representative stars
        >>>     for cluster_id, info in galaxy.selection.items():
        >>>         # Find the index of the representative star in the keys array
        >>>         rep_idx = np.where(galaxy.keys == info['star'])[0][0]
        >>>         plt.scatter(coordinates[rep_idx, 0], coordinates[rep_idx, 1],
        >>>                   s=100, c='red', edgecolor='black', label=f'Cluster {cluster_id}')
        >>>     plt.legend()
        >>>     plt.title('Star Map with Representative Stars')
        >>>     plt.show()
        """
        if self.distances is None:
            raise ValueError("Distance matrix is not computed.")

        mds = MDS(n_components=2, dissimilarity="precomputed")
        coordinates = mds.fit_transform(self.distances)
        return coordinates

    def save(self, file_path):
        """
        Save the current object instance to a file using pickle serialization.

        Parameters
        ----------
        file_path : str
            The path to the file where the object will be saved.
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                pickle.dump(self, f)
            print(f"Saved object to {file_path}")
        except Exception as e:
            print(f"Failed to save object: {e}")

    def getParams(self):
        """
        Returns the parameters of the Galaxy instance.

        Returns
        -------
        dict
            A dictionary containing the parameters of the Galaxy instance.
        """
        params = {
            "params": self.params,
            "data": self.data,
            "cleanDir": self.cleanDir,
            "projDir": self.projDir,
            "outDir": self.outDir,
            "metric": self.metric,
            "selector": self.selector,
            "nReps": self.nReps,
            "YAML_PATH": self.YAML_PATH,
            "verbose": self.verbose,
        }
        return params

    def writeParams_toYaml(self, YAML_PATH=None):
        """
        Write the parameters of the Galaxy instance to a YAML file.

        Parameters
        ----------
        YAML_PATH: str, optional
            The path to the YAML file. If not provided, the YAML_PATH
            attribute of the instance will be used.

        Returns
        -------
        None
        """

        if YAML_PATH is None and self.YAML_PATH is not None:
            YAML_PATH = self.YAML_PATH

        if YAML_PATH is None and self.YAML_PATH is None:
            raise ValueError("Please provide a valid filepath to YAML")

        if not os.path.isfile(YAML_PATH):
            raise TypeError("File path does not point to a YAML file")

        with open(YAML_PATH, "r") as f:
            params = OmegaConf.load(f)

        params.Galaxy = self.getParams()["params"]
        params.Galaxy.stars = list(self.getParams()["params"].keys())

        with open(YAML_PATH, "w") as f:
            OmegaConf.save(params, f)

        print("YAML file successfully updated")

    def summarize_graphClustering(self):
        """
        Summarizes the graph clustering results.

        Returns
        -------
        dict
            A dictionary of the clusters and their corresponding graph members.
            The keys are the cluster names and the values are lists of graph
            file names.
        """
        pass
