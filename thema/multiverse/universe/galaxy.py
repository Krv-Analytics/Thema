# File: multiverse/universe.py
# Lasted Updated: 10/21/25
# Updated By: SG

import glob
import importlib
import itertools
import logging
import os
import pickle
from collections import Counter
import time
from typing import cast

import numpy as np
import networkx as nx
from omegaconf import OmegaConf
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS

from .utils import starFilters, starSelectors

from ... import config
from ...utils import (
    create_file_name,
    function_scheduler,
    get_current_logging_config,
)
from . import geodesics

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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
            assert os.path.isfile(
                YAML_PATH
            ), "yaml parameter file could not be found."
            try:
                with open(YAML_PATH, "r") as f:
                    yamlParams = OmegaConf.load(f)
            except Exception as e:
                print(e)

            data = yamlParams.data
            cleanDir = os.path.join(
                yamlParams.outDir, yamlParams.runName + "/clean/"
            )
            projDir = os.path.join(
                yamlParams.outDir, yamlParams.runName + "/projections/"
            )
            outDir = os.path.join(
                yamlParams.outDir, yamlParams.runName + "/models/"
            )

            metric = yamlParams.Galaxy.metric
            selector = yamlParams.Galaxy.selector
            nReps = yamlParams.Galaxy.nReps
            filter_fn = yamlParams.Galaxy.get("filter", None)

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
        self.YAML_PATH = YAML_PATH

        self.metric = metric
        self.selector = selector
        self.nReps = nReps
        # Store YAML params for filter setup later (avoid pickling issues)
        self._yaml_filter = filter_fn
        self._yamlParams = yamlParams if YAML_PATH is not None else None

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

        self.data = cast(str, self.data)
        self.cleanDir = cast(str, self.cleanDir)
        self.projDir = cast(str, self.projDir)
        self.outDir = cast(str, self.outDir)

    def _setup_filter(self, yamlParams):
        logger.info("Checking yaml for filter configuration.")
        if yamlParams and yamlParams.Galaxy.get("filter"):
            filter_type = yamlParams.Galaxy.get("filter")
            if filter_type in config.filter_configs:

                filter_config = config.filter_configs[filter_type]
                logger.info(
                    f"Loading supported filter function: `{filter_type}`"
                )
                params = {
                    **filter_config["params"],
                    **yamlParams.Galaxy.get("filter_params", {}),
                }
                logger.info(f"Using filter parameters: {params}")
                func = getattr(starFilters, filter_config["function"])(**params)
                # Tag the callable with a human-friendly name for logging
                try:
                    setattr(func, "_display_name", str(filter_type))
                except Exception:
                    pass
                return func

        # Default to no-op filter with a stable display name
        nf = starFilters.nofilterfunction
        try:
            setattr(nf, "_display_name", "nofilterfunction")
        except Exception:
            pass
        return nf

    def _log_graph_distribution(self, files_to_use):

        out_dir = cast(str, self.outDir)
        file_paths = [
            os.path.join(out_dir, f)
            for f in os.listdir(out_dir)
            if f.endswith(".pkl")
        ]
        component_counts = []

        for file_path in file_paths:
            try:
                with open(file_path, "rb") as f:
                    star_obj = pickle.load(f)
                if star_obj.starGraph and star_obj.starGraph.graph:
                    component_counts.append(
                        nx.number_connected_components(star_obj.starGraph.graph)
                    )
            except:
                continue

        if component_counts:
            counts = Counter(component_counts)
            logger.debug("Component distribution:")
            for n, count in sorted(counts.items()):
                bar = "█" * count
                logger.debug(f"  {n:>2} components: {bar} ({count})")

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

        # Get current logging config to pass to child processes
        logging_config = get_current_logging_config()

        subprocesses = []

        for starName, starParamsDict in self.params.items():
            star_configName = config.tag_to_class[starName]
            cfg = getattr(config, star_configName)
            module = importlib.import_module(cfg.module)
            star = module.initialize()

            # Load matching files
            clean_dir = cast(str, self.cleanDir)
            cleanfile_pattern = os.path.join(clean_dir, "*.pkl")
            valid_cleanFiles = glob.glob(cleanfile_pattern)

            proj_dir = cast(str, self.projDir)
            projfile_pattern = os.path.join(proj_dir, "*.pkl")
            valid_projFiles = glob.glob(projfile_pattern)

            for j, projFile in enumerate(valid_projFiles):
                projFilePath = os.path.join(proj_dir, projFile)
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
                            logging_config,
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
        if failed_saves > 0:
            logger.warning(f"{failed_saves}/{len(results)} star saves failed")

    def _instantiate_star(
        self,
        data_path,
        cleanFile,
        projFile,
        star,
        starParameters,
        starName,
        id,
        logging_config,
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
        logging_config : dict or None
            Logging configuration from parent process

        Returns
        -------
        bool
            True if saved successfully, False otherwise

        See Also
        --------
        `Star` class and stars directory for more info on an individual fit.
        """
        # Configure logging in this child process
        from ...utils import configure_child_process_logging

        configure_child_process_logging(logging_config)

        try:
            my_star = star(
                data_path=data_path,
                clean_path=cleanFile,
                projection_path=projFile,
                **starParameters,
            )
            my_star.fit()
            output_file = create_file_name(starName, starParameters, id)
            out_dir = cast(str, self.outDir)
            output_file = os.path.join(out_dir, output_file)
            return my_star.save(output_file)
        except Exception as e:
            logger.error(
                f"Star {starName} #{id} failed - params: {starParameters}, error: {str(e)}"
            )
            return False

    def collapse(
        self,
        metric=None,
        nReps=None,
        selector=None,
        filter_fn=None,
        files: list | None = None,
        distance_threshold: float | None = None,
        **kwargs,
    ):
        """
        Collapses the space of Stars into representative Stars.
        Either nReps (number of clusters) or distance_threshold (AgglomerativeClustering) can be used.

        Parameters
        ----------
        metric : str, optional
            Metric function name for comparing graphs. Defaults to self.metric.
        nReps : int, optional
            Number of clusters for AgglomerativeClustering. Ignored if distance_threshold is set.
        selector : str, optional
            Selection function name to choose representative stars. Defaults to self.selector.
        filter_fn : callable, str, or None
            Filter function to select a subset of graphs. Defaults to no filter.
        files : list[str] or None
            Optional list of file paths to process. Defaults to self.outDir.
        distance_threshold : float, optional
            AgglomerativeClustering distance threshold. Used if nReps is None.
        **kwargs :
            Additional arguments passed to the metric function.

        Returns
        -------
        dict
            Mapping from cluster labels to selected stars and cluster sizes.
        """
        logger.info("Configuring Galaxy Collapse…")
        metric = metric or self.metric
        selector = selector or self.selector
        # Set up filter when needed

        if callable(filter_fn):
            logger.info(
                f"Using provided filter function: {getattr(filter_fn, '__name__', str(type(filter_fn)))}"
            )
        elif filter_fn is None:
            filter_fn = self._setup_filter(self._yamlParams)

        elif isinstance(filter_fn, str):
            logger.info(
                f"Function name provided, attempting to load from supported star filters: {filter_fn}"
            )
            filter_callable = getattr(
                starFilters, filter_fn, starFilters.nofilterfunction
            )
            # Tag display name for logging
            try:
                setattr(filter_callable, "_display_name", str(filter_fn))
            except Exception:
                pass
            filter_fn = filter_callable
            logger.info(
                f"Loaded filter function: {getattr(filter_fn, '__name__', str(type(filter_fn)))}"
            )
        else:
            filter_fn = starFilters.nofilterfunction
            try:
                setattr(filter_fn, "_display_name", "nofilterfunction")
            except Exception:
                pass
            logger.info(f"Defaulting to : {filter_fn.__name__}")

        if not callable(filter_fn):
            raise ValueError(
                f"filter_fn must be None, callable, or string, got {type(filter_fn)}"
            )

        metric_fn = getattr(
            geodesics, metric, geodesics.stellar_curvature_distance
        )
        selector_fn = getattr(starSelectors, selector, starSelectors.max_nodes)

        # Filter/metric/selector names for readability
        filter_fn_name = getattr(
            filter_fn,
            "_display_name",
            getattr(filter_fn, "__name__", str(type(filter_fn))),
        )
        logger.info(
            f"Filter: {filter_fn_name} | Metric: {metric} | Selector: {selector}"
        )

        # Determine files to process
        files_to_use = files if files is not None else self.outDir

        # Build a robust view of file list for logging (without changing behavior)
        file_list: list[str]
        out_dir = cast(str, self.outDir)
        if files is None:
            file_list = [
                os.path.join(out_dir, f)
                for f in os.listdir(out_dir)
                if f.endswith(".pkl")
            ]
        else:
            if isinstance(files, (list, tuple)):
                file_list = list(files)
            elif isinstance(files, str) and os.path.isdir(files):
                dir_str = cast(str, files)
                file_list = [
                    os.path.join(dir_str, f)
                    for f in os.listdir(dir_str)
                    if f.endswith(".pkl")
                ]
            else:
                # Fallback: treat as a single path
                file_list = [str(files)]

        total_files = len(file_list)
        target_desc = (
            f"directory '{self.outDir}'"
            if files is None
            else f"{total_files} provided file(s)"
        )
        logger.info(
            f"Scanning {total_files} candidate graph(s) from {target_desc}."
        )

        # Show graph distribution before filtering if DEBUG enabled
        if logger.isEnabledFor(logging.DEBUG):
            self._log_graph_distribution(files_to_use)

        # Determine concrete type to pass to metric function: either directory (str) or list[str]
        out_dir: str = cast(str, self.outDir)
        if files is None:
            metric_files: str | list[str] = out_dir
        else:
            if isinstance(files, (list, tuple)):
                metric_files = [str(f) for f in files]
            elif isinstance(files, str) and os.path.isdir(files):
                metric_files = files
            else:
                metric_files = [str(files)]

        # Compute distances with timing
        t0 = time.perf_counter()
        self.keys, self.distances = metric_fn(
            files=metric_files, filterfunction=filter_fn, **kwargs
        )
        t1 = time.perf_counter()
        filtered_count = len(self.keys)
        logger.info(
            f"Filter results: {filtered_count}/{total_files} graph(s) passed the filter in {t1 - t0:.2f}s"
        )

        # Distance matrix quick stats (off-diagonal)
        try:
            n = self.distances.shape[0]
            if n == self.distances.shape[1] and n == filtered_count and n > 1:
                mask = ~np.eye(n, dtype=bool)
                dvals = self.distances[mask]
                finite = np.isfinite(dvals)
                if not np.all(finite):
                    bad = np.size(dvals) - np.count_nonzero(finite)
                    logger.warning(
                        f"Distance matrix contains {bad} non-finite value(s) (NaN/inf)."
                    )
                if np.any(finite):
                    dvals_f = dvals[finite]
                    logger.debug(
                        "Distance stats (off-diagonal, finite): min=%.4f | mean=%.4f | max=%.4f | count=%d",
                        float(np.min(dvals_f)),
                        float(np.mean(dvals_f)),
                        float(np.max(dvals_f)),
                        int(dvals_f.size),
                    )
        except Exception:
            # Keep logging resilient
            pass

        # Check if we have enough graphs for clustering
        if filtered_count < 2:
            raise ValueError(
                f"Only {filtered_count} graph(s) passed the filter. "
                "Clustering requires at least 2 graphs. "
                "Consider relaxing your filter criteria."
            )

        # Use nReps or distance_threshold for AgglomerativeClustering
        # Handle clustering configuration clarity
        if nReps is None and distance_threshold is None:
            nReps = self.nReps

        if nReps is not None and distance_threshold is not None:
            logger.warning(
                "Both nReps and distance_threshold provided; using distance_threshold and ignoring nReps."
            )
            nReps = None

        # Check if nReps is valid for the number of filtered graphs
        if nReps and nReps > filtered_count:
            raise ValueError(
                f"Cannot create {nReps} clusters from {filtered_count} graphs. "
                f"Set nReps to {filtered_count} or fewer, or relax your filter."
            )

        model = AgglomerativeClustering(
            metric="precomputed",
            linkage="average",
            compute_distances=True,
            n_clusters=nReps,
            distance_threshold=distance_threshold,
        )
        mode_desc = (
            f"n_clusters={nReps}"
            if nReps is not None
            else f"distance_threshold={distance_threshold}"
        )
        logger.info(
            f"Clustering {filtered_count} graph(s) with AgglomerativeClustering ({mode_desc})…"
        )
        t2 = time.perf_counter()
        model.fit(self.distances)
        t3 = time.perf_counter()

        labels = model.labels_
        subgroups = {label: self.keys[labels == label] for label in set(labels)}

        # Log cluster size distribution
        cluster_sizes = {
            int(lbl): int(len(members)) for lbl, members in subgroups.items()
        }
        size_list = sorted(cluster_sizes.values(), reverse=True)
        logger.info(
            f"Formed {len(subgroups)} cluster(s) in {t3 - t2:.2f}s | sizes: {size_list}"
        )

        self.selection = {}
        for label, subgroup in subgroups.items():
            selected_star = selector_fn(subgroup)
            self.selection[label] = {
                "star": selected_star,
                "cluster_size": len(subgroup),
            }
            # Keep detailed selection at DEBUG to avoid log spam
            try:
                star_name = os.path.basename(str(selected_star))
            except Exception:
                star_name = str(selected_star)
            logger.debug(
                "Cluster %s: selected representative '%s' from %d member(s)",
                str(label),
                star_name,
                len(subgroup),
            )
        total_time = (t1 - t0) + (t3 - t2)
        logger.info(
            f"Galaxy Collapse complete: {len(self.selection)} representative model(s) selected "
            f"({metric}, {mode_desc}). Total compute time ~{total_time:.2f}s"
        )
        logger.info(
            "Access results: this Galaxy's 'selection' maps cluster -> {'star','cluster_size'}. "
            "If using a Thema instance, check its 'selected_model_files' for the chosen file paths."
        )
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

        # Resolve yaml path to a non-None string for type checking
        if YAML_PATH is None:
            if self.YAML_PATH is None:
                raise ValueError("Please provide a valid filepath to YAML")
            yaml_path = cast(str, self.YAML_PATH)
        else:
            yaml_path = str(YAML_PATH)

        if not os.path.isfile(yaml_path):
            raise TypeError("File path does not point to a YAML file")

        with open(yaml_path, "r") as f:
            params = OmegaConf.load(f)

        # Create Galaxy configuration with all required parameters
        galaxy_config = OmegaConf.create(
            {
                "metric": self.metric,
                "selector": self.selector,
                "nReps": self.nReps,
                "stars": list(self.params.keys()),
            }
        )

        # Add star-specific parameters
        for star_name, star_params in self.params.items():
            galaxy_config[star_name] = star_params

        # Add filter configuration if it exists in the original YAML
        if self._yamlParams and hasattr(self._yamlParams, "Galaxy"):
            if hasattr(self._yamlParams.Galaxy, "filter"):
                galaxy_config.filter = self._yamlParams.Galaxy.filter
            if hasattr(self._yamlParams.Galaxy, "filter_params"):
                galaxy_config.filter_params = (
                    self._yamlParams.Galaxy.filter_params
                )
            if hasattr(self._yamlParams.Galaxy, "cosmic_graph"):
                galaxy_config.cosmic_graph = (
                    self._yamlParams.Galaxy.cosmic_graph
                )

        params.Galaxy = galaxy_config

        with open(yaml_path, "w") as f:
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

    def _compute_cosmicGraphHelper(self, starFile, neighborhood):
        """Helper function for compute_cosmicGraph used for parallelization."""
        assert os.path.isfile(starFile), f"{starFile} is a directory?"
        with open(starFile, "rb") as f:
            star = pickle.load(f)
        return star.get_pseudoLaplacian(neighborhood=neighborhood)

    def compute_cosmicGraph(
        self,
        neighborhood="cc",
        threshold=0.0,
    ):
        """Computes the cosmicGraph of the galaxy.

        Parameters
        ---------
        neighborhood: str
            Options are specific to star. Please see docs for get_pseudoLaplacian for the star you are using. (e.g. jmapStar
            has neighborhood options of "cc" and "node" )

        threshold: float, default=0.0
            Percentage of agreement amongst contributing model to constitute an edge.

        """
        starFiles = []

        model_pattern = os.path.join(self.outDir, "*.pkl")
        starFiles = glob.glob(model_pattern)

        # Get the number of data points from one of the star files
        if len(starFiles) == 0:
            raise ValueError("No star files found in models directory")

        with open(starFiles[0], "rb") as f:
            sample_star = pickle.load(f)
        n = len(sample_star.clean)

        subprocesses = []
        for starFile in starFiles:
            cmd = (self._compute_cosmicGraphHelper, starFile, neighborhood)
            subprocesses.append(cmd)

        pseudo_laplacians = function_scheduler(
            subprocesses,
            4,
            "SUCCESS: Pseudo Laplacians ",
            resilient=True,
            verbose=self.verbose,
        )
        galactic_pseudoLaplacian = sum(pseudo_laplacians)
        cosmic_wadj = np.zeros((n, n), dtype=float)
        cosmic_adj = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                if i != j:
                    if (
                        galactic_pseudoLaplacian[i, i]
                        + galactic_pseudoLaplacian[j, j]
                        + galactic_pseudoLaplacian[i, j]
                    ) > 0:
                        cosmic_wadj[i, j] = -(
                            galactic_pseudoLaplacian[i, j]
                            / (
                                galactic_pseudoLaplacian[i, i]
                                + galactic_pseudoLaplacian[j, j]
                                + galactic_pseudoLaplacian[i, j]
                            )
                        )
                    if cosmic_wadj[i, j] > threshold:
                        cosmic_adj[i, j] = 1
        cosmicGraph = nx.from_numpy_array(cosmic_adj)
        for i, j in cosmicGraph.edges():
            cosmicGraph[i][j]["weight"] = cosmic_wadj[i][j]

        self.cosmicGraph = cosmicGraph

    # Ensure Galaxy instances are pickle-friendly for multiprocessing
    def __getstate__(self):
        state = self.__dict__.copy()
        for k, v in list(state.items()):
            if callable(v):
                state[k] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
