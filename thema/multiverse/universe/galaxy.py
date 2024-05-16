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
import plotly.express as px
import plotly.graph_objects as go
import yaml
from omegaconf import OmegaConf
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS

from ... import config
from ...utils import create_file_name, function_scheduler
from . import geodesics, starSelectors


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
    show_MDS() -> None
        plots a 2D representation of model layout
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
    >>> galaxy.show_MDS()
    >>> galaxy.collapse()
    ```
    """

    def __init__(
        self,
        params=None,
        data=None,
        cleanDir=None,
        projDir=None,
        outDir=None,
        metric="stellar_kernel_distance",
        selector="random",
        nReps=3,
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

        self.keys = None
        self.distances = None
        self.verbose = verbose

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
        Configure and generate space of Stars
        Uses the `ProcessPoolExecutor` library to spawn multiple star
        instances and fit them.

        Returns
        ------
        None
            Saves star objects to outDir
        """

        subprocesses = []
        for starName, starParamsDict in self.params.items():
            star_configName = config.tag_to_class[starName]
            cfg = getattr(config, star_configName)
            module = importlib.import_module(cfg.module)
            star = module.initialize()
            cleanfile_pattern = os.path.join(self.cleanDir, "*.pkl")
            valid_cleanFiles = glob.glob(cleanfile_pattern)
            projfile_pattern = os.path.join(self.projDir, "*.pkl")
            valid_projFiles = glob.glob(projfile_pattern)
            for j, projFile in enumerate(valid_projFiles):
                projFile = os.path.join(self.projDir, projFile)
                with open(projFile, "rb") as f:
                    cleanFile = pickle.load(f).get_clean_path()
                    parameter_combinations = itertools.product(
                        itertools.product(
                            *[
                                starParamsDict[attr]
                                for attr in sorted(cfg.__annotations__)
                                if attr not in ["name", "module"]
                            ]
                        )
                    )
                    for k, combination in enumerate(parameter_combinations):
                        starParameters = {
                            key: value
                            for key, value in zip(
                                sorted(starParamsDict.keys()), combination[0]
                            )
                        }
                        cmd = (
                            self._instantiate_star,
                            self.data,
                            cleanFile,
                            projFile,
                            star,
                            starParameters,
                            starName,
                            f"{k}_{j}",
                        )
                        subprocesses.append(cmd)

        # TODO: Optimize max workers
        function_scheduler(
            subprocesses,
            4,
            "SUCCESS: Graph Generation(s)",
            resilient=True,
            verbose=self.verbose,
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
        my_star.save(output_file)

    def collapse(self, metric=None, nReps=None, selector=None, **kwargs):
        """
        Collapses the space of Stars into a small number of representative Stars

        Parameters
        ----------
        metric : str, optional
            The metric used when comparing graphs. Currently, we only support
            `stellar_kernel_distance`. (default: None)
        nReps : int, optional
            The number of representative stars. (default: None)
        selector : str, optional
            The selection criteria to choose representatives from a cluster.
            Currently, only "random" is supported. (default: None)
        **kwargs : dict
            Additional arguments necessary for different metric functions.

        Returns
        -------
        dict
            A dictionary containing the path to the star and the size of the
            group it represents.

        Examples
        --------
        >>> galaxy = Galaxy()
        >>> galaxy.collapse(metric='stellar_kernel_distance', nReps=5, selector='random')
        {'0': {'star': 'path/to/star1', 'cluster_size': 10},
            '1': {'star': 'path/to/star2', 'cluster_size': 8},
            '2': {'star': 'path/to/star3', 'cluster_size': 12},
            '3': {'star': 'path/to/star4', 'cluster_size': 9},
            '4': {'star': 'path/to/star5', 'cluster_size': 11}}
        """

        if metric is None:
            metric = self.metric
        if nReps is None:
            nReps = self.nReps
        if selector is None:
            selector = self.selector
        metric = getattr(geodesics, metric)
        selector = getattr(starSelectors, selector)
        self.keys, self.distances = metric(
            files=self.outDir, filterfunction=None, **kwargs
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
        self.clusters = {}

        for label in labels:
            mask = np.where(labels == label, True, False)
            subkeys = self.keys[mask]
            self.clusters[label] = subkeys

        self.selection = {}
        for key in self.clusters.keys():
            subgroup = self.clusters[key]
            selected_star = selector(subgroup)
            self.selection[key] = {
                "star": selected_star,
                "cluster_size": len(subgroup),
            }
        return self.selection

    def show_mds(self, randomState: int = None):
        """
        Generates an embedding based on precomputed metric.

        Parameters
        ---------
        randomState : int, default None
            seed to set MDS and ensure reproducable results

        Returns
        ------
        None
            Shows a plot of the embedding.
        """

        if self.distances is None:
            metric = getattr(geodesics, self.metric)
            self.keys, self.distances = metric(files=self.outDir, filterfunction=None)
        mds = MDS(dissimilarity="precomputed", random_state=randomState)
        X = mds.fit_transform(X=self.distances)
        df = pd.DataFrame(X, columns=["x", "y"])

        ## --> FIXME weird bug where, when running multiple times, this
        # continues appending to the front of the color scale
        colorscale = px.colors.sequential.Reds.copy()
        colorscale.insert(0, "rgba(255, 255, 255, 0)")

        # Create figure
        fig = go.Figure()

        # Add 2D histogram contour
        fig.add_trace(
            go.Histogram2dContour(
                colorbar={"title": "", "tickvals": []},
                x=df["x"],
                y=df["y"],
                colorscale=colorscale,
                xaxis="x",
                yaxis="y",
            )
        )

        # Add scatter plot
        fig.add_trace(
            go.Scatter(
                x=df["x"],
                y=df["y"],
                mode="markers",
                marker={
                    "color": "grey",
                    "opacity": 0.5,
                    "line": {"width": 1, "color": "black"},
                },
                customdata=df.index,
                hovertemplate="Index: %{customdata}<extra></extra>",
            )
        )

        # Update fig axis template
        fig.update_layout(template="none", width=1100, height=700, margin={"r": 100})
        fig.add_annotation(
            x=1.1,
            y=0.5,
            xref="paper",
            yref="paper",
            text="Model Density",
            showarrow=False,
            font=dict(size=15),
            textangle=90,
        )
        # Show figure
        fig.show()

    def save(self, file_path):
        """
        Save the current object instance to a file using pickle serialization.

        Parameters
        ---------
            file_path:  str
              The path to the file where the object will be saved.

        """
        try:
            with open(file_path, "wb") as f:
                pickle.dump(self, f)
        except Exception as e:
            print(e)

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
        return self.clusters
