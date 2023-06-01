"Object file for Tupper."
import os
from os.path import isfile
from dotenv import load_dotenv
import pickle


class Tupper:
    """A data container class for the various versions needed when modeling using the Mapper algorithm.

    This class points to the locations of three local versions of the user's data:
        1) raw: data pulled directly from a database (e.g. Mongo)
        2) clean: data that has been cleaned via dropping features, scaling, removing NaNs, etc.
        3) projected: data that has been collapsed using a dimensionality reduction technique (e.g. PCA, UMAP).

        Note! These paths are the relative paths from the root of this repository!
    """

    def __init__(self, raw: str, clean: str, projection: str):
        """Constructor for Tupper class.
        Parameters
        ===========
        raw: str
            A path to raw data pickle file.

        clean: str
            A path to clean data pickle file.

        projection: str
            A path to projected data pickle file.
        """

        # Get relative file paths
        load_dotenv()
        root = os.getenv("root")
        raw_abs = os.path.join(root, raw)
        clean_abs = os.path.join(root, clean)
        projection_abs = os.path.join(root, projection)

        self._raw = None
        self._clean = None
        self._projection = None

        # Require Valid file paths
        assert isfile(raw_abs), f"Invalid raw file path: {raw_abs}"
        assert isfile(clean_abs), f"Invalid clean file path: {clean}"
        assert isfile(projection_abs), f"Invalid projection file path: {projection}"

        self._raw = raw
        self._clean = clean
        self._projection = projection

    @property
    def raw(self):
        """returns the raw data in your Tupper."""
        assert self._raw, "Please Specify a valid path to raw data"
        load_dotenv()
        root = os.getenv("root")
        raw_abs = os.path.join(root, self._raw)
        try:
            # Loading data from the pickle file
            with open(raw_abs, "rb") as raw_file:
                raw_df = pickle.load(raw_file)
            return raw_df
        except Exception as e:
            print(
                "There was an error opening your raw data. \
                  Please make sure you have set your raw data reference to the correct pickle file.\n",
                e,
            )

    @property
    def clean(self):
        """Get the clean data in your Tupper."""
        assert self._clean, "Please Specify a valid path to clean data"
        load_dotenv()
        root = os.getenv("root")
        clean_abs = os.path.join(root, self._clean)
        try:
            # Loading clean data from pickle file
            with open(clean_abs, "rb") as clean_file:
                reference = pickle.load(clean_file)
            clean_df = reference["clean_data"]
            return clean_df
        except Exception as e:
            print(
                "There was an error opening your clean data \
                  Please make sure your have set your clean data reference to the correct pickle file.\n",
                e,
            )

    def get_dropped_columns(self):
        """Returns a list of the dropped columns when creating cleaned data"""
        assert self._clean, "Please Specify a valid path to projected data"
        load_dotenv()
        root = os.getenv("root")
        clean_abs = os.path.join(root, self._clean)
        try:
            with open(clean_abs, "rb") as clean_file:
                reference = pickle.load(clean_file)
            dropped_columns = reference["dropped_columns"]
            return dropped_columns
        except Exception as e:
            print(
                "There was an error opening your clean data. \
                  Please make sure you have set your clean data reference to the correct pickle file.\n",
                e,
            )

    @property
    def projection(self):
        """Get the projected data in your Tupper."""
        assert self._projection, "Please Specify a valid path to clean data"
        load_dotenv()
        root = os.getenv("root")
        projection_abs = os.path.join(root, self._projection)
        try:
            with open(projection_abs, "rb") as projection_file:
                reference = pickle.load(projection_file)
            projection_array = reference["projection"]
            return projection_array
        except Exception as e:
            print(
                "There was an error opening your projection data. \
                 Please make sure you have set your projection data reference to the correct pickle file.\n",
                e,
            )

    def get_projection_parameters(self):
        """Get the parameters used to generate the projected data in your Tupper object."""
        assert self._projection, "Please Specify a valid path to projected data"
        load_dotenv()
        root = os.getenv("root")
        projection_abs = os.path.join(root, self._projection)
        try:
            with open(projection_abs, "rb") as projection_file:
                reference = pickle.load(projection_file)
            projection_parameters = reference["hyperparameters"]
            return projection_parameters
        except Exception as e:
            print(
                "There was an error opening your hyperparameters. \
                   Please make sure you have set your projection data reference to the correct pickle file.\n",
                e,
            )
