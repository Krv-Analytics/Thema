"Object file for Tupper."
from os.path import isfile
import pickle


class Tupper:
    """A data container class for the various versions needed when modeling using the Mapper algorithm.

    This class points to the locations of three local versions of the user's data:
        1) raw: data pulled directly from a database (e.g. Mongo)
        2) clean: data that has been cleaned via dropping features, scaling, removing NaNs, etc.
        3) projected: data that has been collapsed using a dimensionality reduction technique (e.g. PCA, UMAP).
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
        self._raw = None
        self._clean = None
        self._projection = None

        # If files exist, set members
        if isfile(raw):
            self._raw = raw
        if isfile(raw):
            self._clean = clean
        if isfile(projection):
            self._projection = projection

    @property
    def raw(self):
        """Get the raw data in your Tupper."""
        assert self._raw, "Please Specify a valid path to raw data"
        with open(self._raw, "rb") as raw_file:
            raw_df = pickle.load(raw_file)

        return raw_df

    @property
    def clean(self):
        """Get the clean data in your Tupper."""
        assert self._clean, "Please Specify a valid path to clean data"
        with open(self._clean, "rb") as clean_file:
            reference = pickle.load(clean_file)
        clean_df = reference["clean_data"]
        # dropped_columns = reference["dropped_columns"]
        return clean_df

    @property
    def projection(self):
        """Get the projected data in your Tupper."""
        assert self._projection, "Please Specify a valid path to clean data"
        with open(self._projection, "rb") as projection_file:
            reference = pickle.load(projection_file)
            projection_array = reference["projection"]

        return projection_array

    def get_projection_parameters(self):
        """Get the parameters used to generate the projected data in your Tupper object."""
        assert self._projection, "Please Specify a valid path to projected data"
        with open(self._projection, "rb") as projection_file:
            reference = pickle.load(projection_file)
            projection_parameters = reference["hyperparameters"]

        return projection_parameters
