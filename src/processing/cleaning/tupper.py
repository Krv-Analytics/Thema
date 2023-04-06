"Dishwasher safe data container"
from os.path import isfile
import pickle


class Tupper:
    def __init__(self, raw: str, clean: str, projection: str):

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
        assert self._raw, "Please Specify a valid path to raw data"
        with open(self._raw, "rb") as raw_file:
            raw_df = pickle.load(raw_file)

        return raw_df

    @property
    def clean(self):
        assert self._clean, "Please Specify a valid path to clean data"
        with open(self._clean, "rb") as clean_file:
            reference = pickle.load(clean_file)
        clean_df = reference["clean_data"]
        # dropped_columns = reference["dropped_columns"]
        return clean_df

    @property
    def projection(self):
        assert self._projection, "Please Specify a valid path to clean data"
        with open(self._projection, "rb") as projection_file:
            reference = pickle.load(projection_file)
            projection_array = reference["projection"]

        return projection_array

    def get_projection_parameters(self):
        assert self._projection, "Please Specify a valid path to clean data"
        with open(self._projection, "rb") as projection_file:
            reference = pickle.load(projection_file)
            projection_parameters = reference["hyperparameters"]

        return projection_parameters
