# File: core.py
# Last Updated: 05/15/24
# Updated By: JW


import pickle
from os.path import isfile

from .utils import unpack_dataPath_types


class Core:
    """A data container class for the various versions needed when modeling
    using the Mapper algorithm.

    This class points to the locations of three local versions
    of the user's data:
    1) data: raw data pulled directly from a database (e.g. Mongo),
        downloaded, or collected locally.
    2) clean: data that has been cleaned via dropping features, scaling,
        removing NaNs, etc.
    3) projection: data that has been collapsed using a dimensionality
        reduction technique (e.g. PCA, UMAP).

    Parameters
    ----------
    data_path : str
        A path to raw data pickle file (relative from root).

    clean_path : str
        A path to clean data pickle file (relative from root).

    projection_path : str
        A path to projected data pickle file (relative from root).

    Attributes
    ----------
    _data : str
        The path to the raw data pickle file.

    _clean : str
        The path to the clean data pickle file.

    _projection : str
        The path to the projected data pickle file.

    Methods
    -------
    data()
        Returns the raw data in your Core.

    clean()
        Get the clean data in your Core.

    projection()
        Get the projected data in your Core.

    get_data_path()
        Returns the path to the raw data file.

    get_clean_path()
        Returns the path to the clean data file.

    get_projection_path()
        Returns the path to the projection data file.

    set_data_path(path)
        Sets the raw data path to a new data file.

    set_clean_path(path)
        Sets the clean data path to a new data file.

    set_projection_path(path)
        Sets the projection data path to a new data file.

    Examples
    --------
    >>> core = Core("/path/to/raw/data.csv", "/path/to/clean.pkl", \
    "/path/to/projection.pkl")
    >>> core.data
    pd.DataFrame([[1, 2, 3], [4, 5, 6]])
    >>> core.clean
    pd.DataFrame([[1, 2, 3], [4, 5, 6]])
    >>> core.projection
    np.array([[1, 2, 3], [4, 5, 6]])
    """

    def __init__(self, data_path, clean_path, projection_path):
        self._data = data_path
        self._clean = clean_path
        self._projection = projection_path

        if self._data is not None:
            assert isfile(self._data), f"Invalid raw data file path: {data_path}"
        if self._clean is not None:
            assert isfile(self._clean), f"Invalid clean file path: {clean_path}"
        if self._projection is not None:
            assert isfile(
                self._projection
            ), f"Invalid projection file path: {projection_path}"

    @property
    def data(self):
        """
        Returns the raw data in your Core.

        Handles `.csv`, `.xlsx`, and `.pkl` file types,
        and returns pandas DataFrame.

        Parameters
        ----------
        None

        Returns
        -------
        pd.DataFrame
            The raw data in your Core.

        Raises
        ------
        ValueError
            If the raw data file was not properly initialized.

        Examples
        --------
        >>> core = Core("/path/to/raw/data.csv", "/path/to/clean.pkl", \
        "/path/to/projection.pkl")
        >>> core.data
        pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        """

        if self._data is not None:
            return unpack_dataPath_types(self._data)
        else:
            raise ValueError("Your raw data file was not initialized.")

    @property
    def clean(self):
        """
        Returns the clean data from your Core.

        This method handles `.pkl` files,
        reading in the clean data file and returning a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            The clean data from your Core.

        Raises
        ------
        ValueError
            If the clean data file was not properly initialized.

        Examples
        --------
        >>> core = Core("/path/to/raw/data.csv", "/path/to/clean.pkl", \
        "/path/to/projection.pkl")
        >>> core.clean
        pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        """

        if self._clean is not None:
            try:
                # Loading clean data from pickle file
                with open(self._clean, "rb") as clean_file:
                    moon = pickle.load(clean_file)
                return moon.imputeData
            except Exception as e:
                print(
                    "There was an error opening your clean data. "
                    "Please make sure you have set your clean data reference \
                    to the correct pickle file.\n",
                    e,
                )
        else:
            raise ValueError("Your clean data file was not initialized.")

    @property
    def projection(self):
        """
        Returns the projected data from your Core.
        
        This method handles `.pkl` files,
        reading in the clean data file and returning a numpy array.    

        Returns
        -------
        np.ndarray
            The projected data from your Core.

        Raises
        ------
        ValueError
            If the projection data file was not properly initialized.

        Examples
        --------
        >>> core = Core("/path/to/raw/data.csv", "/path/to/clean.pkl", \
        "/path/to/projection.pkl")
        >>> core.projection
        np.array([[1, 2, 3], [4, 5, 6]])
        """

        if self._projection is not None:
            with open(self._projection, "rb") as projection_file:
                projectile = pickle.load(projection_file)
            return projectile.projectionArray
        else:
            raise ValueError("You projection data file was not initialized.")

    def get_data_path(self):
        """
        Get path to the user's raw data file.

        Returns
        -------
        str
            Path to raw data.
        
        Examples
        --------
        >>> core = Core("/path/to/raw/data.csv", "/path/to/clean.pkl", \
        "/path/to/projection.pkl")
        >>> core.get_data_path()
        "/path/to/raw/data.csv"
        """
        return self._data

    def get_clean_path(self):
        """
        Get path to the associated clean file.

        Returns
        -------
        str
            Path to clean data.
        
        Examples
        --------
        >>> core = Core("/path/to/raw/data.csv", "/path/to/clean.pkl", \
        "/path/to/projection.pkl")
        >>> core.get_clean_path()
        "/path/to/clean.pkl"
        """
        return self._clean

    def get_projection_path(self):
        """
        Get path to the associated projection file.

        Returns
        -------
        str
            Path to projection data.
        
        Examples
        --------
        >>> core = Core("/path/to/raw/data.csv", "/path/to/clean.pkl", \
        "/path/to/projection.pkl")
        >>> core.get_projection_path()
        "/path/to/projection.pkl"
        """
        return self._projection

    def set_data_path(self, path):
        """
        Sets the data path to a new raw file.

        Parameters
        ----------
        path : str
            Path to new raw data file.
        
        Examples
        --------
        >>> core = Core("/path/to/raw/data.csv", "/path/to/clean.pkl", \
        "/path/to/projection.pkl")
        >>> core.data
        pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        >>> core.set_data_path("/path/to/new/data.csv")
        >>> core.get_data_path()
        "/path/to/new/data.csv"
        >>> core.data
        pd.DataFrame([[7, 8, 9], [10, 11, 12]])
        """

        self._data = path

    def set_clean_path(self, path):
        """
        Sets the clean data path to a new file.

        Parameters
        ----------
        path : str
            Path to new clean data file.
        
        Examples
        --------
        >>> core = Core("/path/to/raw/data.csv", "/path/to/clean.pkl", \
        "/path/to/projection.pkl")
        >>> core.clean
        pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        >>> core.set_clean_path("/path/to/new/clean.pkl")
        >>> core.get_clean_path()
        "/path/to/new/clean.pkl"
        >>> core.clean
        pd.DataFrame([[7, 8, 9], [10, 11, 12]])
        """
        self._clean = path

    def set_projection_path(self, path):
        """
        Sets the projection data path to a new file.

        Parameters
        ----------
        path : str
            Path to new projection data file.
        
        Examples
        --------
        >>> core = Core("/path/to/raw/data.csv", "/path/to/clean.pkl", \
        "/path/to/projection.pkl")
        >>> core.projection
        np.array([[1, 2, 3], [4, 5, 6]])
        >>> core.set_projection_path("/path/to/new/projection.pkl")
        >>> core.get_projection_path()
        "/path/to/new/projection.pkl"
        >>> core.projection
        np.array([[7, 8, 9], [10, 11, 12]])
        """

        self._projection = path
