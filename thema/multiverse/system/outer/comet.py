# File: thema/multiverse/system/outer/comet.py
# Last Update: 05/15/24
# Updated by: JW

import pickle
from abc import abstractmethod

from ....core import Core


class Comet(Core):
    """
    Collapse or Modify Existing Tabulars
    -----

    A COMET is a base class template for projection (dimensionality reduction)
    algorithms. As a parent class, Comet enforces structure on data management
    and projection, enabling a 'universal' procedure for
    generating these objects.

    Members
    -------
    data : pd.DataFrame
        a pandas dataframe of raw data
    clean : pd.DataFrame
        a pandas dataframe of complete, encoded, and scaled data

    Functions
    --------
    save()
        saves Comet to `.pkl` serialized object file

    See Also
    -------
    docs/development/comet.md
        see for more information on implementing a realization of Comet

    Examples
    --------
    >>> from thema.multiverse.system.outer import Comet
    >>> class PCA(Comet):
    ...     def fit(self):
    ...         pass
    >>> pca = PCA(data_path='data.csv', clean_path='clean.csv')
    >>> pca.fit()
    """

    def __init__(self, data_path: str, clean_path: str):
        """
        Initialize a new instance of the class.

        Parameters
        ----------
        data_path : str
            The path to the data file.
        clean_path : str
            The path to save the cleaned data file.
        """
        super().__init__(
            data_path=data_path, clean_path=clean_path, projection_path=None
        )

    @abstractmethod
    def fit(self):
        """
        Abstract method to be implemented by Comet's child.

        Notes
        -----
        Method must initialize the projectionArray member.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the child class.
        """
        raise NotImplementedError

    def save(self, file_path):
        """
        Save the current object instance to a file using pickle serialization.

        Parameters
        ----------
            file_path (str): The path to the file w
                                here the object will be saved.
        Raises
        ----------
            Exception: If the file cannot be saved.

        Examples
        --------
        >>> from thema.multiverse.system.outer import Comet
        >>> class PCA(Comet):
        ...     def fit(self):
        ...         pass
        >>> pca = PCA(data_path='data.csv', clean_path='clean.csv')
        >>> pca.fit()
        >>> pca.save('pca.pkl')
        """
        try:
            with open(file_path, "wb") as f:
                pickle.dump(self, f)
        except Exception as e:
            print(e)
