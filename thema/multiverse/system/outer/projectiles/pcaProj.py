# File: multiverse/system/outer/projectiles/pcaProj.py
# Last Update: 05/15/24
# Updated by: JW

import logging
from sklearn.decomposition import PCA

from ..comet import Comet

# Configure module logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def initialize():
    """
    Returns the pcaeProj class object from module. This is a general method
    that allows us to initialize arbitrary projectile objects.

    Returns
    -------
    pcaProj : object
        The PCA projectile object.
    """
    return pcaProj


class pcaProj(Comet):
    """
    PCA Projectile Class.

    Inherits from Comet.

    Projects data into lower dimensional space using sklearn's PCA Projection.
    See: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

    Attributes
    ----------
    data : pd.DataFrame
        A pandas dataframe of raw data.
    clean : pd.DataFrame
        A pandas dataframe of complete, encoded, and scaled data.
    projectionArray : np.array
        A projection array.
    dimensions : int
        Number of dimensions for the embedding.
    seed : int
        Seed for randomization.

    Methods
    -------
    __init__(self, data_path, clean_path, dimensions, seed)
        Constructs a pcaProj instance.

    fit(self)
        Performs a PCA projection based on the configuration parameters.

    save(self)
        Saves pcaProj to `.pkl` serialized object file.
    """

    def __init__(self, data_path, clean_path, dimensions, seed):
        """
        Constructs a pcaProj instance.

        Parameters
        ----------
        data_path : str
            The path to the data file.
        clean_path : str
            The path to save the cleaned data file.
        dimensions : int
            The number of dimensions for the embedding.
        seed : int
            The seed for randomization.
        """
        super().__init__(data_path=data_path, clean_path=clean_path)
        self.dimensions = dimensions
        self.seed = seed
        self.projectionArray = None

    def fit(self):
        """
        Performs a PCA projection based on the configuration parameters.

        Returns
        -------
        None
            Initializes projectionArray member.
        """
        data = self.clean
        pca = PCA(n_components=self.dimensions, random_state=self.seed)
        self.projectionArray = pca.fit_transform(data)
