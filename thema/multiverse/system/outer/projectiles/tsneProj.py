# File: multiverse/system/outer/projectiles/tsneProj.py
# Last Update: 05/15/24
# Updated by: JW

import logging
from sklearn.manifold import TSNE

from ..comet import Comet

# Configure module logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def initialize():
    """
    Returns the tsneProj class object from module. This is a general method
    that allows us to initialize arbitrary projectile objects.

    Returns
    -------
    tsneProj : object
        The t-SNE projectile object.
    """
    return tsneProj


class tsneProj(Comet):
    """
    t-SNE Projectile Class.

    Inherits from Comet.

    Projects data into lower dimensional space using the
    T-distributed Stochastic Neighbor Embedding.
    See: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

    Members
    -------
    data : pd.DataFrame
        A pandas dataframe of raw data.
    clean : pd.DataFrame
        A pandas dataframe of complete, encoded, and scaled data.
    projectionArray : np.array
        A projection array.
    perplexity : int
        A tsne configuration parameter.
    dimensions : int
        Number of dimensions for the embedding.
    seed : int
        Seed for randomization.

    Functions
    --------
    fit()
        Fits a tsne projection from given parameters and saves to projectionArray.
    save()
        Saves tsneProj to `.pkl` serialized object file.
    """

    def __init__(self, data_path, clean_path, perplexity, dimensions, seed):
        """
        Constructs a tsneProj instance.

        Parameters
        ----------
        data_path : str
            The path to the data file.
        clean_path : str
            The path to save the cleaned data file.
        perplexity : int
            Related to the number of nearest neighbors.
        dimensions : int
            The number of dimensions for the embedding.
        seed : int
            The seed for randomization.
        """

        super().__init__(data_path, clean_path)
        self.perplexity = perplexity
        self.dimensions = dimensions
        self.seed = seed
        

    def fit(self):
        """
        Performs a TSNE projection based on the configuration parameters.

        Returns
        ------
        None
            Initializes projectionArray member.
        """
        data = self.clean
        tsne = TSNE(
            n_components=self.dimensions,
            random_state=self.seed,
            perplexity=self.perplexity,
        )
        self.projectionArray = tsne.fit_transform(data)
