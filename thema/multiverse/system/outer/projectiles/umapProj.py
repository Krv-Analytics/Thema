# File: multiverse/system/outer/projectiles/umapProj.py
# Last Update: 05/15/24
# Updated by: JW

from umap import UMAP

from ..comet import Comet


def initialize():
    """
    Returns the umapProj class object from module. This is a general method
    that allows us to initialize arbitrary projectile objects.

    Returns
    -------
    umapProj : object
        The UMAP projectile object.
    """
    return umapProj


class umapProj(Comet):
    """
    UMAP Projectile Class.

    Inherits from Comet.

    Projects data into lower dimensional space using the
    Uniform Manifold Approximation and Projection.
    See: https://umap-learn.readthedocs.io/en/latest/

    Parameters
    ----------
    data_path : str
        A path to the raw data file.
    clean_path : str
        A path to a cofigured Moon object file.
    nn : int
        The number of nearest neighbors for UMAP alg.
    minDist : float
        The minimum distance threshold for clustering.
    dimensions : int
        The number of dimensions for the embedding.
    seed : int
        The seed for randomization.

    Attributes
    ----------
    data : pd.DataFrame
        A pandas dataframe of raw data.
    clean : pd.DataFrame
        A pandas dataframe of complete, encoded, and scaled data.
    projectionArray : np.array
        A projection array.
    nn : int
        Number of nearest neighbors.
    minDist : float
        Minimum distance threshold for clustering.
    dimensions : int
        Number of dimensions for the embedding.
    seed : int
        Seed for randomization.

    Methods
    -------
    fit()
        Fits a UMAP projection from given parameters and
        saves to projectionArray.
    save()
        Saves umapProj to `.pkl` serialized object file.
    """

    def __init__(self, data_path, clean_path, nn, minDist, dimensions, seed):
        super().__init__(data_path=data_path, clean_path=clean_path)
        self.nn = nn
        self.minDist = minDist
        self.dimensions = dimensions
        self.seed = seed

    def fit(self):
        """
        Performs a UMAP projection based on the configuration parameters.

        Returns
        -------
        None
            Initializes projectionArray member.
        """
        data = self.clean
        if data.isna().any().any() and self.verbose:
            print(
                "Warning: your data contains NA values that will be \
                dropped without remorse before projecting."
            )

        data = data.dropna()

        umap = UMAP(
            min_dist=self.minDist,
            n_neighbors=self.nn,
            n_components=self.dimensions,
            init="random",
            random_state=self.seed,
            n_jobs=1,
        )

        self.projectionArray = umap.fit_transform(data)
