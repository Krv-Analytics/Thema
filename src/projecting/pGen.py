# File: src/projecting/gGen.py 
# Last Update: 03-04-24
# Updated by: SW 


import argparse
import os
import pickle
import re
from omegaconf import OmegaConf
from termcolor import colored
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from .projecting_utils import projection_file_name

class pGen:
    """
    Projection Generator Class


     TODO: Update With Proper Doc String
    """
    
    def __init__(self, data, n, d, YAML_PATH, verbose=True):
        """TODO: Update with Proper Doc String"""
        try:
            with open(YAML_PATH, "r") as f:
                self.params = OmegaConf.load(f)
        except:
            print("params.yaml file note found!")

        assert os.path.isfile(data), "\n Invalid path to Clean Data"
        self.data_path = data
        with open(data, "rb") as clean:
            reference = pickle.load(clean)
            self.data = reference["clean_data"]
        rel_outdir = (
        "data/" + self.params["Run_Name"] + "/projections/" + self.params["projector"] + "/"
    )
        self.output_dir = os.path.join(self.params.root_dir, rel_outdir)
        self.verbose = verbose
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.projector = self.params.projector
        self.nn = n 
        self.mindist = d
        assert self.projector in [
        "UMAP",
        "TSNE",
        "PCA",
        ], "Only UMAP, TSNE, and PCA are currently supported."


    # Fitting a Projection
    def fit(self, n, d): 
        """
        This function performs a projection of a DataFrame.

        Returns:
        -----------
        dict
            A dictionary containing the projection and hyperparameters.
    """

        data = self.data.dropna()

        if self.projector == "UMAP":
            umap_2d = UMAP(
                min_dist=d,
                n_neighbors=n,
                n_components=self.params.projector_dimension,
                init="random",
                random_state=self.params.projector_random_seed,
            )

            projection = umap_2d.fit_transform(data)

        if self.projector == "TSNE":
            num_samples = self.data.shape[0]
            perplexity = min(30, num_samples - 1)
            tsne = TSNE(n_components=self.params.projector_dimension, random_state=self.params.projector_random_seed, perplexity=perplexity)
            projection = tsne.fit_transform(data)

        if self.projector == "PCA":
            pca = PCA(n_components=self.params.projector_dimension, random_state=self.params.projector_random_seed)
            projection = pca.fit_transform(data)
        
        
        self.results = {"projection": projection, "hyperparameters": [n, d, self.params.projector_dimension]}

    
    def save_to_file(self): 
        """TODO: Update with Proper Doc String"""
        
        # TODO: move to helper file 
        # Get Imputation ID
        match = re.search(r"\d+", self.data_path)
        impute_id = match.group(0)
        
        if self.projector == "UMAP": 
            output_file = projection_file_name(
                projector=self.params.projector,
                impute_method=self.params.data_imputation.method,
                impute_id=impute_id,
                n=self.nn,
                d=self.mindist,
                dimensions=2,
                seed=self.params.projector_random_seed,
                )
        if self.projector == "TSNE": 
            output_file = projection_file_name(
                projector=self.params.projector,
                impute_method=self.params.data_imputation.method,
                impute_id=impute_id,
                dimensions=2,
                perplexity = self.perplexity,
                seed=self.params.projector_random_seed,
                )
        
        if self.projector == "PCA": 
            output_file = projection_file_name(
                projector=self.params.projector,
                impute_method=self.params.data_imputation.method,
                impute_id=impute_id,
                dimensions=2,
                seed=self.params.projector_random_seed,
                )
        
        output_file = os.path.join(self.output_dir, output_file)
        with open(output_file, "wb") as f:
            pickle.dump(self.results, f)
        
        # Output Message
        rel_outdir = "/".join(output_file.split("/")[-3:])

        with open(output_file, "wb") as f:
            pickle.dump(self.results, f)

        if  self.verbose:
            print("\n")
            print(
                "-------------------------------------------------------------------------------------- \n\n"
            )

            print(
                colored(f"SUCCESS: Completed Projection!", "green"),
                "Written to {rel_outdir}",
            )
            print("\n")
            print(
                "-------------------------------------------------------------------------------------- \n\n"
            )


