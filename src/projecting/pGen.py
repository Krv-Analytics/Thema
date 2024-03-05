# File: src/projecting/gGen.py 
# Last Update: 03-04-24
# Updated by: SW 

import os
import pickle
import re
from omegaconf import OmegaConf
from termcolor import colored
import pandas as pd
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from .projecting_utils import projection_file_name

class pGen:
    """
    Projection Generator Class


     TODO: Update With Proper Doc String
    """
    
    def __init__(self, data, projector="UMAP", verbose=True, **kwargs):
        """TODO: Update with Proper Doc String"""
        
        self.verbose = verbose
        # Setting data member 
        if type(data) == str:
            try:
                assert os.path.isfile(data), "\n Invalid path to Clean Data"
                with open(data, "rb") as clean:
                    reference = pickle.load(clean)
                    self.data = reference["clean_data"]
                self.data_path = data
            except:
                print("There was an issue opening your data file. Please make sure it exists and is a pickle file.")
        
        elif type(data)==pd.DataFrame:
            self.data = data 
            self.data_path = -1 
        else: 
            raise ValueError("'data' must be a pickle file or pd.DataFrame object")
        
        # Setting Projector type 
        assert projector in [
        "UMAP",
        "TSNE",
        "PCA",
        ], "Only UMAP, TSNE, and PCA are currently supported."
        self.projector = projector 

        # Configuring UMAP Parameters  
        if self.projector == "UMAP": 
            self.nn = kwargs["nn"] 
            self.minDist = kwargs["minDist"]
            self.dimensions = kwargs["dimensions"]
            self.seed = kwargs["seed"]

        # Configuring TSNE Parameters 
        elif self.projector == "TSNE": 
            self.perplexity = kwargs["perplexity"]
            self.dimensions = kwargs["dimensions"]
            self.seed = kwargs["seed"]
        
        # Configuring PCA Parameters 
        elif self.projector == "PCA": 
            self.dimensions = kwargs["dimensions"]
            self.seed = kwargs["seed"]




    # Fitting a Projection
    def fit(self): 
        """
        This function performs a projection of a DataFrame.

        Returns:
        -----------
        dict
            A dictionary containing the projection and hyperparameters.
        """

        # Print a warning if containing NA values 
        if self.data.isna().any().any() and self.verbose: 
            print("Warning: your data contains NA values that will be dropped without remorse before projecting.")
        
        # Drop NA before projection 
        data = self.data.dropna()
        
        # Fitting UMAP 
        if self.projector == "UMAP":
            umap_2d = UMAP(
                min_dist=self.minDist,
                n_neighbors=self.nn,
                n_components=self.dimensions,
                init="random",
                random_state=self.seed,
            )

            projection = umap_2d.fit_transform(data)
            self.results = {"projection": projection, "description": [self.projector, self.nn, self.minDist, self.dimensions, self.seed, self.data_path]}

        # Fitting TSNE 
        elif self.projector == "TSNE":
            num_samples = self.data.shape[0]
            perplexity = min(30, num_samples - 1)
            tsne = TSNE(n_components=self.dimensions, random_state=self.seed, perplexity=self.perplexity)
            projection = tsne.fit_transform(data)
            self.results = {"projection": projection, "description": [self.projector, self.perplexity, self.seed, self.data_path]}

        # Fitting PCA 
        elif self.projector == "PCA":
            pca = PCA(n_components=self.params.projector_dimension, random_state=self.params.projector_random_seed)
            self.projection = pca.fit_transform(self.data)
            self.results = {"projection": projection, "description": [ self.projector, self.dimensions, self.seed, self.data_path]}
        
        # Unknown Projector Case Handling 
        else: 
            raise ValueError("Only UMAP, TSNE, and PCA are currently supported. Please make sure you have set the correct projector.")

    
    def save_to_file(self, out_dir, impute_method=None, impute_id=None): 
        """TODO: Update with Proper Doc String"""
        try: 
            # Create Directory if it does not exist       
            if not os.path.isdir(out_dir):
                os.makedirs(self.out_dir)
            
            if self.projector == "UMAP": 
                output_file = projection_file_name(
                    projector=self.projector,
                    impute_method=impute_method,
                    impute_id=impute_id,
                    n=self.nn,
                    d=self.mindist,
                    dimensions=2,
                    seed=self.params.projector_random_seed,
                    )
            if self.projector == "TSNE": 
                output_file = projection_file_name(
                    projector=self.params.projector,
                    impute_method=impute_method,
                    impute_id=impute_id,
                    dimensions=2,
                    perplexity = self.perplexity,
                    seed=self.seed,
                    )
            
            if self.projector == "PCA": 
                output_file = projection_file_name(
                    projector=self.params.projector,
                    impute_method=impute_method,
                    impute_id=impute_id,
                    dimensions=2,
                    seed=self.seed,
                    )
            
            # Create absolute file path 
            output_file = os.path.join(self.out_dir, output_file)
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

        except Exception as e: 
            print(e)


