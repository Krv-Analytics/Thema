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
    
    def __init__(self, data, projector: str="UMAP", verbose:bool=True, id:int=None, **kwargs):
        """TODO: Update with Proper Doc String"""
        
        if projector == "UMAP" and not kwargs:  
            kwargs = {
                'nn': 4,
                'dimensions': 2,
                'minDist': 0.1,
                'seed': 42
            }
        self.verbose = verbose

        if id is None: 
            self.id = ""
        else: 
            self.id = str(id)

        # Setting data member 
        if type(data) == str:
            try:
                assert os.path.isfile(data), "\n Invalid path to Clean Data"
                with open(data, "rb") as clean:
                    reference = pickle.load(clean)
                    if isinstance(reference, dict):
                        self.data = reference["data"]
                    elif isinstance(reference, pd.DataFrame):
                        self.data = reference 
                    else: 
                        raise ValueError("Please provide a data path to a dictionary (with a 'data' key) or a pandas Dataframe")
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
            assert self.data.shape[0] >= kwargs["perplexity"]
            self.perplexity = kwargs["perplexity"]
            self.dimensions = kwargs["dimensions"]
            self.seed = kwargs["seed"]
        
        # Configuring PCA Parameters 
        elif self.projector == "PCA": 
            self.dimensions = kwargs["dimensions"]
            self.seed = kwargs["seed"]

        else: 
            raise ValueError("Only UMAP, PCA, and TSNE are supported.")




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
        
        # Ensure No NAs before projection  
        data = self.data.dropna()
        
        # Fitting UMAP 
        if self.projector == "UMAP":
            umap = UMAP(
                min_dist=self.minDist,
                n_neighbors=self.nn,
                n_components=self.dimensions,
                init="random",
                random_state=self.seed,
                n_jobs=1
            )

            self.projection = umap.fit_transform(data)
            self.results = {"projection": self.projection, "description": {"projector": self.projector, 
                                                                           "nn": self.nn, 
                                                                           "minDist": self.minDist,
                                                                            "dimensions": self.dimensions, 
                                                                            "seed": self.seed, 
                                                                            "clean": self.data_path}}

        # Fitting TSNE 
        elif self.projector == "TSNE":
            tsne = TSNE(n_components=self.dimensions, random_state=self.seed, perplexity=self.perplexity)
            self.projection = tsne.fit_transform(data)
            self.results = {"projection": self.projection, "description": {"projector": self.projector,
                                                                           "perplexity": self.perplexity, 
                                                                           "dimensions": self.dimensions,
                                                                           "seed": self.seed, 
                                                                           "clean": self.data_path}}

        # Fitting PCA 
        elif self.projector == "PCA":
            pca = PCA(n_components=self.dimensions, random_state=self.seed)
            self.projection = pca.fit_transform(self.data)
            self.results = {"projection": self.projection, "description": {"projector": self.projector, 
                                                                           "dimensions": self.dimensions, 
                                                                           "seed": self.seed, 
                                                                           "clean": self.data_path}}
        
        # Unknown Projector Case Handling 
        else: 
            raise ValueError("Only UMAP, TSNE, and PCA are currently supported. Please make sure you have set the correct projector.")

    
    def dump(self, out_dir, id=None): 
        """TODO: Update with Proper Doc String"""

        id = self.id + str(id)
        try:       
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
        except Exception as e: 
            print(e)
        
        if self.projector == "UMAP": 
            output_file = projection_file_name(
                projector=self.projector,
                id=id,
                nn=self.nn,
                minDist=self.minDist,
                dimensions=2,
                seed=self.seed,
                )
        if self.projector == "TSNE": 
            output_file = projection_file_name(
                projector=self.projector,
                id=id,
                dimensions=self.dimensions,
                perplexity = self.perplexity,
                seed=self.seed,
                )
        
        if self.projector == "PCA": 
            output_file = projection_file_name(
                projector=self.projector,
                id=id,
                dimensions=self.dimensions,
                seed=self.seed,
                )
        
        # Create absolute file path 
        output_file = os.path.join(out_dir, output_file)
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
                f"Written to {rel_outdir}",
            )
            print("\n")
            print(
                "-------------------------------------------------------------------------------------- \n\n"
            )


        

    def save(self, file_path): 
        """
        Save the current object instance to a file using pickle serialization.

        Parameters:
            file_path (str): The path to the file where the object will be saved.

        """
        try:
            with open(file_path, "wb") as f:
                pickle.dump(self, f)
        except Exception as e:
            print(e)


