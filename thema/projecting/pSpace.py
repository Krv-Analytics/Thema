# File: src/projecting/projection_utils.py 
# Last Update: 03-04-24
# Updated by: SW 

import os
from omegaconf import OmegaConf
from .pGen import pGen 
from ..utils import function_scheduler

class pSpace(): 
    """
    Projection Space 

    TODO: Update with Proper Doc String
    """
    def __init__(self, projector=None, clean_dir=None, out_dir=None, YAML_PATH=None, **kwargs): 
        """
         TODO: Update with Proper Doc String
        """
        if YAML_PATH is not None: 
            assert os.path.isfile(YAML_PATH), "yaml parameter file could not be found."
            try: 
                with open(YAML_PATH, "r") as f:
                    params = OmegaConf.load(f)
            except Exception as e: 
                print(e)

            projector = params.projecting.projector
            clean_dir = os.path.join(self.params.out_dir, self.params.runName + "/clean/") 
            out_dir = os.path.join(self.params.out_dir, self.params.runName + "/projections/") 

            if projector == "UMAP": 
                minDists = params.projecting.UMAP.minDists 
                nn = params.projecting.UMAP.nn 
                dimensions = params.projecting.dimensions
                seed = params.projecting.UMAP.seed 
            elif projector == "TSNE": 
                perplexity = params.projecting.TSNE.perplexity 
                dimensions = params.projecting.dimensions 
                seed = params.projecting.TSNE.seed
            elif projector == "PCA": 
                dimensions = params.projecting.dimensions 
                seed = params.projecting.PCA.seed 
            else: 
                raise ValueError("Supported projectors: 'UMAP', 'TSNE', and 'PCA'")
        
        else: 
            assert projector is not None and clean_dir is not None and out_dir is not None, "Missing parameters!"

        if projector == "UMAP": 
            self.minDists = minDists
            self.nn = nn 
            self.dimensions = dimensions
            self.seed = seed 
        elif projector == "TSNE": 
            self.perplexity = perplexity 
            self.dimensions = dimensions 
            self.seed = seed
        elif projector == "PCA": 
            self.dimensions = dimensions 
            self.seed = seed 
        else: 
            raise ValueError("Supported projectors: 'UMAP', 'TSNE', and 'PCA' ")

        self.clean_dir = clean_dir
        self.out_dir = out_dir
        
        assert os.path.isdir(self.clean_dir), "Invalid clean data directory."
        assert os.listdir(self.clean_dir) > 0, "No clean data found. Please make sure you generated clean data."

        
        if not os.path.isdir(self.out_dir):
            try: 
                os.makedirs(out_dir)
            except Exception as e:
                print(e)

    
    def fit(self):  
        """
         TODO: Update with Proper Doc String
        """   
        if self.projector == "UMAP": 
            # Number of loops
            num_loops = len(self.minDists) * len(self.nn) * len(self.seed) * len(self.clean_dir)

            # Creating list of Subprocesses
            subprocesses = []

            ## GRID SEARCH PROJECTIONS
            for nn in self.nn:
                for minDist in self.minDists:
                    for seed in self.seed:
                        for clean_id, clean_data_file in enumerate(sorted(os.listdir(self.clean_dir))):
                            cmd = (self._instantiate_umap_projection, clean_data_file, clean_id, nn, minDist, self.dimensions, seed)
                            subprocesses.append(cmd)

        elif self.projector == "TSNE": 
            num_loops = len(self.perplexity) * len(self.seed) * len(self.clean_dir)
            for perp in self.perplexity: 
                for seed in self.seed: 
                    for clean_id, clean_data_file in enumerate(sorted(os.listdir(self.clean_dir))):
                        cmd = (self._instantiate_tsne_projection, clean_data_file, clean_id, perp, self.dimensions, seed)
                        subprocesses.append(cmd)

        elif self.projector == "PCA": 
            num_loops = len(self.dimensions) * len(self.seed) * len(os.listdir(self.clean_dir))
            for seed in self.seed: 
                for clean_id, clean_data_file in enumerate(sorted(os.listdir(self.clean_dir))):
                    cmd = (self._instantiate_pca_projection, clean_data_file, clean_id, self.dimensions, seed)
                    subprocesses.append(cmd)
        
        else: 
            raise ValueError("Supported projectors: 'UMAP', 'TSNE', and 'PCA'")
        
        # Handles Process scheduling
        function_scheduler(
            subprocesses, num_loops, "SUCCESS: Projections Grid", resilient=True
        )

    

    def _instantiate_umap_projection(self, clean_data_file, clean_id, nn, minDist, dimensions, seed):
        """ Private Member Function Used in Parallelization"""
        my_projector = pGen(data=clean_data_file, projector="UMAP", clean_id=clean_id, nn=nn, minDist=minDist, dimensions=dimensions, seed=seed, verbose=False)
        my_projector.fit() 
        my_projector.dump(self.out_dir)
    
    def _instantiate_tsne_projection(self, clean_data_file, clean_id, perplexity, dimensions, seed): 
        """ Private Member Function Used in Parallelization"""
        my_projector = pGen(data=clean_data_file, projector="TSNE", clean_id=clean_id, perplexity=perplexity, dimensions=dimensions, seed=seed, verbose=False)
        my_projector.fit() 
        my_projector.dump(self.out_dir)

    def _instantiate_pca_projection(self, clean_data_file, clean_id, dimensions, seed): 
        """ Private Member Function Used in Parallelization"""
        my_projector = pGen(data=clean_data_file, projector="PCA",clean_id=clean_id, dimensions=dimensions, seed=seed, verbose=False)
        my_projector.fit() 
        my_projector.dump(self.out_dir)
    


    def write_params_to_yaml(self, yaml_path): 
        pass 