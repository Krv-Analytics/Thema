# File: src/thema.py 
# Last Update: 03-13-24
# Updated by: SW 

import os
import pickle 
import pandas as pd 
from omegaconf import OmegaConf

from .cleaning.iSpace import iSpace
from .projecting.pGen import pGen 


class Thema: 
    """
    A Class to drive all of your Topological Hyperparameter Evaluations and Mapping Needs! 
    """

    def __init__(self, YAML_PATH):
        
        if os.path.isfile(YAML_PATH) and YAML_PATH.endswith(".yaml"):
            self.YAML_PATH = YAML_PATH
            with open(YAML_PATH, "r") as f:
                self.params = OmegaConf.load(f) 
            self.YAML_PATH = YAML_PATH
        else:
            raise ValueError("There was an issue with your yaml parameter file")
        
        self.clean_files = None 
        self.projection_files = None 
        self.graph_files = None 


    def get_iSpace_object(self, example_yaml=None): 
        # instantiates with yaml file oarameters ad returns a an iSpace object 
        pass
    
    def get_pSpace_object(self, example_yaml=None): 
        # instantiatnes with yaml parameters a pSpace object
        pass

    def get_Selection_object(self, example_yaml=None):
        # instantiates selectin yaml with a blah blah 
        pass 

    def fit_iSpace(self):
        """
        TODO: Update Doc String 
        """ 
        clean_outdir = os.path.join(self.params.out_dir, self.params.runName + "/clean")
        if os.path.isdir(clean_outdir):
            assert os.listdir(clean_outdir) == 0, "Your clean data directory is not empty. Please clean it with .sweep_iSpace()" 
        else: 
            os.mkdir(clean_outdir)
        thema_iSpace = iSpace(YAML_PATH=self.YAML_PATH)
        thema_iSpace.fit_space(out_dir=clean_outdir) 
        self.clean_files = os.listdir(clean_outdir)

    def sweep_iSpace(self): 
        """
        TODO: Update Doc String 
        """
        clean_outdir = os.path.join(self.params.out_dir, self.params.runName + "/clean")
        if os.path.isdir(clean_outdir):
            for filename in os.listdir(clean_outdir):
                file_path = os.path.join(clean_outdir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        os.rmdir(filename)
                except Exception as e:
                    print(f"Error while deleting {file_path}: {e}")


    def fit_pSpace(self): 
        """
        
        """
        clean_outdir = os.path.join(self.params.out_dir, self.params.runName + "/clean")
        if os.path.isdir(clean_outdir):
            assert os.listdir(clean_outdir) > 0, "You do not have any clean data. Run .fit_iSpace() to generate some."
        
        
        # Instantiates pSpace and fits it 
    
    

    def sweep_pSpace(self): 
        pass 


    def fit_gSpace(self): 
        # Create and collapse 
        pass 

    def sweep_gSpace(self): 
        pass 
    

    def investigate_model(self, model_id): 
        
        # Instantiates an (old) thema object
        pass 
