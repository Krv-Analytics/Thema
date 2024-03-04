import os
from omegaconf import OmegaConf

from ..projecting.projector import Projector
from .utils import env, get_imputed_files
from .gridTracking_helper import function_scheduler

class pGrid(): 

    def __init__(self, YAML_PATH): 
        self.YAML_PATH = YAML_PATH
        if os.path.isfile(YAML_PATH):
            with open(YAML_PATH, "r") as f:
                self.params = OmegaConf.load(f)
        else:
         print("params.yaml file note found!")

        # DATA
        self.raw = self.params["raw_data"]
        self.clean = self.params["clean_data"]
        self.data_imputation = self.params["data_imputation"]

        # Projector (UMAP) Parameters
        self.N_neighbors = self.params["projector_Nneighbors"]
        self.min_Dists = self.params["projector_minDists"]
        self.projector = self.params["projector"]

        # Projection Script 
        self.projector_script = os.path.join(self.params["root_dir"], "scripts/python/projector.py")

        # Updating Projections Path in self.params.yaml
        self.params["projected_data"] = (
        "data/" + self.params["Run_Name"] + "/projections/" + self.params["projector"] + "/"
        )
        try:
            with open(YAML_PATH, "w") as f:
                OmegaConf.save(self.params, f)
        except:
               print("Unable to write back to your parameter file. Please make sure you have set proper file permissions.")
        # Check that raw data exists
        assert os.path.isfile(os.path.join(self.params["root_dir"], self.raw)), "No raw data found. Please make sure you have specified the correct path in your params file."
        # Check that clean data exits
        assert os.path.isfile(os.path.join(self.params["root_dir"], self.clean)), "No clean data found. Please make sure you generated clean data."

    def fit(self):     
        clean_files = os.path.join(self.params["root_dir"], "data/" + self.params["Run_Name"] +"/clean")
        # TODO: add logging information pertaining to data handling when NAs are present
        imputation_files = get_imputed_files(clean_files, key=self.data_imputation.method)

        # Number of loops
        num_loops = len(self.N_neighbors) * len(self.min_Dists) * len(imputation_files)

        # Creating list of Subprocesses
        subprocesses = []

        ## GRID SEARCH PROJECTIONS
        for n in self.N_neighbors:
            for d in self.min_Dists:
                for f in imputation_files:
                    cmd = (self._instantiate_projection, n, d, f)
                    subprocesses.append(cmd)

        # Handles Process scheduling
        function_scheduler(
            subprocesses, num_loops, "SUCCESS: Projections Grid", resilient=True
        )

    def _instantiate_projection(self, n, d, f):
        my_projector = Projector(data=f, n=n, d=d, YAML_PATH=self.YAML_PATH, verbose=False)
        my_projector.fit(n,d) 
        my_projector.save_to_file()
