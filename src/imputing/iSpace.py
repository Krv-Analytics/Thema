# File: src/imputing/iSpace.py 
# Last Upated: 03-04-24
# Updated By: SW

import os
from omegaconf import OmegaConf
from termcolor import colored
from .iGen import iGen
from ..utils import function_scheduler

class iSpace: 
    """
    Imputation Space 
    TODO: Update Doc String"""
    
    def __init__(self, YAML_PATH):
        """ TODO: Update Doc String"""
        assert os.path.isfile(YAML_PATH), "yaml config file not found!"
        try: 
            with open(YAML_PATH, "r") as f:
                self.params = OmegaConf.load(f)
        except Exception as e: 
            print(e)


        self.impute_method = self.params.data_imputation.method
        self.num_samples = self.params.data_imputation.num_samples

    
    def fit(self): 
        """Fits DataFrame Distribution"""
        try:
            # Non-sampling methods require only a single imputation
            assert self.impute_method in ["random_sampling", "drop", "mean", "median", "mode"], "Supprted imputation methods are 'random_sampling', 'drop', 'mean', 'median', 'mode'"
            if self.num_samples != 1 and self.impute_method != "random_sampling":
                print(
                    "\n\n-------------------------------------------------------------------------------- \n\n"
                )

                print(
                    colored("WARNING:", "yellow"),
                    f"`{self.impute_method}` does not require sampling and will only produce a single, deterministic imputation.",
                )
                print(
                    "\n\n-------------------------------------------------------------------------------- \n\n"
                )
                self.num_samples = 1

            # Non-sampling methods require only a single imputation
            if self.num_imputations <= 0:
                print(
                    "\n\n-------------------------------------------------------------------------------- \n\n"
                )

                print(
                    colored("WARNING:", "yellow"),
                    f"`num_samplings` must be > 0. Defaulting to 1.",
                )
                print(
                    "\n\n-------------------------------------------------------------------------------- \n\n"
                )
                self.num_imputations = 1

            out_dir = os.join(self.params["root_dir"], f"data/{self.params["Run_Name"]}/imputations/")
            if not os.isdir(out_dir):
                os.makedirs(out_dir)
            
            subprocesses = []
            # Impute Scaled Data (multiple versions of)
            for i in range(self.num_samples):
                cmd = (self._instantiate_imputation, self.params.clean_data, self.params.data_imputation_methods,
                        self.params.impute_columns, i)
                subprocesses.append(cmd)
            # Handles Process scheduling
            function_scheduler(
                subprocesses, self.num_samples, max_workers=min(4, self.num_samples), out_message="SUCCESS: Imputation(s)", resilient=True
                )

        except (AttributeError, TypeError) as e:
            print(e)
    
    
    def _instantiate_imputation(self, data, impute_methods, impute_columns, i, out_dir): 
        """
        """
        my_iGen = iGen(data=data, impute_methods=impute_methods, impute_columns=impute_columns, id=i)
        my_iGen.fit()
        my_iGen.save_to_file()
