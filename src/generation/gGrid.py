import logging
import os
import warnings
import os
import sys

from dotenv import load_dotenv

from omegaconf import OmegaConf
from python_log_indenter import IndentedLoggerAdapter
from .utils import env
warnings.simplefilter("ignore")


################################################################################################
#   Loading Config Data
################################################################################################

root, src = env()  # Load .env
from .gridTracking_helper import log_error, function_scheduler


class gGrid: 
    
    def __init__(self, YAML_PATH):
        if os.path.isfile(YAML_PATH):
            with open(YAML_PATH, "r") as f:
                self.params = OmegaConf.load(f)
        else:
            print("params.yaml file note found!")

        # HDBSCAN
        self.min_cluster_size = self.params["jmap_min_cluster_size"]
        self.max_cluster_size = self.params["jmap_max_cluster_size"]

        # MAPPER
        self.n_cubes = self.params["jmap_nCubes"]
        self.perc_overlap = self.params["jmap_percOverlap"]
        self.min_intersection = self.params["jmap_minIntersection"]
        self.random_seed = self.params["jmap_random_seed"]

        # DATA
        self.raw = self.params["raw_data"]
        self.clean = self.params["clean_data"]
        self.projections = self.params["projected_data"]


        # Grid Files
        self.data_dir = os.path.join(self.params["root_dir"], "data") 
        # TODO: Use different graph generators (ie Ball Mapper)
        self.graph_generator = os.path.join(self.params["root_dir"], "src/jmapping/fitting/jmap_generator.py") 
        self.g_files = {}

    ################################################################################################
    #   Checking for necessary files
    ################################################################################################

        # Check that raw data exists
        if not os.path.isfile(os.path.join(self.raw)):
            log_error(
                "No raw data found. Please make sure you have specified the correct path in your self.params file."
            )

        # Check that clean data exits
        if not os.path.isfile(os.path.join(self.clean)):
            log_error(
                "No clean data found. Please make sure you generated clean data using `make process-data`."
            )

        # Check that Projections Exist
        if not os.path.isdir(os.path.join(self.projections)) or not os.listdir(
            os.path.join(self.projections)
        ):
            log_error(
                "No projections found. Please make sure you have generated projections using `make projections`."
            )



    def fit(self): 
        """ Fits your jmap grid search. """
    
    ################################################################################################
    #   Scheduling Subprocesses
    ################################################################################################

        # Number of loops
        num_loops = (
            len(self.n_cubes)
            * len(self.perc_overlap)
            * len(self.min_intersection)
            * len(self.min_cluster_size)
            * len(os.listdir(self.projections))
        )

        # Creating list of Subprocesses
        subprocesses = []
        ## GRID SEARCH PROJECTIONS
        for N in self.n_cubes:
            for P in self.perc_overlap:
                for I in self.min_intersection:
                    for C in self.min_cluster_size:
                        for file in os.listdir(self.projections):
                            if file.endswith(".pkl"):
                                D = os.path.join(self.projections, file)
                                subprocesses.append(
                                    [
                                        "python",
                                        f"{self.graph_generator}",
                                        f"-n{N}",
                                        f"-r{self.raw}",
                                        f"-c{self.clean}",
                                        f"-D{D}",
                                        f"-m{C}",
                                        f"-p {P}",
                                        f"-I {I}",
                                    ]
                                )

        # Handles Process scheduling
        subprocess_scheduler(
            subprocesses,
            num_loops,
            "SUCCESS: Completed JMAP generation grid.",
            resilient=True,
        )


        # Iterate through each directory in the base directory
        for dir_name in os.listdir(self.data_dir):
            # Check if the directory name ends with '_policy_group'
            if dir_name.endswith('_policy_groups') and os.path.isdir(os.path.join(self.data_dir, dir_name)):
                # Get the policy group number from the directory name
                policy_group_number = dir_name.split('_')[0]
                
                # List to store file paths
                file_paths_list = []
                
                # Iterate through files in the directory
                for file_name in os.listdir(os.path.join(self.data_dir, dir_name)):
                    # Construct full file path
                    file_path = os.path.join(self.data_dir, dir_name, file_name)
                    # Add file path to the list
                    file_paths_list.append(file_path)
                
                # Add list of file paths to the dictionary with policy group number as key
                self.g_files[policy_group_number] = file_paths_list

    def search_grid(self, ncubes=None, percOverlap=None, clusterer_nn=None, projector=None, Nbors=None, minDist=None ):
        " Search for files that match input parameter"
        ncubes_key = "n_cubes" + str(ncubes) if ncubes is not None else ""
        percOverlap_key = str(percOverlap) + "perc" if percOverlap is not None else ""
        clusterer_nn_key = "hdbscan" + str(clusterer_nn) if clusterer_nn is not None else ""
        projector_key = str(projector) if projector is not None else ""
        Nbors_key = str(Nbors) + "Nbors" if Nbors is not None else ""
        minDist_key = "minDist" + str(minDist) if minDist is not None else ""

        matching_files = []
        # Iterate over keys in the dictionary
        for key in self.g_files.keys():
            for file in self.g_files[key]:
                # Extract parameters from the key
                search_params = file.split('_')
                # Check if all specified parameters match
                if (ncubes is None or int(search_params[1]) == ncubes_key) and \
                (percOverlap is None or int(search_params[2]) == percOverlap_key) and \
                (clusterer_nn is None or int(search_params[3]) == clusterer_nn_key) and \
                (projector is None or int(search_params[4]) == projector_key) and \
                (Nbors is None or int(search_params[5]) == Nbors_key) and \
                (minDist is None or float(search_params[7]) == minDist_key):
                    # If all parameters match, return the corresponding file paths
                    matching_files.append(file)
            # If no matching file is found, return None
        return matching_files


if __name__ == "__main__":
    YAML_PATH = os.getenv("params")
    my_gGrid = gGrid(YAML_PATH=YAML_PATH)
    my_gGrid.fit()
