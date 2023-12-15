"Project cleaned data using UMAP."

import argparse
import os
import pickle
import sys
import time

######################################################################
# Silencing UMAP Warnings
import warnings

from numba import NumbaDeprecationWarning
from omegaconf import OmegaConf

warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="umap")

os.environ["KMP_WARNINGS"] = "off"
######################################################################

import os
import sys
from dotenv import load_dotenv
load_dotenv()
src = os.getenv("src")
root = os.getenv("root")
sys.path.append(src)
sys.path.append(src+root)

from populate_helper import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    YAML_PATH = os.getenv("params")
    if os.path.isfile(YAML_PATH):
        with open(YAML_PATH, "r") as f:
            params = OmegaConf.load(f)
    else:
        print("params.yaml file note found!")

    parser.add_argument(
        "-c",
        "--clean_data",
        type=str,
        default=os.path.join(root, params["clean_data"]),
        help="Location of Cleaned data set",
    )

    ############################################################

    args = parser.parse_args()
    this = sys.modules[__name__]

    '''
    Read in clean data
    '''
    assert os.path.isfile(args.clean_data), "\n Invalid path to Clean Data"
    # Load Dataframe
    with open(args.clean_data, "rb") as clean:
        reference = pickle.load(clean)
        df = reference["clean_data"]

    df = add_imputed_flags(df)

    '''
    Configure filepaths
    '''
    rel_outdir = "data/" + params["Run_Name"] + "/clean/"
    output_dir = os.path.join(root, rel_outdir)


    '''
    Set up the imputation & write files to `clean` dir
    '''
    fill_method = params.data_imputation.fill_method

    if fill_method == "Random_normal":
        iteration_number = params.data_imputation.number_for_random

        for num in range(1, iteration_number+1):
            imputed_data = perturbulate_data(df)

            file_name = populate_nas_filename(params.Run_Name, fill_method=fill_method, number=num)
            output_filepath = output_dir + file_name

            with open(output_filepath, "wb") as f:
                pickle.dump(imputed_data, f)

    elif fill_method == "Drop":

        imputed_data = perturbulate_data(df, fill_method='drop')
        file_name = populate_nas_filename(params.Run_Name, fill_method=fill_method)
        output_filepath = output_dir + file_name

        with open(output_filepath, "wb") as f:
            pickle.dump(imputed_data, f)


            





