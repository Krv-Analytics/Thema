"""Understand and compare the coal plants comprising seperate connected components"""

import argparse
import os
import sys
import numpy as np
import pickle
import pandas as pd

from dotenv import load_dotenv
from utils import curvature_iterator, generate_results_filename


cwd = os.path.dirname(__file__)

if __name__ == "__main__":

    load_dotenv()
    client = os.getenv("mongo_client")
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=os.path.join(cwd, "./../outputs/curvature/results_ncubes6_40perc_K8_coal_mapper_one_hot_scaled_TSNE.pkl"),
        help="Select location of local data set, as pulled from Mongo.",
    )
 
    args = parser.parse_args()
    this = sys.modules[__name__]

    assert os.path.isfile(args.path), "Invalid Input Data"
    # Load Dataframe
    with open(args.path, "rb") as f:
        print("Reading pickle file")
        df = pickle.load(f)

    TM = df[1]
    TM.populate_raw_data(mongo_client = client)
    #TM._raw_data.head()
    print(len(TM.mapper.connected_components()))
