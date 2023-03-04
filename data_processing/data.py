"""Pull Latest Data from MongoDB"""

import argparse
import os
import numpy as np
from dotenv import load_dotenv

from accessMongo import mongo_pull


if __name__ == "__main__":

    load_dotenv()
    client = os.getenv("mongo_client")
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--database",
        type=str,
        default="cleaned",
        help="Select database to pull from Mongo.",
    )
    parser.add_argument(
        "-c",
        "--col",
        type=str,
        default="coal_mapper",
        help="Select collection to pull from within `--database`.",
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="pkl",
        help="Select file type for local data.",
    )
    parser.add_argument(
        "-e",
        "--one_hot",
        type=bool,
        default=True,
        help="Option for categorical variables to be one hot encoded via `pd.dummies`.",
    )
    parser.add_argument(
        "-s",
        "--scaled",
        type=bool,
        default=True,
        help="Option to standardize data by removing the mean and scaling to unit variance.",
    )

    parser.add_argument(
        "-p",
        "--TSNE_project",
        type=bool,
        default=False,
        help="Option to project the data into two dimensions using TSNE",
    )

    args = parser.parse_args()

    cwd = os.path.dirname(__file__)
    output_dir = os.path.join(cwd, "local_data/")
    # If ./local_data/ doesn't exist yet, create it
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    file = mongo_pull(
        client,
        filepath=output_dir,
        database=args.database,
        one_hot=args.one_hot,
        scaled=args.scaled,
        TSNE_project=args.TSNE_project,
        type=args.type,
        col=args.col,
    )
    print(__file__)
    assert os.path.isfile(f"{file}.{args.type}"), "Failed to write data locally"
    print(
        f"Data successfully pulled from the `{args.col}` collection in the `{args.database}` Mongo database."
    )
    print(f"Written to: {file}")
