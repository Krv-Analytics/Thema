import json
import logging
import os
import subprocess

from dotenv import load_dotenv
from python_log_indenter import IndentedLoggerAdapter

if __name__ == "__main__":

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    log = IndentedLoggerAdapter(logging.getLogger(__name__))
    load_dotenv()
    root = os.getenv("root")
    src = os.getenv("src")
    JSON_PATH = os.getenv("params")
    if os.path.isfile(JSON_PATH):
        with open(JSON_PATH, "r") as f:
            params_json = json.load(f)
    else:
        print("params.json file note found!")

    # HDBSCAN
    min_cluster_size = params_json["model_min_cluster_size"]
    max_cluster_size = params_json["model_max_cluster_size"]

    # MAPPER
    n_cubes = params_json["model_nCubes"]
    perc_overlap = params_json["model_percOverlap"]
    min_intersection = params_json["model_minIntersection"]
    random_seed = params_json["model_random_seed"]

    # DATA
    raw = params_json["raw_data"]
    clean = params_json["clean_data"]
    projections = params_json["path_to_projections"]

    model_generator = os.path.join(src, "modeling/model_generator.py")
    log.info("Computing Model Grid Search!")
    log.info(
        "--------------------------------------------------------------------------------"
    )
    log.info(f"Choices for min_cluster_size: {min_cluster_size}")
    log.info(f"Choices for n_cubes: {n_cubes}")
    log.info(f"Choices for perc_overlap: {perc_overlap}")
    log.info(f"Choices for min_intersection: {min_intersection}")
    log.info(
        "--------------------------------------------------------------------------------"
    )
    log.add()
    ## GRID SEARCH PROJECTIONS
    for N in n_cubes:
        for P in perc_overlap:
            for I in min_intersection:
                for file in os.listdir(os.path.join(root, projections)):
                    if file.endswith(".pkl"):
                        D = os.path.join(projections, file)
                        subprocess.run(
                            [
                                "python",
                                f"{model_generator}",
                                f"-n{N}",
                                f"-r{raw}",
                                f"-c{clean}",
                                f"-D{D}",
                                f"-m{min_cluster_size}",
                                f"-p {P}",
                                f"-I {I}",
                            ]
                        )
