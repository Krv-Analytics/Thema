import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from dotenv import load_dotenv
from python_log_indenter import IndentedLoggerAdapter

load_dotenv()
root = os.getenv("root")
sys.path.append(root + "/run_logging/")

#from run_log import Run_Log


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
    projections = params_json["projected_data"]

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

    # Number of loops 
    num_loops = len(n_cubes)*len(perc_overlap)*len(min_intersection)*len(min_cluster_size) *len(os.listdir(os.path.join(root, projections)))
    
    # Instantiate a Run Log
    #runlog = Run_Log() 
    #runlog.start_model_log() 
    #runlog.set_model_gridSize(num_loops)
    #runlog.log_model_startTime() 

    # Running Grid in Parallel 
    subprocesses = []
    ## GRID SEARCH PROJECTIONS
    for N in n_cubes:
        for P in perc_overlap:
            for I in min_intersection:
                for C in min_cluster_size:
                    for file in os.listdir(os.path.join(root, projections)):
                        if file.endswith(".pkl"):
                            D = os.path.join(projections, file)
                            subprocesses.append(
                                [
                                    "python",
                                    f"{model_generator}",
                                    f"-n{N}",
                                    f"-r{raw}",
                                    f"-c{clean}",
                                    f"-D{D}",
                                    f"-m{C}",
                                    f"-p {P}",
                                    f"-I {I}"
                                ]
                            )
    
    #runlog.log_model_finishTime() # Logs finish time and sets total run time
    
    # Running processes in Parallel 
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(subprocess.run, cmd) for cmd in subprocesses]
        # Setting Progress bar to track number of completed subprocesses 
        progress_bar = tqdm(total=num_loops, desc='Progress', unit='subprocess')
        for future in as_completed(futures):
        # Update the progress bar for each completed subprocess
            progress_bar.update(1)
        progress_bar.close()
