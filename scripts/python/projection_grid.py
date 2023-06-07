import json
import logging
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

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

    N_neighbors = params_json["projector_Nneighbors"]
    min_Dists = params_json["projector_minDists"]
    projector = params_json["projector"]

    projector_script = os.path.join(src, "processing/projecting/projector.py")
    log.info("Computing Projection Grid Search!")
    log.info(
        "--------------------------------------------------------------------------------"
    )
    log.info(f"Choice of Projector: {projector}")
    log.info(f"Choices for n_neighbors: {N_neighbors}")
    log.info(f"Choices for min_dist: {min_Dists}")
    log.info(
        "--------------------------------------------------------------------------------"
    )
    log.add()


    # Updating Projections Path in params.json
    params_json["projected_data"] = "data/" + params_json["Run_Name"] + "/projections/" + params_json["projector"] + "/"
    try:
        with open(JSON_PATH, "w") as f:
            json.dump(params_json, f, indent=4)
    except:
        print("There was a problem writing to your parameter file!")


    # Number of loops 
    num_loops = len(N_neighbors)*len(min_Dists)

    # Running Grid Search in Parallel 
    subprocesses = []
    ## GRID SEARCH PROJECTIONS
    for n in N_neighbors:
        for d in min_Dists:
            cmd = ["python", f"{projector_script}", f"-n {n}", f"-d {d}"]
            subprocesses.append(cmd)

    # Running processes in Parallel 
    # TODO: optimize based on max_workers 
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(subprocess.run, cmd) for cmd in subprocesses]
        # Setting Progress bar to track number of completed subprocesses 
        progress_bar = tqdm(total=num_loops, desc='Progress', unit='subprocess')
        for future in as_completed(futures):
        # Update the progress bar for each completed subprocess
            progress_bar.update(1)
    progress_bar.close()