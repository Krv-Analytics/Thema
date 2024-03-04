# File: src/fitting/gGen.py 
# Last Update: 03-04-24
# Updated by: SW 

import os
import sys
import pickle
from hdbscan import HDBSCAN
from pyballmapper import BallMapper

from .jmapper import jMapper
from .jgraph import jGraph
from .nammu.curvature import ollivier_ricci_curvature
from .fitting_utils import generate_gModel_filename
from .tupper import Tupper


class gGen: 
    """
    Graph Generator
    """

    def __init__(self, gen_method="jmap", **kwargs): 
        """
        Init File 
        """
        self.gen_method = gen_method

        if gen_method == "jmap": 
            # Data 
            self.raw = kwargs["raw"]
            self.clean = kwargs["clean"]
            self.projection = kwargs["projection"]

            # jMapper Arguments 
            self.nn = kwargs["nn"]
            self.minDist = kwargs["minDist"]
            self.percOverlap = kwargs["percOverlap"]
            self.dim = kwargs["dim"]
            self.minIntersection = kwargs["minIntersection"]

            # Clusterer Arguments
            self.clusterer = kwargs["clusterer"]
            self.clusterer_params = kwargs["hdbscan_params"]
        
        elif gen_method == "pyball": 
            self.pointCloud = ["pointCloud"]
            self.eps = kwargs["eps"]

        else: 
            print("Only 'jmap' and 'pyBall' are supported at this time.")
            raise
    
        # Initialize a `Tupper`
        self.tupper = Tupper(self.params.raw, self.params.clean, self.params.projection)

        self.n, self.p = self.params.jmap_nCubes, self.params.jmap_percOverlap
        self.min_intersections = self.params.jmap_minIntersection
        

    def fit(self):
        """
        Fit your Graph Model
        """

        if self.gen_method == "jmap":
            # HDBSCAN
            min_cluster_size, max_cluster_size = self.hdbscan_params
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                max_cluster_size=max_cluster_size,
            )
            # Configure JMapper
            jmapper = jMapper(self.tupper, self.nn, self.percOverlap, clusterer)

            if len(jmapper.complex["links"]) > 0:
                jmapper.min_intersection = self.minIntersection
                jmapper.jgraph = jGraph(jmapper.nodes, self.minIntersection)
                # Compute Curvature and Persistence Diagram
                if jmapper.jgraph.is_EdgeLess:
                    self.gModel = -1  # Empty Graph error code
                else:
                    jmapper.jgraph.curvature = ollivier_ricci_curvature
                    jmapper.jgraph.calculate_homology()
                    self.gModel = jmapper
            else:
                self.gModel -2  # Empty Simplicial Complex Code

        elif self.gen_method == "pyball":
            self.gModel = BallMapper(X=self.pointCloud, eps=self.eps) 



    def save_to_file(self, out_dir): 
        """Saves to a directory in output Directory"""
        if self.gen_method == "jmap": 
            
            output = {"jmapper": self.gModel}
        
            # CHECKING TYPE HERE
            if(type(self.gModel) == int):
                #TODO: Put in run log info - two errors     
                return 
            num_policy_groups = len(self.gModel.jgraph.components)
            if num_policy_groups > len(self.gModel.tupper.clean):
                # TODO: Improve Runlog tracking
                return
        

        output_file = generate_gModel_filename(
            args,
            nbors,
            d,
            min_intersection=args.min_intersection,
        )
    # Check if output directory already exists
    if os.path.isdir(out_dir):
        output_file = os.path.join(output_dir, output_file)
    else:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, output_file)

    output["hyperparameters"] = (
        n,
        p,
        nbors,
        d,
        hdbscan_params,
        args.min_intersection,
    )

    out_dir_message = output_file
    out_dir_message = "/".join(out_dir_message.split("/")[-2:])


    # Check for error codes from jmap_generator
    if jmapper == -1:
        print("EMPTY!")
        # runlog.log_emptyGraph_EVENT()
        # TODO: Write out the hyperparameter culprits

    elif jmapper == -2:
        # runlog.log_emptyComplex_EVENT()
        # TODO: Write out the hyperparameter culprits
        print("Empty Complex")
    else:
        with open(output_file, "wb") as handle:
            pickle.dump(
                output,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        if args.Verbose:
            print("\n")
            print(
                "-------------------------------------------------------------------------------------- \n\n"
            )
            print("Successfully generated `JMapper`.")
            print("Written to:")
            print(out_dir_message)

            print(
                "\n\n -------------------------------------------------------------------------------------- "
            )
