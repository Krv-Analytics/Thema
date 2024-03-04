"Helper functions for Mapping Policy jmaps"

import os
import sys
import math

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from hdbscan import HDBSCAN

from .jmapper import JMapper
from .tupper import Tupper
from .jgraph import JGraph
from .nammu.curvature import ollivier_ricci_curvature


def jmap_generator(
    tupper,
    n_cubes,
    perc_overlap,
    hdbscan_params,
    min_intersection,
):
    """
    Fit a graph jmap using JMapper.

    Parameters
    -----------
    tupper: <tupper.Tupper>
            A data container that holds raw, cleaned, and projected
            versions of user data.

    n_cubes: int
            Number of cubes used to cover of the latent space.
            Used to construct a kmapper.Cover object.

    perc_overlap: float
        Percentage of intersection between the cubes covering
        the latent space. Used to construct a kmapper.Cover object.

    hdbscan_params: tuple
        (m:int,M:int) where `m` is min_cluster_size
        and `M` is max_cluster_size.

    min_interesection: int
        Min_intersection value for generating a networkx graph
        from a simplicial complex.

    verbose: bool, default is False
            If True, use kmapper logging levels.

    Returns
    -----------
    jmapper : <jmapper.JMapper>
        A jmapper object if associated simplicial complex and jgraph is non-empty
    --

    -1: Empty graph error code
        The min intersection value resulted in an edgeless graph

    --

    -2: Empty simplicial complex error code
        The parameters for the kmapper fitting resulted in an empty simplicial complex
    """
    #

    # HDBSCAN
    min_cluster_size, max_cluster_size = hdbscan_params
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
    )
    # Configure JMapper
    jmapper = JMapper(tupper, n_cubes, perc_overlap, clusterer)

    if len(jmapper.complex["links"]) > 0:
        jmapper.min_intersection = min_intersection
        jmapper.jgraph = JGraph(jmapper.nodes, min_intersection)
        # Compute Curvature and Persistence Diagram
        if jmapper.jgraph.is_EdgeLess:
            return -1  # Empty Graph error code
        else:
            jmapper.jgraph.curvature = ollivier_ricci_curvature
            jmapper.jgraph.calculate_homology()
            return jmapper
    else:
        return -2  # Empty Simplicial Complex Code


def generate_jmap_filename(args, n_neighbors, min_dist, min_intersection):
    """Generate output filename string from CLI arguments.
    A helper function for the `jmap_generator` script.

    Parameters
    -----------
    args: argparse.ArgumentParser
        CLI arguments from the `jmap_generator` script.

    n_neighbors: int
        Number of neighbors used in the UMAP projection used
        to fit a JMapper.

    min_dist: float
        Minimal distance between points in the UMAP projection used
        to fit a JMapper.

    min_intersection: int
        Minimum intersection considered when computing the graph.
        An edge will be created only when the intersection between
        two nodes is greater than or equal to `min_intersection`.

    Returns
    -----------
    output_file: str
        A unique filename to identify JMappers.

    """

    min_cluster_size, p, n = (
        args.min_cluster_size,
        args.perc_overlap,
        args.n_cubes,
    )

    output_file = f"mapper_ncubes{n}_{int(p*100)}perc_hdbscan{min_cluster_size}_UMAP_{n_neighbors}Nbors_minDist{min_dist}_min_int{min_intersection}.pkl"

    return output_file


# TODO: Remove this function
def env():
    """Load .env file and add necessary folders to your `sys` path."""
    load_dotenv()
    root = os.getenv("root")
    src = os.getenv("src")
    sys.path.append(root)
    sys.path.append(src)
    sys.path.append(src + "jmapping/")
    return root


# NOTE: Why do we need this?
def script_paths(paths):
    root = env()
    scripts_dir = os.path.join(root, "scripts/")
    new_paths = []
    for rel_path in paths:
        abs_path = os.path.join(scripts_dir, rel_path)
        new_paths.append(abs_path)
    return new_paths
