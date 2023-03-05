"Utility Functions"
import ot
from sklearn.cluster import KMeans

from mapper import MapperTopology
from nammu.curvature import ollivier_ricci_curvature


def curvature_analysis(
    X,
    n_cubes,
    perc_overlap,
    K,
    min_intersection_vals,
    random_state=None,
):

    # Configure CoalMapper
    mapper = MapperTopology(X=X)
    clusterer = KMeans(n_clusters=K, random_state=random_state, n_init=1)
    cover = (n_cubes, perc_overlap)
    # Generate Graphs
    results = {}

    print("Computing Curvature Values and Persistence Diagrams")
    for val in min_intersection_vals:
        mapper.set_graph(cover=cover, clusterer=clusterer, min_intersection=val)
        mapper.calculate_homology(filter_fn=ollivier_ricci_curvature, use_min=True)
        results[val] = mapper
    return results


def ot_metric(arr1, arr2):
    """Compare the curvature arrays between two mapper graphs using optimal transport"""
    # Cost Matrix
    M = ot.dist(arr1, arr2)
    d = ot.emd(arr1, arr2, M)
    # TODO: Maybe add in sinkhord regularisation parameter? Depends on how well this does
    return d


def generate_results_filename(args, suffix=".pkl"):
    """Generate output filename string from CLI arguments when running curvature_analysis script."""

    K, p, n = args.KMeans, args.perc_overlap, args.n_cubes

    output_file = f"results_ncubes{n}_{int(p*100)}perc_K{K}.pkl"

    return output_file
