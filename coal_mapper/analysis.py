from coal_mapper.mapper import CoalMapper
from coal_mapper.utils import MapperTopology
from coal_mapper.nammu.curvature import ollivier_ricci_curvature
from data_processing.coal_mapper_dataset import getCompiledData, getEIA_allCoalData

from sklearn.cluster import KMeans
import pickle


def curvature_analysis(
    mongo_client,
    n_cubes,
    perc_overlap,
    K,
    min_intersection_vals,
):

    # Load data from mongo
    data = getCompiledData(mongo_client).to_numpy()

    # Configure CoalMapper
    mapper = MapperTopology(X=data)
    clusterer = KMeans(n_clusters=K, n_init="auto")
    cover = (n_cubes, perc_overlap)
    # Generate Graphs
    results = {}

    for val in min_intersection_vals:
        mapper.set_graph(cover=cover, clusterer=clusterer, min_intersection=val)
        mapper.calculate_homology(filter_fn=ollivier_ricci_curvature, use_min=True)
        results[val] = (
            mapper.graph,
            mapper.curvature,
            mapper.diagram,
        )
    output_dir = "../outputs/curvature/"
    output_file = f"results_ncubes{n_cubes}_{perc_overlap}perc_K{K}.pkl"

    with open(output_dir + output_file, "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
