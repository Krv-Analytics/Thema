from hdbscan import HDBSCAN
from coal_mapper import CoalMapper

# from nammu import ollivier_ricci_curvature


def coal_mapper_generator(
    data,
    projection,
    n_cubes,
    perc_overlap,
    hdbscan_params,
    min_intersection_vals,
    random_state=0,
    verbose=0,
):
    """ """

    # HDBSCAN
    min_cluster_size, max_cluster_size = hdbscan_params
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
    )

    # Configure CoalMapper
    coal_mapper = CoalMapper(data, projection)
    coal_mapper.fit(n_cubes, perc_overlap, clusterer)

    # Generate Graphs
    results = {}

    if len(coal_mapper.complex["links"]) > 0:
        print("Computing Curvature Values and Persistence Diagrams")
        for val in min_intersection_vals:
            coal_mapper.to_networkx(min_intersection=val)
            # coal_mapper.curvature = ollivier_ricci_curvature
            coal_mapper.calculate_homology()
            results[val] = coal_mapper
        return results
    else:
        if verbose:
            print(
                "-------------------------------------------------------------------------------- \n\n"
            )
            print(f"Empty Simplicial Complex. No file written")

            print(
                "\n\n -------------------------------------------------------------------------------- "
            )
        return results


def generate_results_filename(args, n_neighbors, min_dist):
    """Generate output filename string from CLI arguments when running  script."""

    min_cluster_size, p, n = (
        args.min_cluster_size,
        args.perc_overlap,
        args.n_cubes,
    )

    output_file = f"results_ncubes{n}_{int(p*100)}perc_hdbscan{min_cluster_size}_UMAP_{n_neighbors}Nbors_minD{min_dist}.pkl"

    return output_file
