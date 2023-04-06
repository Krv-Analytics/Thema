from hdbscan import HDBSCAN
from coal_mapper import CoalMapper


from nammu.curvature import ollivier_ricci_curvature


def coal_mapper_generator(
    tupper,
    n_cubes,
    perc_overlap,
    hdbscan_params,
    min_intersection_vals,
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
    coal_mapper = CoalMapper(tupper)
    coal_mapper.fit(n_cubes, perc_overlap, clusterer)

    results = {}

    if len(coal_mapper.complex["links"]) > 0:
        for val in min_intersection_vals:
            # Generate Graph
            coal_mapper.to_networkx(min_intersection=val)
            coal_mapper.connected_components()
            # Compute Curvature and Persistence Diagram
            coal_mapper.curvature = ollivier_ricci_curvature
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


def generate_mapper_filename(args, n_neighbors, min_dist):
    """Generate output filename string from CLI arguments when running  script."""

    min_cluster_size, p, n = (
        args.min_cluster_size,
        args.perc_overlap,
        args.n_cubes,
    )

    output_file = f"mapper_ncubes{n}_{int(p*100)}perc_hdbscan{min_cluster_size}_UMAP_{n_neighbors}Nbors_minD{min_dist}.pkl"

    return output_file
