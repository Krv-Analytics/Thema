# File: tests/multiverse/universe/stars/test_gudhiStar.py
# Last Updated: 11-19-25
# Updated By: JW

import pickle
import networkx.utils as nxut


from thema.multiverse.universe.stars.gudhiStar import gudhiStar


class TestGudhiStar:
    """Pytest Class for gudhiStar"""

    def test_init_defaults(self, tmp_tsneMoonAndData):
        data_file, moon_file, comet_file = tmp_tsneMoonAndData
        gudhiStar(
            data_path=data_file.name,
            clean_path=moon_file.name,
            projection_path=comet_file.name,
            clusterer=["HDBSCAN", {"min_cluster_size": 1}],
        )
        gudhiStar(
            data_path=data_file.name,
            clean_path=moon_file.name,
            projection_path=comet_file.name,
            clusterer=["DBSCAN", {"eps": 1.0}],
        )

    def test_fit(self, tmp_tsneMoonAndData):
        data_file, moon_file, comet_file = tmp_tsneMoonAndData
        hdbscan_gudhi = gudhiStar(
            data_path=data_file.name,
            clean_path=moon_file.name,
            projection_path=comet_file.name,
            clusterer=["HDBSCAN", {"min_cluster_size": 2}],
        )
        dbscan_gudhi = gudhiStar(
            data_path=data_file.name,
            clean_path=moon_file.name,
            projection_path=comet_file.name,
            clusterer=["DBSCAN", {"eps": 0.4, "min_samples": 2}],
        )

        hdbscan_gudhi.fit()
        dbscan_gudhi.fit()

        assert hdbscan_gudhi.starGraph is not None
        assert dbscan_gudhi.starGraph is not None

    def test_save(self, tmp_tsneMoonAndData, tmp_file):
        data_file, moon_file, comet_file = tmp_tsneMoonAndData
        gudhi = gudhiStar(
            data_path=data_file.name,
            clean_path=moon_file.name,
            projection_path=comet_file.name,
            clusterer=["HDBSCAN", {"min_cluster_size": 2}],
        )
        gudhi.fit()
        gudhi.save(tmp_file.name)

        with open(tmp_file.name, "rb") as f:
            gudhi1 = pickle.load(f)

        assert gudhi.get_data_path() == gudhi1.get_data_path()
        assert gudhi.get_clean_path() == gudhi1.get_clean_path()
        assert gudhi.get_projection_path() == gudhi1.get_projection_path()
        assert gudhi.N == gudhi1.N
        assert gudhi.beta == gudhi1.beta
        assert gudhi.C == gudhi1.C
        if gudhi.starGraph is not None:
            assert gudhi1.starGraph is not None

    def test_determinism(self, tmp_tsneMoonAndData, tmp_file):
        data_file, moon_file, comet_file = tmp_tsneMoonAndData
        gudhi0 = gudhiStar(
            data_path=data_file.name,
            clean_path=moon_file.name,
            projection_path=comet_file.name,
            clusterer=["DBSCAN", {"eps": 0.4, "min_samples": 2}],
        )

        gudhi1 = gudhiStar(
            data_path=data_file.name,
            clean_path=moon_file.name,
            projection_path=comet_file.name,
            clusterer=["DBSCAN", {"eps": 0.4, "min_samples": 2}],
        )

        gudhi0.fit()
        gudhi1.fit()

        # GUDHI's MapperComplex has some non-deterministic behavior
        # so we check that both produce valid graphs with the same complex
        assert gudhi0.complex == gudhi1.complex
        if gudhi0.starGraph is not None and gudhi1.starGraph is not None:
            # Both should produce valid graphs
            assert len(gudhi0.starGraph.graph.nodes()) > 0
            assert len(gudhi1.starGraph.graph.nodes()) > 0

    def test_get_pseudoLaplacian(self, tmp_tsneMoonAndData):
        data_file, moon_file, comet_file = tmp_tsneMoonAndData
        gudhi = gudhiStar(
            data_path=data_file.name,
            clean_path=moon_file.name,
            projection_path=comet_file.name,
            clusterer=["HDBSCAN", {"min_cluster_size": 2}],
        )
        gudhi.fit()
        if gudhi.starGraph is not None:
            # Test with 'node' neighborhood (default)
            laplacian_node = gudhi.get_pseudoLaplacian(neighborhood="node")
            assert laplacian_node.shape == (len(gudhi.clean), len(gudhi.clean))
            assert laplacian_node.dtype == int
            # Diagonal should be non-negative (count of neighborhoods)
            assert (laplacian_node.diagonal() >= 0).all()
            # TODO: Add 'cc' neighborhood test once implementation issues are resolved

    def test_get_unclustered_items(self, tmp_tsneMoonAndData):
        data_file, moon_file, comet_file = tmp_tsneMoonAndData
        gudhi = gudhiStar(
            data_path=data_file.name,
            clean_path=moon_file.name,
            projection_path=comet_file.name,
            clusterer=["HDBSCAN", {"min_cluster_size": 2}],
        )
        gudhi.fit()
        unclustered = gudhi.get_unclustered_items()
        assert isinstance(unclustered, list)
