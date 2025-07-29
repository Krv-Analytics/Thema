# File: tests/multiverse/universe/stars/test_jmapStar.py
# Lasted Updated: 07-29-25
# Updated By: JW

import pickle
import networkx.utils as nxut


from thema.multiverse.universe.stars.jmapStar import jmapStar


class TestJmapStar:
    """Pytest Class for jmapStar"""

    def test_init_defaults(self, tmp_umapMoonAndData):
        data_file, moon_file, comet_file = tmp_umapMoonAndData
        jmapStar(
            data_path=data_file.name,
            clean_path=moon_file.name,
            projection_path=comet_file.name,
            nCubes=5,
            percOverlap=0.4,
            minIntersection=-1,
            clusterer=["HDBSCAN", {"min_cluster_size": 4}],
        )
        jmapStar(
            data_path=data_file.name,
            clean_path=moon_file.name,
            projection_path=comet_file.name,
            nCubes=4,
            percOverlap=0.4,
            minIntersection=-1,
            clusterer=["DBSCAN", {"eps": 0.4}],
        )

    def test_fit(self, tmp_umapMoonAndData):
        data_file, moon_file, comet_file = tmp_umapMoonAndData
        hdbscan_jmap = jmapStar(
            data_path=data_file.name,
            clean_path=moon_file.name,
            projection_path=comet_file.name,
            nCubes=4,
            percOverlap=0.8,
            minIntersection=-1,
            clusterer=["HDBSCAN", {"min_cluster_size": 2}],
        )
        dbscan_jmap = jmapStar(
            data_path=data_file.name,
            clean_path=moon_file.name,
            projection_path=comet_file.name,
            nCubes=4,
            percOverlap=0.8,
            minIntersection=-1,
            clusterer=["DBSCAN", {"eps": 0.4}],
        )

        hdbscan_jmap.fit()
        dbscan_jmap.fit()

        assert hdbscan_jmap.starGraph is not None
        assert dbscan_jmap.starGraph is not None

    def test_save(self, tmp_umapMoonAndData, tmp_file):
        data_file, moon_file, comet_file = tmp_umapMoonAndData
        jmap = jmapStar(
            data_path=data_file.name,
            clean_path=moon_file.name,
            projection_path=comet_file.name,
            nCubes=4,
            percOverlap=0.8,
            minIntersection=-1,
            clusterer=["HDBSCAN", {"min_cluster_size": 2}],
        )
        jmap.fit()
        jmap.save(tmp_file.name)

        with open(tmp_file.name, "rb") as f:
            jmap1 = pickle.load(f)

        assert jmap.get_data_path() == jmap1.get_data_path()
        assert jmap.get_clean_path() == jmap1.get_clean_path()
        assert jmap.get_projection_path() == jmap1.get_projection_path()
        assert jmap.nCubes == jmap1.nCubes
        assert jmap.percOverlap == jmap1.percOverlap
        assert jmap.minIntersection == jmap1.minIntersection
        if jmap.starGraph is not None:
            assert jmap1.starGraph is not None

    def test_determinism(self, tmp_umapMoonAndData, tmp_file):
        data_file, moon_file, comet_file = tmp_umapMoonAndData
        jmap0 = jmapStar(
            data_path=data_file.name,
            clean_path=moon_file.name,
            projection_path=comet_file.name,
            nCubes=4,
            percOverlap=0.8,
            minIntersection=-1,
            clusterer=["HDBSCAN", {"min_cluster_size": 2}],
        )

        jmap1 = jmapStar(
            data_path=data_file.name,
            clean_path=moon_file.name,
            projection_path=comet_file.name,
            nCubes=4,
            percOverlap=0.8,
            minIntersection=-1,
            clusterer=["HDBSCAN", {"min_cluster_size": 2}],
        )

        jmap0.fit()
        jmap1.fit()

        assert jmap0.complex == jmap1.complex
        assert nxut.graphs_equal(jmap0.starGraph.graph, jmap1.starGraph.graph)
