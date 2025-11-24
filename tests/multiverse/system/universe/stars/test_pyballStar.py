# File: tests/multiverse/universe/stars/test_pyballStar.py
# Last Updated: 11-24-25
# Updated By: JW

import pickle
import networkx.utils as nxut


from thema.multiverse.universe.stars.pyballStar import pyballStar


class TestPyballStar:
    """Pytest Class for pyballStar"""

    def test_init_defaults(self, tmp_tsneMoonAndData):
        data_file, moon_file, comet_file = tmp_tsneMoonAndData
        pyballStar(
            data_path=data_file.name,
            clean_path=moon_file.name,
            projection_path=comet_file.name,
            EPS=0.1,
        )
        pyballStar(
            data_path=data_file.name,
            clean_path=moon_file.name,
            projection_path=comet_file.name,
            EPS=0.5,
        )

    def test_fit(self, tmp_tsneMoonAndData):
        data_file, moon_file, comet_file = tmp_tsneMoonAndData
        pyball1 = pyballStar(
            data_path=data_file.name,
            clean_path=moon_file.name,
            projection_path=comet_file.name,
            EPS=0.3,
        )
        pyball2 = pyballStar(
            data_path=data_file.name,
            clean_path=moon_file.name,
            projection_path=comet_file.name,
            EPS=0.5,
        )

        pyball1.fit()
        pyball2.fit()

        assert pyball1.starGraph is not None
        assert pyball2.starGraph is not None

    def test_save(self, tmp_tsneMoonAndData, tmp_file):
        data_file, moon_file, comet_file = tmp_tsneMoonAndData
        pyball = pyballStar(
            data_path=data_file.name,
            clean_path=moon_file.name,
            projection_path=comet_file.name,
            EPS=0.3,
        )
        pyball.fit()
        pyball.save(tmp_file.name)

        with open(tmp_file.name, "rb") as f:
            pyball1 = pickle.load(f)

        assert pyball.get_data_path() == pyball1.get_data_path()
        assert pyball.get_clean_path() == pyball1.get_clean_path()
        assert pyball.get_projection_path() == pyball1.get_projection_path()
        assert pyball.EPS == pyball1.EPS
        if pyball.starGraph is not None:
            assert pyball1.starGraph is not None

    def test_determinism(self, tmp_tsneMoonAndData, tmp_file):
        data_file, moon_file, comet_file = tmp_tsneMoonAndData
        pyball0 = pyballStar(
            data_path=data_file.name,
            clean_path=moon_file.name,
            projection_path=comet_file.name,
            EPS=0.3,
        )

        pyball1 = pyballStar(
            data_path=data_file.name,
            clean_path=moon_file.name,
            projection_path=comet_file.name,
            EPS=0.3,
        )

        pyball0.fit()
        pyball1.fit()

        assert pyball0.complex == pyball1.complex
        # Compare graph structure (nodes and edges)
        assert set(pyball0.starGraph.graph.nodes()) == set(
            pyball1.starGraph.graph.nodes()
        )
        assert set(pyball0.starGraph.graph.edges()) == set(
            pyball1.starGraph.graph.edges()
        )

    def test_get_pseudoLaplacian(self, tmp_tsneMoonAndData):
        data_file, moon_file, comet_file = tmp_tsneMoonAndData
        pyball = pyballStar(
            data_path=data_file.name,
            clean_path=moon_file.name,
            projection_path=comet_file.name,
            EPS=0.3,
        )
        pyball.fit()
        if pyball.starGraph is not None:
            # Test with 'node' neighborhood (default)
            laplacian_node = pyball.get_pseudoLaplacian(neighborhood="node")
            assert laplacian_node.shape == (len(pyball.clean), len(pyball.clean))
            assert laplacian_node.dtype == int
            # Diagonal should be non-negative (count of neighborhoods)
            assert (laplacian_node.diagonal() >= 0).all()
            # TODO: Add 'cc' neighborhood test once implementation issues are resolved

    def test_get_unclustered_items(self, tmp_tsneMoonAndData):
        data_file, moon_file, comet_file = tmp_tsneMoonAndData
        pyball = pyballStar(
            data_path=data_file.name,
            clean_path=moon_file.name,
            projection_path=comet_file.name,
            EPS=0.3,
        )
        pyball.fit()
        unclustered = pyball.get_unclustered_items()
        assert isinstance(unclustered, list)
