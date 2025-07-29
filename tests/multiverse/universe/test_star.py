# File: tests/multiverse/universe/test_star.py
# Lasted Updated: 07/29/25
# Updated By: JW

import pytest
import pickle

from thema.multiverse.universe.star import Star


class TestStar:
    """PyTesting Class for Star"""

    def test_inits_empty(self):
        with pytest.raises(TypeError):
            Star()

    def test_inits_defaults(self, tmp_umapMoonAndData):
        tmp_dataFile, tmp_moon, tmp_projectile = tmp_umapMoonAndData
        Star(
            data_path=tmp_dataFile.name,
            clean_path=tmp_moon.name,
            projection_path=tmp_projectile.name,
        )

    def test_fit(self, tmp_umapMoonAndData):
        tmp_dataFile, tmp_moon, tmp_projectile = tmp_umapMoonAndData
        with pytest.raises(NotImplementedError):
            star = Star(
                data_path=tmp_dataFile.name,
                clean_path=tmp_moon.name,
                projection_path=tmp_projectile.name,
            )
            star.fit()

    def test_save(self, tmp_umapMoonAndData, tmp_file):
        tmp_dataFile, tmp_moon, tmp_projectile = tmp_umapMoonAndData
        star = Star(
            data_path=tmp_dataFile.name,
            clean_path=tmp_moon.name,
            projection_path=tmp_projectile.name,
        )
        star.starGraph = -1
        star.save(tmp_file.name, force=True)

        with open(tmp_file.name, "rb") as f:
            star1 = pickle.load(f)

        assert star1.get_data_path() == star.get_data_path()
        assert star1.get_clean_path() == star.get_clean_path()
        assert star1.get_projection_path() == star.get_projection_path()
        assert star1.starGraph == -1
