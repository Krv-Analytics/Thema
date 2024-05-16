# File: tests/multiverse/system/outer/test_umapProj.py
# Lasted Updated: 04-04-24
# Updated By: SW

import os
import pytest
import pickle
import random
import tempfile
import numpy as np
from tests import test_utils as ut
from pandas.testing import assert_frame_equal
from thema.multiverse.system.outer.projectiles.umapProj import umapProj


class Test_umapProj:
    """ """

    def test_init_empty(self):
        with pytest.raises(TypeError):
            umapProj()

    def test_init(self, tmp_moonAndData):
        tmp_dataFile, tmp_moon = tmp_moonAndData
        minDist = 0.1
        nn = 4
        dimensions = 2
        seed = 42
        x = umapProj(
            data_path=tmp_dataFile.name,
            clean_path=tmp_moon.name,
            minDist=minDist,
            nn=nn,
            dimensions=dimensions,
            seed=seed,
        )

        assert x.minDist == minDist
        assert_frame_equal(x.data, ut._test_data_0)
        assert_frame_equal(x.clean, ut._test_cleanData_0)
        assert x.get_data_path() == tmp_dataFile.name
        assert x.get_clean_path() == tmp_moon.name
        assert x.nn == nn
        assert x.dimensions == dimensions
        assert x.seed == seed

    def test_fit(self, tmp_moonAndData):
        tmp_dataFile, tmp_moon = tmp_moonAndData
        minDist = 0.1
        nn = 2
        dimensions = 2
        seed = 42
        x = umapProj(
            data_path=tmp_dataFile.name,
            clean_path=tmp_moon.name,
            minDist=minDist,
            nn=nn,
            dimensions=dimensions,
            seed=seed,
        )
        x.fit()

        assert x.minDist == minDist
        assert_frame_equal(x.data, ut._test_data_0)
        assert_frame_equal(x.clean, ut._test_cleanData_0)
        assert x.get_data_path() == tmp_dataFile.name
        assert x.get_clean_path() == tmp_moon.name
        assert x.nn == nn
        assert x.dimensions == dimensions
        assert x.seed == seed
        assert x.projectionArray.shape[1] == dimensions
        assert x.projectionArray.shape[0] == x.data.shape[0]

    def test_save(self, tmp_moonAndData):
        tmp_dataFile, tmp_moon = tmp_moonAndData
        minDist = 0.1
        nn = 2
        dimensions = 2
        seed = 42
        x = umapProj(
            data_path=tmp_dataFile.name,
            clean_path=tmp_moon.name,
            minDist=minDist,
            nn=nn,
            dimensions=dimensions,
            seed=seed,
        )
        x.fit()
        with tempfile.NamedTemporaryFile() as temp_file:
            x.save(temp_file.name)
            assert os.path.exists(temp_file.name)
            with open(temp_file.name, "rb") as f:
                y = pickle.load(f)
            assert np.array_equal(x.projectionArray, y.projectionArray)
            assert x.get_data_path() == y.get_data_path()
            assert x.get_clean_path() == y.get_clean_path()
            assert x.nn == y.nn
            assert x.minDist == y.minDist
            assert x.seed == y.seed
        temp_file.close()

    def test_determinism(self, tmp_moonAndData):
        tmp_dataFile, tmp_moon = tmp_moonAndData
        minDist = 0.1
        nn = 2
        dimensions = 2
        seed = random.randint(1, 100)
        x = umapProj(
            data_path=tmp_dataFile.name,
            clean_path=tmp_moon.name,
            minDist=minDist,
            nn=nn,
            dimensions=dimensions,
            seed=seed,
        )
        y = umapProj(
            data_path=tmp_dataFile.name,
            clean_path=tmp_moon.name,
            minDist=minDist,
            nn=nn,
            dimensions=dimensions,
            seed=seed,
        )

        x.fit()
        y.fit()

        assert np.array_equal(x.projectionArray, y.projectionArray)
        assert x.get_data_path() == y.get_data_path()
        assert x.get_clean_path() == y.get_clean_path()
        assert x.nn == y.nn
        assert x.minDist == y.minDist
        assert x.seed == y.seed
