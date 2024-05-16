# File: tests/multiverse/system/outer/test_comet.py
# Lasted Updated: 04-05-24
# Updated By: SW

import os
import pytest
import pickle
import tempfile
from tests import test_utils as ut
from pandas.testing import assert_frame_equal
from thema.multiverse.system.outer.comet import Comet


class Test_Comet:
    """
    Testing class for Comet
    """

    def test_init_empty(self):
        with pytest.raises(TypeError):
            Comet()

    def test_init(self, tmp_moonAndData):
        tmp_dataFile, tmp_moon = tmp_moonAndData
        comet = Comet(data_path=tmp_dataFile.name, clean_path=tmp_moon.name)
        assert_frame_equal(comet.data, ut._test_data_0)
        assert_frame_equal(comet.clean, ut._test_cleanData_0)
        assert comet.get_data_path() == tmp_dataFile.name
        assert comet.get_clean_path() == tmp_moon.name

    def test_fit(self, tmp_moonAndData):
        tmp_dataFile, tmp_moon = tmp_moonAndData
        comet = Comet(data_path=tmp_dataFile.name, clean_path=tmp_moon.name)
        with pytest.raises(NotImplementedError):
            comet.fit()

    def test_save(self, tmp_moonAndData):
        tmp_dataFile, tmp_moon = tmp_moonAndData
        comet = Comet(data_path=tmp_dataFile.name, clean_path=tmp_moon.name)
        with tempfile.NamedTemporaryFile(suffix=".pkl", mode="wb") as tmp_comet:
            comet.save(file_path=tmp_comet.name)
            with open(tmp_comet.name, "rb") as f:
                comet_postSave = pickle.load(f)
                assert_frame_equal(comet_postSave.data, ut._test_data_0)
                assert_frame_equal(comet_postSave.clean, ut._test_cleanData_0)
                assert comet.get_data_path() == tmp_dataFile.name
                assert comet.get_clean_path() == tmp_moon.name
        tmp_comet.close()
