# File: tests/multiverse/system/outer/test_oort.py
# Lasted Updated: 04-05-24
# Updated By: SW

import os
import pytest
import tempfile
from omegaconf import OmegaConf
from pandas.testing import assert_frame_equal

from thema.multiverse import Oort
from tests import test_utils as ut


class Test_oort:
    """Testing class for Oort"""

    def test_init_defaults(self, tmp_planetAndData, tmp_outDir, test_params1):
        tmp_dataFile, tmp_planetDir = tmp_planetAndData
        oort = Oort(
            params=test_params1,
            data=tmp_dataFile.name,
            cleanDir=tmp_planetDir,
            outDir=tmp_outDir,
        )

        assert_frame_equal(oort.data, ut._test_data_0)
        assert oort.cleanDir == tmp_planetDir
        assert oort.params == test_params1

    def test_init_erroneous(
        self,
        tmp_planetAndData,
        tmp_outDir,
        test_erroneous_params0,
    ):
        tmp_dataFile, tmp_planetDir = tmp_planetAndData
        with pytest.raises(AssertionError):
            Oort(
                params=test_erroneous_params0,
                data=tmp_dataFile.name,
                cleanDir=tmp_planetDir,
                outDir=tmp_outDir,
            )

    def test_yaml(self, temp_projYaml_0):
        oort = Oort(YAML_PATH=temp_projYaml_0.name)
        oort.fit()

    def test_fit(
        self,
        tmp_planetAndData,
        tmp_outDir,
        test_erroneous_params1,
        test_params1,
    ):
        tmp_dataFile, tmp_planetDir = tmp_planetAndData
        with pytest.raises(KeyError):
            oort = Oort(
                params=test_erroneous_params1,
                data=tmp_dataFile.name,
                cleanDir=tmp_planetDir,
                outDir=tmp_outDir,
            )
            oort.fit()

        oort = Oort(
            params=test_params1,
            data=tmp_dataFile.name,
            cleanDir=tmp_planetDir,
            outDir=tmp_outDir,
        )
        oort.fit()
        assert len(os.listdir(oort.outDir)) == 36  # Was 60 when UMAP was included

    def test_getParams(
        self,
        tmp_planetAndData,
        tmp_outDir,
        test_params1,
    ):
        tmp_dataFile, tmp_planetDir = tmp_planetAndData
        oort = Oort(
            params=test_params1,
            data=tmp_dataFile.name,
            cleanDir=tmp_planetDir,
            outDir=tmp_outDir,
        )

        oort2 = Oort(**oort.getParams())
        assert oort.get_data_path() == oort2.get_data_path()
        assert oort.cleanDir == oort2.cleanDir
        assert oort.outDir == oort2.outDir
        assert oort.params == oort2.params

    def test_writeParams_toYaml(
        self,
        tmp_planetAndData,
        tmp_outDir,
        test_params1,
        temp_projYaml_0,
    ):
        tmp_dataFile, tmp_planetDir = tmp_planetAndData
        oort = Oort(
            params=test_params1,
            data=tmp_dataFile.name,
            cleanDir=tmp_planetDir,
            outDir=tmp_outDir,
        )
        oort.writeParams_toYaml(temp_projYaml_0.name)
        oort2 = Oort(YAML_PATH=temp_projYaml_0.name)

        assert oort.params == oort2.params
